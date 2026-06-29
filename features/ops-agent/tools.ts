// 운영 에이전트의 "도구 표면" — 전부 읽기전용.
// 쓰기 도구는 존재하지 않는다(=하네스 레벨에서 "읽고 보고만" 보장).
// DB 도구는 try/catch 로 graceful-degrade: DATABASE_URL 미설정/네트워크 불가여도
// {available:false} 를 돌려주고 에이전트는 계속 진행한다.
import { prisma } from '@/lib/prisma';

export interface ToolDef {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

const ML_ANOMALY_URL = process.env.ML_ANOMALY_URL ?? 'http://127.0.0.1:8200';

/** BigInt/Date/Decimal 을 JSON 직렬화 가능한 값으로 깊은 변환. */
function jsonSafe(v: unknown): unknown {
  if (v === null || v === undefined) return v;
  if (typeof v === 'bigint') return Number(v);
  if (v instanceof Date) return v.toISOString();
  if (Array.isArray(v)) return v.map(jsonSafe);
  if (typeof v === 'object') {
    const o = v as Record<string, unknown>;
    if (typeof (o as { toNumber?: unknown }).toNumber === 'function') {
      return (o as { toNumber: () => number }).toNumber(); // Prisma Decimal
    }
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(o)) out[k] = jsonSafe(o[k]);
    return out;
  }
  return v;
}

async function safe(fn: () => Promise<unknown>): Promise<unknown> {
  try {
    return jsonSafe(await fn());
  } catch (e) {
    return { available: false, error: e instanceof Error ? e.message : String(e) };
  }
}

// ── 개별 executor ──────────────────────────────────────────────

/** L2 ML 이상탐지 서비스를 호출(에이전트↔ML 결합 지점). */
async function getMlAnomalies(input: { window_hours?: number }): Promise<unknown> {
  const w = input.window_hours ?? 72;
  const res = await fetch(`${ML_ANOMALY_URL}/anomalies?window_hours=${w}`);
  if (!res.ok) throw new Error(`ml-anomaly HTTP ${res.status}`);
  return res.json();
}

/** L5: ML 입력 분포 드리프트(PSI, 레퍼런스 vs 현재)를 조회한다. */
async function getModelDrift(input: { window_hours?: number }): Promise<unknown> {
  const w = input.window_hours ?? 168;
  const res = await fetch(`${ML_ANOMALY_URL}/drift?window_hours=${w}`);
  if (!res.ok) throw new Error(`ml-anomaly drift HTTP ${res.status}`);
  return res.json();
}

async function getBackupStatus(): Promise<unknown> {
  const recent = await prisma.$queryRaw<Record<string, unknown>[]>`
    SELECT id, started_at, finished_at, status, error_msg, size_bytes
    FROM db_backup_log ORDER BY started_at DESC LIMIT 5`;
  const lastOk = await prisma.$queryRaw<{ started_at: Date }[]>`
    SELECT started_at FROM db_backup_log WHERE status = 'success'
    ORDER BY started_at DESC LIMIT 1`;
  let hoursSinceLastSuccess: number | null = null;
  if (lastOk[0]?.started_at) {
    hoursSinceLastSuccess =
      (Date.now() - new Date(lastOk[0].started_at).getTime()) / 3_600_000;
  }
  return {
    recent,
    hoursSinceLastSuccess,
    overdue: hoursSinceLastSuccess !== null && hoursSinceLastSuccess > 25,
  };
}

async function getSystemMetrics(): Promise<unknown> {
  const row = await prisma.system_metrics.findFirst({ orderBy: { collected_at: 'desc' } });
  if (!row) return { available: false, error: 'no system_metrics rows' };
  const num = (v: unknown) => (v == null ? null : Number(v));
  const diskTotal = num(row.disk_total_bytes);
  const diskFree = num(row.disk_free_bytes);
  const memTotal = num(row.mem_total_bytes);
  const memFree = num(row.mem_free_bytes);
  return {
    collected_at: row.collected_at,
    disk_used_pct: diskTotal ? 1 - (diskFree ?? 0) / diskTotal : null,
    mem_used_pct: memTotal ? 1 - (memFree ?? 0) / memTotal : null,
    load1: num(row.load1),
    load5: num(row.load5),
    load15: num(row.load15),
  };
}

async function getDbHealth(): Promise<unknown> {
  const size = await prisma.$queryRaw<{ mb: unknown; cnt: unknown }[]>`
    SELECT ROUND(SUM(data_length + index_length) / 1024 / 1024, 1) AS mb, COUNT(*) AS cnt
    FROM information_schema.tables WHERE table_schema = DATABASE()`;
  const conn = await prisma.$queryRaw<{ connected: unknown; max_conn: unknown }[]>`
    SELECT
      (SELECT variable_value FROM information_schema.global_status
        WHERE variable_name = 'THREADS_CONNECTED') AS connected,
      @@max_connections AS max_conn`;
  return {
    sizeMb: Number(size[0]?.mb ?? 0),
    tableCount: Number(size[0]?.cnt ?? 0),
    connectionsUsed: Number(conn[0]?.connected ?? 0),
    connectionsMax: Number(conn[0]?.max_conn ?? 0),
  };
}

async function getSalesSummary(input: { days?: number }): Promise<unknown> {
  const days = input.days ?? 7;
  const rows = await prisma.$queryRaw<Record<string, unknown>[]>`
    SELECT DATE(created_at) AS d, COUNT(*) AS cnt,
           COALESCE(SUM(amount), 0) AS revenue,
           SUM(refunded_at IS NOT NULL) AS refunds
    FROM online_booking
    WHERE created_at >= (NOW() - INTERVAL ${days} DAY)
    GROUP BY DATE(created_at) ORDER BY d DESC`;
  return { days, daily: rows };
}

// ── 도구 정의(에이전트에 노출) + 디스패처 ───────────────────────

export const opsTools: ToolDef[] = [
  {
    name: 'get_ml_anomalies',
    description:
      'ML 이상탐지 서비스에서 최근 운영 시계열(시스템 메트릭·예약/매출)의 이상치 피드를 가져온다. severity 순으로 정렬됨.',
    input_schema: {
      type: 'object',
      properties: { window_hours: { type: 'integer', description: '조회 기간(시간), 기본 72' } },
    },
  },
  {
    name: 'get_model_drift',
    description:
      'ML 모델 입력 분포의 드리프트(PSI, 레퍼런스 vs 현재)를 피처별로 조회한다. drifted=true 면 성능 저하 신호 → 재학습/점검 검토.',
    input_schema: {
      type: 'object',
      properties: { window_hours: { type: 'integer', description: '조회 기간(시간), 기본 168' } },
    },
  },
  {
    name: 'get_backup_status',
    description: 'DB 백업 최근 이력과 마지막 성공 후 경과시간(overdue 여부)을 조회한다.',
    input_schema: { type: 'object', properties: {} },
  },
  {
    name: 'get_system_metrics',
    description: '최신 호스트 메트릭(디스크/메모리 사용률, load)을 조회한다.',
    input_schema: { type: 'object', properties: {} },
  },
  {
    name: 'get_db_health',
    description: 'DB 크기·테이블 수·커넥션 사용량 vs 최대치를 조회한다.',
    input_schema: { type: 'object', properties: {} },
  },
  {
    name: 'get_sales_summary',
    description: '최근 N일 온라인 예약 매출·건수·환불 일별 요약을 조회한다.',
    input_schema: {
      type: 'object',
      properties: { days: { type: 'integer', description: '기본 7' } },
    },
  },
];

export async function executeTool(name: string, input: Record<string, unknown>): Promise<unknown> {
  switch (name) {
    case 'get_ml_anomalies':
      return safe(() => getMlAnomalies(input));
    case 'get_model_drift':
      return safe(() => getModelDrift(input));
    case 'get_backup_status':
      return safe(() => getBackupStatus());
    case 'get_system_metrics':
      return safe(() => getSystemMetrics());
    case 'get_db_health':
      return safe(() => getDbHealth());
    case 'get_sales_summary':
      return safe(() => getSalesSummary(input));
    default:
      return { error: `unknown tool: ${name}` };
  }
}
