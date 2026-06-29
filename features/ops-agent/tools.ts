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

async function getSalesSummary(): Promise<unknown> {
  // 실제 키오스크 매출(키오스크 매출조회 기준 A+B−C) — 설치 현장별.
  //   online_booking(온라인 사전결제)이 아니라, 기존 통계 페이지와 동일한 getSiteStats 를
  //   재사용한다(복잡한 매출 분해를 중복 구현하지 않음). null = 전 현장(권한 필터 없음).
  const { getSiteStats } = await import('@/features/delivery/queries/getSiteStats');
  const stats = await getSiteStats(null);
  const sites = stats.rows.map((r) => ({
    site: r.siteName,
    lockers: r.lockerCount,
    todayCount: r.todayCount, // 오늘 이용건수
    monthCount: r.monthCount, // 최근 30일 이용건수
    monthRevenue: r.monthRevenue, // 최근 30일 매출(원)
    totalRevenue: r.totalRevenue, // 전기간 누적 매출(원)
    longStorage: r.longStorageCount,
  }));
  return {
    basis: '키오스크 매출조회 기준(A+B−C), 설치 현장별',
    totalToday: stats.totalToday,
    totalMonth: stats.totalMonth,
    totalMonthRevenue: stats.totalMonthRevenue,
    totalAllRevenue: stats.totalAllRevenue,
    siteCount: sites.length,
    sites,
  };
}

/**
 * 결제정합성 점검 — 전부 읽기전용 집계(SELECT COUNT). 위반 "건수"만 보고하고
 * 데이터는 건드리지 않는다. 거짓경보를 피하려고 "명백한 위반"만 골랐다:
 *  - 취소레코드+payment_status=1 공존처럼 매출모델(A+B−C)상 정상인 패턴은 제외.
 *  - 레거시 노이즈를 피하려 키오스크 결제고아는 최근 7일로 한정(=실측상 0이 정상).
 * 각 점검은 독립 try/catch → 한 테이블/쿼리가 실패해도 나머지는 보고된다.
 */
async function getPaymentConsistency(): Promise<unknown> {
  type Sev = 'info' | 'warning' | 'critical';
  type Check = { key: string; label: string; severity: Sev; count: number | null; note?: string; error?: string };

  const scalar = (rows: { n: bigint | number }[]) => Number(rows[0]?.n ?? 0);
  const one = async (
    key: string,
    label: string,
    severity: Sev,
    run: () => Promise<number>,
    note?: string,
  ): Promise<Check> => {
    try {
      const count = await run();
      return { key, label, severity: count > 0 ? severity : 'info', count, note };
    } catch (e) {
      return { key, label, severity: 'info', count: null, error: e instanceof Error ? e.message : String(e) };
    }
  };

  const checks = await Promise.all([
    // ── online_booking(EasyPay 온라인 사전결제) — 깨끗한 상태기계라 불변식이 명확 ──
    one('ob_oversell', '온라인예약 오버셀(같은 칸 동시 점유)', 'critical', async () =>
      scalar(await prisma.$queryRaw<{ n: bigint }[]>`
        SELECT COUNT(*) AS n FROM (
          SELECT locker_id, box_id FROM online_booking
          WHERE status IN ('reserved','paid','opened')
            AND (status = 'opened' OR reservation_expires_at > NOW(3))
          GROUP BY locker_id, box_id HAVING COUNT(*) > 1
        ) t`)),
    one('ob_unswept_holds', '온라인예약 만료 미정리(스윕/cron 누락 의심)', 'warning', async () =>
      scalar(await prisma.$queryRaw<{ n: bigint }[]>`
        SELECT COUNT(*) AS n FROM online_booking
        WHERE status IN ('reserved','paid')
          AND reservation_expires_at IS NOT NULL
          AND reservation_expires_at < (NOW(3) - INTERVAL 15 MINUTE)`)),
    one('ob_paid_missing_pg', '결제완료인데 PG번호/승인번호 없음', 'warning', async () =>
      scalar(await prisma.$queryRaw<{ n: bigint }[]>`
        SELECT COUNT(*) AS n FROM online_booking
        WHERE status = 'paid' AND (approval_no IS NULL OR pg_cno IS NULL)`)),
    one('ob_refunded_missing_ts', '환불상태인데 환불시각 없음', 'warning', async () =>
      scalar(await prisma.$queryRaw<{ n: bigint }[]>`
        SELECT COUNT(*) AS n FROM online_booking
        WHERE status = 'refunded' AND refunded_at IS NULL`)),
    // ── 멤버십 동기화 정합성 전용 테이블 — 미해결 행 = 진짜 불일치(거짓경보 없음) ──
    one('membership_sync_open', '멤버십 동기화 미해결 이슈', 'warning', async () =>
      scalar(await prisma.$queryRaw<{ n: bigint }[]>`
        SELECT COUNT(*) AS n FROM membership_sync_issue WHERE resolved_at IS NULL`)),
    // ── 키오스크 결제고아: 최근 7일 성공결제인데 이용기록(use_log) 없음 ──
    one('kiosk_orphan_payment', '키오스크 결제고아(최근7일 성공결제에 이용기록 없음)', 'warning', async () =>
      scalar(await prisma.$queryRaw<{ n: bigint }[]>`
        SELECT COUNT(*) AS n
        FROM dklocker.use_log_payment p
        LEFT JOIN dklocker.use_log u
          ON u.locker_id = p.locker_id AND u.use_log_id = p.use_log_id
        WHERE p.payment_status = 1 AND p.amount > 0
          AND p.use_log_create_date >= UNIX_TIMESTAMP(NOW() - INTERVAL 7 DAY)
          AND u.use_log_id IS NULL`),
      '실측상 use_log_payment≈use_log 1:1 → 0이 정상. >0이면 결제 후 동기화 유실 의심.'),
  ]);

  const rank: Record<Sev, number> = { info: 0, warning: 1, critical: 2 };
  const worstSeverity = checks.reduce<Sev>(
    (w, c) => (rank[c.severity] > rank[w] ? c.severity : w),
    'info',
  );
  return {
    worstSeverity,
    issues: checks.filter((c) => (c.count ?? 0) > 0), // 0건 초과만(= 실제 위반)
    allChecks: checks, // 투명성: 전체 점검 목록(0건 포함)
  };
}

/**
 * 이중결제/고아결제 신호 — 읽기전용. 키오스크 단말의 이중결제 방지 가드가
 * 중앙 event_log 로 보내는 결제 이벤트를 키오스크별로 집계한다.
 *  - PAYMENT_VOID(300003) = 고아 카드승인 자동 망취소(이중결제 방지)가 작동한 횟수.
 *    단건은 정상 자동회수. 한 키오스크에서 빈발(≥임계)하면 그 단말이 이중결제를
 *    반복 유발(보드/네트워크 의심) → 경고.
 *  - PAYMENT_FAIL(300002) = 결제 실패. 자동취소 실패(void_failed=실제 환불 필요)가
 *    현재 이 코드로 뭉뚱그려 남으므로, >0 이면 조사 대상으로 표시(정밀 분리는 Phase 2).
 */
async function getDoubleChargeSignals(): Promise<unknown> {
  const VOID_CODE = 300003;
  const FAIL_CODE = 300002;
  // 이중결제는 드물지만 중요한 신호 → 군집을 놓치지 않게 2주 룩백.
  const windowDays = 14;
  const spikeThreshold = 3; // 14일 내 자동취소 ≥3회 = 그 키오스크 이상 신호(실제 현장 이중결제 군집 사례에서 도출)

  const rows = await prisma.$queryRaw<
    { locker_id: number | bigint; locker_name: string | null; voids: bigint; fails: bigint; latest: bigint | null }[]
  >`
    SELECT e.locker_id,
           l.locker_name,
           SUM(e.event_log_code = ${VOID_CODE}) AS voids,
           SUM(e.event_log_code = ${FAIL_CODE}) AS fails,
           MAX(e.event_log_create_date)         AS latest
    FROM event_log e
    LEFT JOIN locker l ON l.locker_id = e.locker_id
    WHERE e.event_log_code IN (${VOID_CODE}, ${FAIL_CODE})
      AND e.event_log_create_date >= UNIX_TIMESTAMP(NOW() - INTERVAL 14 DAY)
    GROUP BY e.locker_id, l.locker_name
    ORDER BY voids DESC, fails DESC`;

  const perKiosk = rows.map((r) => {
    const voids = Number(r.voids ?? 0);
    const fails = Number(r.fails ?? 0);
    // Phase 1 은 critical 없이 보수적으로 — void 는 자동회수됨(돈 안 잃음), fail 은
    // 정상거절과 void_failed 가 섞여 있어 단정 못 함. 둘 다 "조사 필요" warning.
    const severity: 'info' | 'warning' = voids >= spikeThreshold || fails > 0 ? 'warning' : 'info';
    return { lockerId: Number(r.locker_id), site: r.locker_name, voids, fails, severity };
  });

  const issues = perKiosk.filter((k) => k.severity !== 'info');
  return {
    windowDays,
    spikeThreshold,
    note:
      'PAYMENT_VOID 단건=정상 자동회수(돈 안 잃음). 한 키오스크 빈발(≥임계)=그 단말 이중결제 반복 유발(하드웨어 점검). PAYMENT_FAIL>0=자동취소 실패 가능(수동 환불 조사). 정밀 분리는 void_failed 전용 이벤트(Phase 2) 필요.',
    worstSeverity: issues.length ? 'warning' : 'info',
    totals: {
      voids: perKiosk.reduce((s, k) => s + k.voids, 0),
      fails: perKiosk.reduce((s, k) => s + k.fails, 0),
    },
    issues,
    perKiosk,
  };
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
      'ML 모델 입력 분포의 드리프트(PSI, 레퍼런스 vs 현재)를 피처별로 조회한다. drifted=true 면 성능 저하 신호 → 재학습/점검 검토. ' +
      '단 sufficient_history=false 면 기준선(baseline)이 아직 부족한 것이므로(baseline_hours 참고) 드리프트 판정을 하지 말고 "기준선 수집 중"으로만 보고할 것.',
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
    description:
      '설치 현장별 실매출(키오스크 매출조회 기준 A+B−C)과 이용건수를 조회한다. 온라인 사전결제(online_booking)가 아니라 실제 키오스크 매출이며, 현장별 today/30일/누적 매출·건수를 포함한다 → 매출이 평소와 다른(예: 30일 매출은 있는데 오늘 0건) 현장 파악에 사용.',
    input_schema: { type: 'object', properties: {} },
  },
  {
    name: 'get_payment_consistency',
    description:
      '결제정합성 점검(읽기전용). 온라인예약 오버셀/만료미정리/PG번호누락/환불불일치, 멤버십 동기화 미해결, 키오스크 결제고아 등 "명백한 위반" 건수를 집계한다. worstSeverity 와 issues(0건 초과만)를 보고에 반영하라. issues 가 비어있으면 결제정합성 정상.',
    input_schema: { type: 'object', properties: {} },
  },
  {
    name: 'get_double_charge_signals',
    description:
      '카드 이중결제/고아결제 신호(읽기전용). 키오스크 방지 인프라가 중앙 event_log 로 보내는 PAYMENT_VOID(고아 자동취소)·PAYMENT_FAIL 을 키오스크별로 집계한다. 특정 키오스크에서 PAYMENT_VOID 빈발(issues) = 그 단말이 이중결제 반복 유발(하드웨어 점검 필요). PAYMENT_FAIL>0 = 자동취소 실패 가능(수동 환불 조사). issues 가 비어있으면 이중결제 신호 정상.',
    input_schema: { type: 'object', properties: {} },
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
      return safe(() => getSalesSummary());
    case 'get_payment_consistency':
      return safe(() => getPaymentConsistency());
    case 'get_double_charge_signals':
      return safe(() => getDoubleChargeSignals());
    default:
      return { error: `unknown tool: ${name}` };
  }
}
