// 리포트 출력/저장/알림 — 부수효과는 하네스가 통제한다(에이전트엔 푸시 도구 없음).
import type { OpsReport } from './schema';

export function printReport(report: OpsReport, meta: { brain: string; toolCalls: string[] }): void {
  const line = '─'.repeat(48);
  console.log(`\n${line}\n운영 점검 리포트  (brain=${meta.brain})`);
  console.log(`도구 호출: ${meta.toolCalls.join(', ') || '(없음)'}`);
  console.log(`심각도 : ${report.severity}`);
  console.log(`헤드라인: ${report.headline}`);
  if (report.sections.length) {
    console.log('— 영역별 —');
    for (const s of report.sections) console.log(`  • ${s.title}: ${s.detail}`);
  }
  if (report.anomalies.length) {
    console.log('— 이상치 —');
    for (const a of report.anomalies) console.log(`  · (${a.severity}) ${a.summary}`);
  }
  if (report.recommendations.length) {
    console.log('— 권고 —');
    for (const r of report.recommendations) console.log(`  → ${r}`);
  }
  console.log(line);
}

/**
 * ops_agent_report 테이블에 저장(있으면). 테이블 미생성(DDL 미적용) 환경에서는
 * 조용히 건너뛴다 — 스키마 변경은 db/*.sql + DBeaver 로만 적용한다.
 */
export async function persistReport(
  report: OpsReport,
  meta: { brain: string; toolCalls: string[] },
): Promise<void> {
  try {
    const { prisma } = await import('@/lib/prisma');
    // run_at 은 UTC 로 저장한다. mariadb 드라이버가 DATETIME 을 Node 프로세스
    // 로컬(컨테이너=UTC) 기준으로 읽고, 대시보드(/api/ops-agent/reports)가
    // toISOString()→브라우저 toLocaleString(KST) 로 변환하므로, DB 로컬시각(KST)인
    // NOW(3) 로 저장하면 KST 값이 UTC 로 오해돼 화면에 +9h 가 더해진다. UTC_TIMESTAMP
    // 로 저장하면 "UTC 저장 → KST 표시" 가 일관되게 맞는다.
    await prisma.$executeRaw`
      INSERT INTO ops_agent_report (run_at, severity, headline, body_json, brain, tool_calls)
      VALUES (UTC_TIMESTAMP(3), ${report.severity}, ${report.headline},
              ${JSON.stringify(report)}, ${meta.brain}, ${meta.toolCalls.join(',')})`;
  } catch (e) {
    console.warn('[ops-agent] persist 건너뜀:', e instanceof Error ? e.message : String(e));
  }
}

/**
 * severity ≥ warning 이면 알림. dry-run(기본, OPS_AGENT_DRY_RUN≠0)에서는 발송하지 않고 로그만.
 * 실발송은 기존 ops 알림 인프라(lib/ops-alert)를 재사용 — throttle(ops_alert_log) +
 * notification_config 라우팅(기본 개발팀, /notification-settings 에서 변경·끄기 가능)을 그대로 따른다.
 */
export async function dispatchAlert(report: OpsReport, dryRun: boolean): Promise<void> {
  if (report.severity === 'info') return;
  if (dryRun) {
    console.log(`[ops-agent] (dry-run) 알림 보류: ${report.severity} — ${report.headline}`);
    return;
  }
  const { maybeSendOpsAlert } = await import('@/lib/ops-alert');
  // critical 은 자주 재알림(3h), warning 은 하루 1회 수준(12h)으로 throttle — 중복 발송 방지.
  const critical = report.severity === 'critical';
  const sent = await maybeSendOpsAlert({
    alertKey: critical ? 'ops_agent_critical' : 'ops_agent_warning',
    throttleMin: critical ? 180 : 720,
    event: 'ops_agent_alert',
    data: { severity: report.severity, summary: report.headline },
  });
  console.log(
    `[ops-agent] 알림 ${sent ? '발송' : '보류(throttle/실패)'}: ${report.severity} — ${report.headline}`,
  );
}
