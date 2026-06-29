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
    await prisma.$executeRaw`
      INSERT INTO ops_agent_report (run_at, severity, headline, body_json, brain, tool_calls)
      VALUES (NOW(3), ${report.severity}, ${report.headline},
              ${JSON.stringify(report)}, ${meta.brain}, ${meta.toolCalls.join(',')})`;
  } catch (e) {
    console.warn('[ops-agent] persist 건너뜀:', e instanceof Error ? e.message : String(e));
  }
}

/**
 * severity ≥ warning 이면 알림. dry-run(기본)에서는 발송하지 않고 로그만.
 * 실제 푸시는 lib/push-events.ts 에 ops_agent_alert 이벤트 추가 후 연결(후속 작업).
 */
export async function dispatchAlert(report: OpsReport, dryRun: boolean): Promise<void> {
  if (report.severity === 'info') return;
  if (dryRun) {
    console.log(`[ops-agent] (dry-run) 알림 보류: ${report.severity} — ${report.headline}`);
    return;
  }
  console.log('[ops-agent] 실발송 미연결(ops_agent_alert 이벤트 추가 필요).');
}
