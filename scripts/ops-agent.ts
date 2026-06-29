// 운영 모니터링 에이전트 진입점 (cron 1회 실행).
//
//   ┌──────────────────────────────────────────────────────────────┐
//   │ 1. 두뇌 선택: OPS_AGENT_BRAIN(anthropic|mock).                 │
//   │    미지정 시 ANTHROPIC_API_KEY 있으면 anthropic, 없으면 mock.  │
//   │ 2. 두뇌가 읽기전용 도구로 운영 상태를 점검 → 구조화 리포트.    │
//   │ 3. 출력 + (있으면) 저장 + severity≥warning 시 알림(dry-run 기본)│
//   └──────────────────────────────────────────────────────────────┘
//
// 실행: npm run ops-agent:once   (mock 기본, API 키 있으면 anthropic)
//   ML 서비스가 떠 있어야 get_ml_anomalies 가 동작: services/ml-anomaly (uvicorn :8200)
import { buildSystemPrompt } from '@/features/ops-agent/prompt';
import { opsTools, executeTool } from '@/features/ops-agent/tools';
import {
  AnthropicBrain,
  MockBrain,
  OllamaBrain,
  FallbackBrain,
  type Brain,
} from '@/features/ops-agent/brains';
import { printReport, persistReport, dispatchAlert } from '@/features/ops-agent/report';

/**
 * 두뇌 폴백 체인 구성. 기본 선택이 실패하면 다음으로 자동 폴백한다.
 *   ollama    → [ollama, anthropic(키 있으면), mock]
 *   anthropic → [anthropic, mock]
 *   mock      → [mock]
 * 미지정 시 키 있으면 anthropic, 없으면 mock.
 */
function buildBrain(): Brain {
  const hasKey = !!process.env.ANTHROPIC_API_KEY;
  const choice = process.env.OPS_AGENT_BRAIN ?? (hasKey ? 'anthropic' : 'mock');

  const chain: Brain[] = [];
  if (choice === 'ollama') {
    chain.push(new OllamaBrain());
    if (hasKey) chain.push(new AnthropicBrain());
    chain.push(new MockBrain());
  } else if (choice === 'anthropic') {
    chain.push(new AnthropicBrain());
    chain.push(new MockBrain());
  } else {
    chain.push(new MockBrain());
  }
  return chain.length === 1 ? chain[0] : new FallbackBrain(chain);
}

async function main(): Promise<void> {
  // 부수효과 안전: 기본 dry-run. 실발송은 OPS_AGENT_DRY_RUN=0 으로 명시.
  const dryRun = process.env.OPS_AGENT_DRY_RUN !== '0';
  const brain = buildBrain();
  console.log(`[ops-agent] brain=${brain.name} dryRun=${dryRun}`);

  const { report, toolCalls, usage, usedBrain } = await brain.run({
    system: buildSystemPrompt(),
    dataTools: opsTools,
    execute: executeTool,
  });
  const actual = usedBrain ?? brain.name;

  printReport(report, { brain: actual, toolCalls });
  if (usage) console.log('[ops-agent] usage:', JSON.stringify(usage));

  await persistReport(report, { brain: actual, toolCalls });
  await dispatchAlert(report, dryRun);
}

main()
  .then(() => process.exit(0))
  .catch((e) => {
    console.error('[ops-agent] 실패:', e);
    process.exit(1);
  });
