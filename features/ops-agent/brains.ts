// "두뇌" 추상화 — 모델을 갈아끼우는 부품으로 취급한다.
//  - AnthropicBrain: Claude Opus 4.8 tool-use 루프(실제).
//  - MockBrain: 결정론적. 모든 데이터 도구를 1회 호출해 리포트를 합성한다.
//    → API 키 없이 도구 실행·L2 연결·리포트 조립을 검증할 수 있다.
// (나중에 OllamaBrain 추가 가능 — 같은 인터페이스.)
import type Anthropic from '@anthropic-ai/sdk';
import { opsReportSchema, opsReportJsonSchema, type OpsReport, type Severity } from './schema';
import type { ToolDef } from './tools';

export interface BrainInput {
  system: string;
  dataTools: ToolDef[];
  execute: (name: string, input: Record<string, unknown>) => Promise<unknown>;
}

export interface BrainResult {
  report: OpsReport;
  toolCalls: string[];
  usage?: unknown;
  /** 폴백 체인에서 실제로 리포트를 만든 두뇌 이름. */
  usedBrain?: string;
}

export interface Brain {
  name: string;
  run(input: BrainInput): Promise<BrainResult>;
}

const SUBMIT_TOOL: ToolDef = {
  name: 'submit_report',
  description: '운영 점검 결과를 구조화된 리포트로 제출하고 종료한다. 반드시 마지막에 호출.',
  input_schema: opsReportJsonSchema as unknown as Record<string, unknown>,
};

const MAX_STEPS = Number(process.env.OPS_AGENT_MAX_STEPS ?? 12);

export class AnthropicBrain implements Brain {
  name = 'anthropic';

  async run({ system, dataTools, execute }: BrainInput): Promise<BrainResult> {
    const { default: AnthropicSDK } = await import('@anthropic-ai/sdk');
    const client = new AnthropicSDK();
    const model = process.env.OPS_AGENT_MODEL ?? 'claude-opus-4-8';

    const tools: Anthropic.Tool[] = [...dataTools, SUBMIT_TOOL].map((t) => ({
      name: t.name,
      description: t.description,
      input_schema: t.input_schema as Anthropic.Tool['input_schema'],
    }));

    const messages: Anthropic.MessageParam[] = [
      { role: 'user', content: '운영 상태를 점검하고 submit_report 로 리포트를 제출하세요.' },
    ];
    const toolCalls: string[] = [];

    for (let step = 0; step < MAX_STEPS; step++) {
      const resp = await client.messages.create({
        model,
        max_tokens: 8000,
        thinking: { type: 'adaptive' },
        system,
        tools,
        messages,
      });

      const toolUses = resp.content.filter(
        (b): b is Anthropic.ToolUseBlock => b.type === 'tool_use',
      );
      if (toolUses.length === 0) {
        throw new Error('에이전트가 submit_report 없이 종료함');
      }
      messages.push({ role: 'assistant', content: resp.content });

      const submit = toolUses.find((b) => b.name === 'submit_report');
      if (submit) {
        return { report: opsReportSchema.parse(submit.input), toolCalls, usage: resp.usage };
      }

      const results: Anthropic.ToolResultBlockParam[] = [];
      for (const tu of toolUses) {
        toolCalls.push(tu.name);
        const out = await execute(tu.name, (tu.input ?? {}) as Record<string, unknown>);
        results.push({
          type: 'tool_result',
          tool_use_id: tu.id,
          content: JSON.stringify(out),
        });
      }
      messages.push({ role: 'user', content: results });
    }
    throw new Error(`최대 단계(${MAX_STEPS}) 초과`);
  }
}

export class MockBrain implements Brain {
  name = 'mock';

  async run({ dataTools, execute }: BrainInput): Promise<BrainResult> {
    const toolCalls: string[] = [];
    const gathered: Record<string, unknown> = {};
    for (const t of dataTools) {
      toolCalls.push(t.name);
      gathered[t.name] = await execute(t.name, {});
    }

    const ml = gathered['get_ml_anomalies'] as
      | { items?: { severity: Severity; summary: string }[]; counts?: Record<string, number> }
      | undefined;

    const anomalies: OpsReport['anomalies'] = [];
    let severity: Severity = 'info';
    if (ml && Array.isArray(ml.items)) {
      for (const it of ml.items.slice(0, 8)) {
        anomalies.push({ severity: it.severity, summary: it.summary });
      }
      if ((ml.counts?.critical ?? 0) > 0) severity = 'critical';
      else if ((ml.counts?.warning ?? 0) > 0) severity = 'warning';
    }

    const sections = dataTools.map((t) => ({
      title: t.name,
      detail: summarize(gathered[t.name]),
    }));

    const report: OpsReport = {
      severity,
      headline:
        severity === 'critical'
          ? `심각 이상 ${ml?.counts?.critical ?? 0}건 감지(ML)`
          : severity === 'warning'
            ? `주의 항목 ${ml?.counts?.warning ?? 0}건`
            : '특이사항 없음',
      sections,
      anomalies,
      recommendations:
        severity === 'info' ? [] : ['상위 이상치부터 담당자 확인 및 원인 점검'],
    };
    return { report, toolCalls };
  }
}

export class OllamaBrain implements Brain {
  name = 'ollama';

  async run({ system, dataTools, execute }: BrainInput): Promise<BrainResult> {
    const baseUrl = (process.env.OLLAMA_BASE_URL ?? 'http://127.0.0.1:11434').replace(/\/$/, '');
    const model = process.env.OLLAMA_MODEL ?? 'gemma4:e4b';
    const numCtx = Number(process.env.OLLAMA_NUM_CTX ?? 32768);

    // Tools-on-rails: a local ~8B model isn't reliable at orchestrating a
    // multi-step tool loop, so the harness gathers all read-only data first and
    // the model only does reasoning/synthesis. (Same gather step as MockBrain,
    // but a real local model writes the report — free + private.)
    const toolCalls: string[] = [];
    const gathered: Record<string, unknown> = {};
    for (const t of dataTools) {
      toolCalls.push(t.name);
      gathered[t.name] = await execute(t.name, {});
    }

    const prompt = [
      '아래는 운영 점검 도구들이 수집한 데이터(JSON)입니다. 종합해 리포트를 작성하세요.',
      '결과가 {"available":false}인 도구는 "확인 불가"로 처리하세요.',
      '',
      '```json',
      JSON.stringify(gathered, null, 2),
      '```',
    ].join('\n');

    const report = await this.generate(baseUrl, model, numCtx, system, prompt);
    return { report, toolCalls };
  }

  private async generate(
    baseUrl: string,
    model: string,
    numCtx: number,
    system: string,
    prompt: string,
  ): Promise<OpsReport> {
    const call = async (format: unknown): Promise<string> => {
      const res = await fetch(`${baseUrl}/api/generate`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          model,
          system,
          prompt,
          stream: false,
          format,
          options: { num_ctx: numCtx, temperature: 0.2 },
        }),
      });
      if (!res.ok) throw new Error(`ollama HTTP ${res.status}`);
      const json = (await res.json()) as { response?: string };
      return (json.response ?? '').trim();
    };

    // 1) schema-constrained output (Ollama 0.5+). 2) fallback to plain json.
    let lastErr: unknown;
    for (const format of [opsReportJsonSchema, 'json'] as const) {
      try {
        const parsed = opsReportSchema.safeParse(extractJson(await call(format)));
        if (parsed.success) return parsed.data;
      } catch (e) {
        lastErr = e; // 연결 실패(맥미니 다운 등) 등 — 다음 전략 시도
      }
    }
    throw new Error(
      `ollama 리포트 생성 실패: ${lastErr instanceof Error ? lastErr.message : '스키마 불일치'}`,
    );
  }
}

/**
 * 두뇌 폴백 체인. 앞 두뇌가 실패(예: 맥미니 Ollama 다운)하면 다음 두뇌로 넘어간다.
 * VM 에이전트가 맥미니에 의존하므로, ollama→anthropic→mock 순으로 두면
 * 맥미니가 꺼져도 리포트는 계속 나온다(품질은 떨어질 수 있음).
 */
export class FallbackBrain implements Brain {
  readonly name: string;

  constructor(private readonly brains: Brain[]) {
    this.name = brains.map((b) => b.name).join('→');
  }

  async run(input: BrainInput): Promise<BrainResult> {
    let lastErr: unknown;
    for (const brain of this.brains) {
      try {
        const result = await brain.run(input);
        return { ...result, usedBrain: brain.name };
      } catch (e) {
        lastErr = e;
        console.warn(
          `[ops-agent] 두뇌 '${brain.name}' 실패 → 폴백:`,
          e instanceof Error ? e.message : String(e),
        );
      }
    }
    throw new Error(
      `모든 두뇌 실패: ${lastErr instanceof Error ? lastErr.message : String(lastErr)}`,
    );
  }
}

function summarize(value: unknown): string {
  const s = JSON.stringify(value);
  return s.length > 240 ? `${s.slice(0, 240)}…` : s;
}

/** Parse JSON, tolerating leading/trailing prose around the object. */
function extractJson(raw: string): unknown {
  try {
    return JSON.parse(raw);
  } catch {
    const start = raw.indexOf('{');
    const end = raw.lastIndexOf('}');
    if (start >= 0 && end > start) {
      try {
        return JSON.parse(raw.slice(start, end + 1));
      } catch {
        /* fall through */
      }
    }
    return null;
  }
}
