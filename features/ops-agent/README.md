# ops-agent — LLM operations agent

An autonomous **operations-monitoring agent** for the kiosk platform. It uses
**Claude Opus 4.8** in a tool-use loop to inspect the fleet's operational state through
**read-only** tools, calls the [ML anomaly service](../../services/ml-anomaly) as one of
those tools, reasons about what's wrong, and emits a structured Korean report (alerting
when severity warrants). It is the L1 layer of the AI operations-intelligence platform.

> **Read-and-report only.** There are no write tools — the "agent can't change anything"
> guarantee is enforced at the harness level, not by prompting.

## What it demonstrates

| Capability | Where |
|---|---|
| LLM **agent** (observe → reason → call tool → repeat) | `brains.ts` `AnthropicBrain` |
| Custom **tool surface** (typed, gated, read-only) | `tools.ts` |
| **Structured output** via a terminal `submit_report` tool (Zod-validated) | `schema.ts`, `brains.ts` |
| **Model is a swappable part** (`AnthropicBrain` ↔ `OllamaBrain` ↔ `MockBrain`) | `brains.ts` |
| LLM + ML composition (agent calls the ML service) | `get_ml_anomalies` tool |
| Testability without an API key | `MockBrain` exercises the full tool path deterministically |
| Cost/safety guardrails | step cap, dry-run default, graceful DB degradation |

Maps onto a DT / Data-AI JD: *"이상 탐지 … 자동 판단 … 운영 가능한 모델을 운영"* and
*"실시간 모니터링 및 알림 시스템 구축"*.

## How it works

```
scripts/ops-agent.ts  ──▶  Brain.run({ system, dataTools, execute })
                              │  loop (≤ OPS_AGENT_MAX_STEPS):
                              │   model picks tools → execute() → results back to model
                              │   ┌─ get_ml_anomalies ─▶ services/ml-anomaly  /anomalies
                              │   ├─ get_backup_status / get_system_metrics / get_db_health
                              │   └─ get_sales_summary           (prisma, read-only, RO-degrade)
                              ▼
                       submit_report(structured)  ──▶  print + persist(guarded) + alert(dry-run)
```

The **brain** is an interface, so the model is a swappable part:
- `AnthropicBrain` — real Claude Opus 4.8 tool-use loop (the model orchestrates the tools).
- `OllamaBrain` — a local model (e.g. Mac-mini `gemma4:e4b`), **free + private**. A small
  local model isn't reliable at multi-step tool orchestration, so the harness gathers all
  read-only data first ("tools-on-rails") and the model does the reasoning/synthesis with
  schema-constrained JSON output. (Same Ollama pattern as the meeting-AI worker.)
- `MockBrain` — deterministic; verifies the whole tool path with **no API key, no cost**.

The default run wraps these in a **`FallbackBrain`** chain (e.g. `ollama→anthropic→mock`): if
the primary brain fails — say the Mac-mini Ollama is unreachable — it logs the failure and falls
back to the next, so a report is still produced. This matters in production where the agent runs
on the KT-Cloud VM and borrows the Mac mini's GPU over a tunnel.

## Run

```bash
# 1) start the ML anomaly service (provides get_ml_anomalies)
( cd services/ml-anomaly && .venv/bin/uvicorn app.main:app --port 8200 )

# 2) run the agent once
#    no ANTHROPIC_API_KEY → MockBrain (free, deterministic)
OPS_AGENT_BRAIN=mock ML_ANOMALY_URL=http://127.0.0.1:8200 npx tsx scripts/ops-agent.ts
#    with a key → real Claude Opus 4.8 tool-use loop
ANTHROPIC_API_KEY=sk-... npm run ops-agent:once
#    local Gemma (free, private) — point at the Mac mini or a local Ollama
OPS_AGENT_BRAIN=ollama OLLAMA_BASE_URL=http://127.0.0.1:11434 OLLAMA_MODEL=gemma4:e4b \
  ML_ANOMALY_URL=http://127.0.0.1:8200 npx tsx scripts/ops-agent.ts
```

### Sample output (MockBrain, synthetic ML data)

```
운영 점검 리포트  (brain=mock)
도구 호출: get_ml_anomalies, get_backup_status, get_system_metrics, get_db_health, get_sales_summary
심각도 : critical
헤드라인: 심각 이상 6건 감지(ML)
  · (critical) [예약/매출] revenue 급증 (z=80.6, 기대≈4000)
  · (critical) [시스템] disk_used_pct 이상 (robust-z=8.4)
[ops-agent] (dry-run) 알림 보류: critical — 심각 이상 6건 감지(ML)
```

## Configuration (env)

| Var | Default | Meaning |
|---|---|---|
| `OPS_AGENT_BRAIN` | auto | `anthropic` \| `ollama` \| `mock` (auto = anthropic if `ANTHROPIC_API_KEY` set, else mock) |
| `OPS_AGENT_MODEL` | `claude-opus-4-8` | model id for `AnthropicBrain` |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama host for `OllamaBrain` (e.g. the Mac mini) |
| `OLLAMA_MODEL` | `gemma4:e4b` | local model id for `OllamaBrain` |
| `OPS_AGENT_DRY_RUN` | `1` | `0` to actually dispatch alerts |
| `OPS_AGENT_MAX_STEPS` | `12` | tool-loop cap (bounds cost) |
| `ML_ANOMALY_URL` | `http://127.0.0.1:8200` | ML anomaly service base URL |

## Safety

- **No write tools** → cannot mutate the system, by construction.
- DB tools **graceful-degrade** (`{available:false}`) when `DATABASE_URL` is unset/unreachable.
- Alert dispatch is **dry-run by default**; report persistence is **guarded** (skips silently
  if the `ops_agent_report` table — applied via `db/*.sql` — isn't present yet).

## Next

Wire `ops_agent_alert` into `lib/push-events.ts` for live alerting; add the daily
exec-briefing prompt variant; have MLflow/Airflow (platform L3) schedule and track this run.
