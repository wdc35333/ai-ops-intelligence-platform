> **Portfolio extract.** The AI-ops layers of a production unmanned-kiosk platform, pulled into
> a standalone repo. The **ML service** (`services/ml-anomaly`) and **MLOps stack** (`deploy/mlops`)
> run standalone (synthetic data, `pytest`); the TypeScript **ops-agent** (`features/ops-agent`) is
> shown as it integrates with a Next.js manager app — its `@/lib/*` imports refer to that parent app.

# AI Operations-Intelligence Platform

An end-to-end AI system that watches a fleet of **unmanned IoT kiosks** (locker rental)
and reports what's wrong — in Korean, on a schedule, for free. It pairs an **LLM agent**
(observe → reason → act) with **ML anomaly/drift detection** and a real **MLOps** loop,
and it runs its "brain" on a Mac mini's GPU over a tunnel so production pays no token cost.

Built as a portfolio for **AI-systems / Data-AI roles** (e.g. SK hynix DT · Data 분석·개발).
Everything here is verified locally; nothing requires the production database or an API key
to run (a synthetic data source and a mock brain make the whole thing demoable offline).

> **▶ Read [`docs/CASE_STUDY.md`](docs/CASE_STUDY.md) first.** Four real (sanitized) stories of the
> platform earning its keep: an agent signal that unwound a **two-root-cause, cross-repo payment
> bug**; making drift detection **stop crying wolf** on real data; the discipline of **not shipping
> tools the data can't support**; and the operational seams of a real MLOps deployment. That's the
> part that's hard to fake.

---

## The system at a glance

```
┌──────────────────────────── AI Operations-Intelligence Platform ─────────────────────────────┐
│                                                                                               │
│  L1  LLM ops agent   features/ops-agent · scripts/ops-agent.ts                                │
│      observe → reason → call tool → repeat → submit_report  (structured, READ-AND-REPORT only)│
│      brains:  AnthropicBrain  ·  OllamaBrain (local, free)  ·  MockBrain   ──FallbackBrain──▶  │
│         │ 8 read-only tools                                                                    │
│         ├─ get_ml_anomalies · get_model_drift  ──HTTP──▶  L2/L5  services/ml-anomaly (Python)  │
│         ├─ get_backup_status · get_system_metrics · get_db_health · get_sales_summary (Prisma) │
│         └─ get_payment_consistency · get_double_charge_signals  (payment integrity)            │
│                                    │                                                           │
│  L2 anomaly detection              │   L3 MLOps                        L5 drift                │
│  IsolationForest (host metrics)    └─▶ MLflow  (experiments·registry)  /drift  (PSI per        │
│  + seasonal robust z-score (rev)       Airflow (schedule·retrain)       feature, severity)     │
│                                                                                               │
│  L0 (read-only): system_metrics · online_booking · db_backup_log · use_log_payment · event_log │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

brain over the network (production, no token cost):
   Mac mini  ──autossh -R──▶  KT Cloud VM            VM agent: OLLAMA_BASE_URL=http://…:11434
   (Tailscale · M4 GPU · gemma4:e4b)   (off Tailscale)
```

The agent **autonomously** decides which tools to call and how to judge severity; the harness
only controls the side effects (persist + alert), so "read-and-report-only" is enforced in code
(there are no write tools), not by prompting.

---

## Layers → competencies

| Layer | What it is | Proves (JD) |
|---|---|---|
| **L1** agent | Claude/Ollama tool-use loop, 8 read-only tools, structured `submit_report` | 최신 AI · **에이전트** · 이상 탐지·**자동 판단** · Vibe coding |
| **L1** integrity | `get_payment_consistency` (6 composite checks) + `get_double_charge_signals` over `event_log` | **데이터 정합성·이상 신호 SQL** · 운영 데이터 분석 ([CASE_STUDY §1](docs/CASE_STUDY.md)) |
| **L2** anomaly | Python/FastAPI; IsolationForest (multivariate) + seasonal z-score (univariate) | **Python 백엔드** · **시계열 이상탐지 ML 모델 개발·운영** |
| **L3** MLOps | **MLflow** (experiments, metrics, model registry) + **Airflow** (schedule, retrain) | **MLflow·Airflow MLOps 플랫폼** (JD 필수항목) |
| **L5** drift | `/drift` PSI reference-vs-current per feature, severity-classified, agent-surfaced | **성능(Accuracy, Drift) 실시간 모니터링·알림** |
| brains | Anthropic ↔ Ollama ↔ Mock, `FallbackBrain` auto-failover | 모델 비종속 · Foundation model 활용 · 운영 신뢰성 |
| infra | KT Cloud Docker, multi-repo distributed system, Tailscale/SSH tunneling | Linux·클라우드 · **분산 시스템·트러블슈팅** |

**One-line résumé framing:** *"Built an AI ops platform over an unmanned-IoT kiosk fleet:
time-series & drift ML models, an MLflow/Airflow MLOps pipeline, and an LLM agent that calls
them as tools to auto-judge and report anomalies — with a swappable, self-failing-over brain
served free from on-prem GPU."*

---

## The three brains (a model is a swappable part)

| Brain | Where | Cost | Use |
|---|---|---|---|
| `AnthropicBrain` | Claude Opus 4.8, cloud | $ per token | strongest reasoning; full tool-use loop |
| `OllamaBrain` | local `gemma4:e4b` (Mac mini GPU) | free / private | tools-on-rails (gather → synthesize) for a small model |
| `MockBrain` | deterministic | none | verifies the whole tool path with no key, no cost |

`FallbackBrain` chains them (`ollama→anthropic→mock`): if the Mac mini is unreachable, the
agent logs it and falls back, so a report is still produced. Selected via `OPS_AGENT_BRAIN`.

---

## Where the brain runs (networking)

Production runs the agent on the **KT Cloud VM** but borrows the Mac mini's GPU:

```
Mac mini (on Tailscale, gemma4:e4b)  ──autossh -R 11434:127.0.0.1:11434──▶  KT VM (off Tailscale)
                                                                            agent → 127.0.0.1:11434
```

The Mac mini **dials out** to the VM (a reverse tunnel), so the VM never joins Tailscale.
For dev, a forward tunnel (`-L 11435→11434`) lets a laptop use the same brain. (See
[`features/ops-agent/README.md`](features/ops-agent/README.md) and `the parent app env template`.)

---

## Quick start (no DB, no API key)

```bash
# 1) ML service (synthetic data source by default)
cd services/ml-anomaly && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pytest -q            # 17 passed
uvicorn app.main:app --port 8200                            # /docs, /anomalies, /drift, /pipeline/run

# 2) the agent (MockBrain — free, deterministic)
OPS_AGENT_BRAIN=mock ML_ANOMALY_URL=http://127.0.0.1:8200 npx tsx scripts/ops-agent.ts

# 3) MLOps stack (MLflow UI :5000, Airflow :8080)
docker compose -f deploy/mlops/docker-compose.yml up
```

Use the real Mac mini brain with `OPS_AGENT_BRAIN=ollama OLLAMA_BASE_URL=http://127.0.0.1:11435`,
or Claude with `ANTHROPIC_API_KEY=…`.

---

## What's verified

- `services/ml-anomaly`: **pytest 17/17** — detector recall vs injected anomalies, API contract,
  MLflow pipeline (recall + drift logged, model registered), and drift classification incl. the
  **robust-scale PSI, detrend, and baseline-sufficiency guard** (see `tests/test_drift.py`).
- `scripts/ops-agent.ts` + `features/ops-agent`: **`tsc --noEmit` clean**; end-to-end runs with
  Mock, local `gemma4:e4b`, the Mac mini's `gemma4:e4b` over the tunnel, and the fallback chain.

## Production vs demo (honest status)

| | status |
|---|---|
| ML service, agent, MLOps, drift, brains/fallback, networking | ✅ built + verified locally |
| Running on the **real kiosk DB** (read-only account) | ✅ live — `MLA_DATA_SOURCE=db`, per-site real revenue, payment-integrity tools |
| Deployed to the production VM | ✅ live — `ml-anomaly` container + daily cron `/pipeline/run`, agent on schedule |
| **MLOps loop** (MLflow tracking + registry) | ✅ live — daily scoring, UI behind reverse-proxy Basic Auth (no tunnel) |
| Live push alerts | ✅ live — `ops_agent_alert` wired through `lib/push-events.ts`, throttled by severity |
| Airflow scheduler | ⏳ DAGs in repo; runs on **host cron** instead (Airflow standalone exceeds the VM's RAM budget) |

See [`docs/CASE_STUDY.md`](docs/CASE_STUDY.md) §2 (drift) and §4 (deployment seams) for how each of these
went from "works locally" to "runs unattended in production."

## Roadmap

Per-kiosk fault diagnosis (deferred — central data can't yet separate a real offline fault from
benign states; see CASE_STUDY §3), an exec-briefing prompt variant, and turning the Airflow DAGs on
when the VM gets more headroom. (Multimodal vision was descoped — no in-locker cameras.)

## Map

- [`docs/CASE_STUDY.md`](docs/CASE_STUDY.md) — **four real (sanitized) stories** — start here
- [`services/ml-anomaly`](services/ml-anomaly/README.md) — L2 anomaly + L3 MLflow pipeline + L5 drift
- [`deploy/mlops`](deploy/mlops/README.md) — MLflow + Airflow stack
- [`features/ops-agent`](features/ops-agent/README.md) — L1 agent + brains + fallback
