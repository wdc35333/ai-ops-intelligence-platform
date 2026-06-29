# ml-anomaly — kiosk operations anomaly detection

A small, read-only **Python / FastAPI** microservice that detects anomalies in the
operational time-series of an unmanned IoT locker-kiosk fleet (host metrics and
booking revenue) and exposes them as a ranked feed. It is the ML layer of a larger
**AI operations-intelligence platform**: an LLM ops agent calls this service's
`/anomalies` endpoint as a tool, interprets the result, and reports/alerts.

> Part of the kiosk platform. This service is **read-only** — it never writes
> to the production database.

## Why it exists / what it demonstrates

| Capability | Where in this service |
|---|---|
| Python backend (FastAPI) | `app/main.py` |
| Time-series **anomaly detection** (unsupervised ML) | `app/detectors.py` |
| Two techniques, fit for the signal | IsolationForest (multivariate host metrics) + robust seasonal z-score (univariate revenue) |
| Testable, infra-free design | `app/sources.py` data-source abstraction (`synthetic` ↔ `db`) |
| Recall measured against ground truth | `tests/` inject anomalies, assert detection |
| Production safety | read-only SQL, separate service, feature-flagged data source |

This maps directly onto a DT / Data-AI job description: *"분석하여 … 이상 탐지 … 운영 가능한
AI/ML 모델을 검증/개발/운영"*. The drift-monitoring and MLflow/Airflow layers build on top
of this service (see the platform roadmap).

## Architecture

```
            ┌──────────────────────── ml-anomaly (this service) ────────────────────────┐
 data ──▶   │  sources.py            features.py            detectors.py        main.py   │ ──▶ JSON
(MariaDB    │  DataSource            engineered             IsolationForest /    FastAPI   │   /anomalies
 or         │  ├─ DbSource (RO SQL)  features (disk%,       seasonal z-score     endpoints │   /detect/*
 synthetic) │  └─ SyntheticSource    mem%, load; hourly                                    │
            └────────────────────────────────────────────────────────────────────────────┘
                                                                         ▲
                                              LLM ops agent  ── get_ml_anomalies tool ──┘
```

- **`system_metrics`** (5-min host samples) → derive `disk_used_pct`, `mem_used_pct`,
  `load1` → **IsolationForest** flags joint multivariate outliers.
- **`online_booking`** → aggregate hourly `revenue` → **seasonal robust z-score** flags
  hours that deviate from the same hour-of-day's normal (median/MAD), so it is not fooled
  by the daily cycle.
- `/anomalies` merges both into a single **severity-ranked** feed (`critical`/`warning`)
  with short Korean explanations the agent can relay.

## Run it

No database required — defaults to the synthetic source.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                                   # 6 passed
uvicorn app.main:app --port 8200            # then open http://127.0.0.1:8200/docs
```

Against the live DB (read-only user):

```bash
export MLA_DATA_SOURCE=db
export MLA_DATABASE_URL='mysql+pymysql://ro_user:password@host:3306/kioskdb'
uvicorn app.main:app --port 8200
```

Docker: `docker build -t ml-anomaly . && docker run -p 8200:8200 ml-anomaly`

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + active data source |
| GET | `/detect/system-metrics?window_hours=` | detailed host-metric anomalies |
| GET | `/detect/bookings?window_hours=` | detailed revenue anomalies |
| GET | `/anomalies?window_hours=` | compact, ranked feed (LLM-agent contract) |
| GET | `/drift?window_hours=` | per-feature distribution drift (PSI) + severity (L5) |
| POST | `/pipeline/run?window_hours=` | train/score → log to MLflow (L3) |

### Sample `/anomalies` output (synthetic)

```
counts: {'total': 22, 'critical': 6, 'warning': 16}
  [critical] [예약/매출] revenue 급증 (z=80.6, 기대≈4000)
  [critical] [시스템] disk_used_pct 이상 (robust-z=8.4)
  [critical] [시스템] mem_used_pct 이상 (robust-z=8.6)
```

## Configuration (env, prefix `MLA_`)

| Var | Default | Meaning |
|---|---|---|
| `MLA_DATA_SOURCE` | `synthetic` | `synthetic` or `db` |
| `MLA_DATABASE_URL` | — | read-only SQLAlchemy URL (required for `db`) |
| `MLA_DEFAULT_WINDOW_HOURS` | `168` | look-back window |
| `MLA_IF_CONTAMINATION` | `0.02` | IsolationForest expected outlier fraction |
| `MLA_BOOKING_Z_THRESH` | `3.5` | seasonal z-score threshold |

## Design notes

- **Two detectors on purpose.** Host metrics move together (a disk-full event coincides
  with backup load), so a *joint* model beats per-metric thresholds. Revenue is strongly
  seasonal, so a *season-aware* baseline avoids flagging every quiet night.
- **Data-source abstraction** keeps the service runnable with zero infrastructure (CI,
  demos, reviewers) and gives tests deterministic ground truth.
- **Explanations, not just scores.** Every anomaly carries a human-readable `reason`, so
  the downstream LLM agent can produce a report without re-deriving why.

## MLOps (L3) — `POST /pipeline/run`

`POST /pipeline/run` trains/scores and logs to **MLflow**: params, metrics
(`recall_synthetic`, `anomaly_rate`), a **PSI drift** metric per feature, and the fitted
model (registered as `kiosk-system-anomaly`). **Airflow** schedules it (daily score, weekly
retrain). The full stack — mlflow server + airflow + this service — is in
[`deploy/mlops/`](../../deploy/mlops). Locally it logs to a file store; the compose stack
uses the mlflow server for the registry-backed UI.

## Roadmap (platform)

Drift detection (L5) is live: `/drift` (PSI, reference vs current, severity-classified) +
the agent's `get_model_drift` tool, so drift is surfaced and alertable. Still ahead: a
model-monitoring dashboard page (the MLflow UI already charts the drift metrics over time).
