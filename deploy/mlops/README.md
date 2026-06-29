# MLOps stack (L3) — MLflow + Airflow

Turns the [ml-anomaly](../../services/ml-anomaly) detector from a "run once" model
into an **operated** ML system: experiments and model versions are tracked, drift is
measured, and scoring/retraining run on a schedule with retries and history.

```
Airflow (schedule · order · retries) ──HTTP──▶ ml-anomaly /pipeline/run ──log──▶ MLflow (track · registry)
   ops_intelligence  (daily)                       fit · score · drift                experiments + model versions
   model_retrain     (weekly)
```

## What each piece adds

| Tool | Role | Concretely for the kiosk fleet |
|---|---|---|
| **MLflow** | memory: experiments, metrics, model registry | compare `contamination`/`z_thresh` by **measured** recall; version + roll back the detector; store **drift (PSI)** history (foundation for the L5 drift dashboard) |
| **Airflow** | hands: schedule, ordering, retries, run history | replaces fragile cron — a failed scoring run **retries and is visible**, scoring runs **before** the agent reads it, past days can be **backfilled** |

## Run

```bash
docker compose -f deploy/mlops/docker-compose.yml up
# MLflow UI   → http://localhost:5000   (runs, metrics, registered models)
# Airflow UI  → http://localhost:8080   (DAGs; standalone prints the admin password)
# ml-anomaly  → http://localhost:8200/docs
```

Trigger once without waiting for the schedule:

```bash
curl -X POST "http://localhost:8200/pipeline/run?window_hours=168"   # logs a run to MLflow
# …then open the MLflow UI to see params, metrics (recall_synthetic, psi_*), and the model
```

Point at real data (read-only): set `MLA_DATA_SOURCE=db` and `MLA_DATABASE_URL` in the
environment before `up`.

## DAGs (`airflow/dags/`)

| DAG | Schedule | Does |
|---|---|---|
| `ops_intelligence` | daily 08:30 KST | `ml_pipeline_run` (train/score → MLflow) → `verify_anomaly_feed` |
| `model_retrain` | weekly (Mon) | retrain on a 30-day window, register a new model version |

DAGs use stdlib `urllib` (no curl/provider deps) and orchestrate via HTTP — Airflow
schedules, the service does the work.

## Verified locally

- `POST /pipeline/run` logs two MLflow runs with params, metrics
  (`recall_synthetic=1.0`, `psi_disk_used_pct`, `anomaly_rate`, …) and registers
  `kiosk-system-anomaly` in the Model Registry (`pytest tests/test_pipeline.py`).
- DAG files are syntax-checked; full execution runs in the compose stack above.

## JD mapping

*"MLflow · Kubeflow · Airflow 등 MLOps 플랫폼을 이용한 AI/ML 분석 모델링"* and *"AI 모델의
성능(Accuracy, Drift)에 대한 실시간 모니터링"* — the explicitly-named requirement, demonstrated
end to end (track → schedule → retrain → drift).
