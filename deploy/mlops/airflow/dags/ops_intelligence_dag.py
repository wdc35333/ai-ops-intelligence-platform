"""Daily ops-intelligence DAG.

Orchestrates the ML anomaly pipeline on a schedule, with retries and an explicit
dependency edge (scoring must finish before the downstream task reads the feed).
This is the part a bare cron cannot express: ordering, retries, run history, and
backfill.

Airflow only *orchestrates* — it calls the read-only ml-anomaly service over HTTP,
and the service does the work and logs to MLflow. Uses stdlib urllib so it needs
no extra image packages (no curl/provider deps).
"""

from __future__ import annotations

import os
import urllib.request
from datetime import datetime, timedelta

from airflow.decorators import dag, task

ML_URL = os.environ.get("ML_ANOMALY_URL", "http://ml-anomaly:8200")


def _http(path: str, method: str = "GET", timeout: int = 600) -> int:
    req = urllib.request.Request(f"{ML_URL}{path}", method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status


@dag(
    dag_id="ops_intelligence",
    description="Daily: train/score anomaly models (→ MLflow), then verify the feed",
    schedule="30 23 * * *",  # 08:30 KST (UTC+9) == 23:30 UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["kiosk", "mlops"],
)
def ops_intelligence():
    @task
    def ml_pipeline_run() -> int:
        # train/score + log params, metrics (recall, drift PSI), model → MLflow
        return _http("/pipeline/run?window_hours=168", method="POST")

    @task
    def verify_anomaly_feed(_prev: int) -> int:
        # confirm the downstream feed the LLM agent consumes is reachable
        return _http("/anomalies?window_hours=72")

    verify_anomaly_feed(ml_pipeline_run())

    # Production note: append the LLM ops-agent as a third task. It needs the
    # Next.js manager runtime (node), so run it from the manager container, e.g.
    # BashOperator(task_id="ops_agent_report",
    #     bash_command="docker exec kiosk-manager npm run ops-agent:once")


ops_intelligence()
