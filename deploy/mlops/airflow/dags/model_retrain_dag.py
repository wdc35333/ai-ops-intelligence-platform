"""Weekly model-retrain DAG.

Retrains the anomaly model on a longer (30-day) window and re-registers a new
version in the MLflow Model Registry. Separating retrain cadence from daily
scoring is the standard MLOps split: score often, retrain rarely.
"""

from __future__ import annotations

import os
import urllib.request
from datetime import datetime, timedelta

from airflow.decorators import dag, task

ML_URL = os.environ.get("ML_ANOMALY_URL", "http://ml-anomaly:8200")


@dag(
    dag_id="model_retrain",
    description="Weekly: retrain on a 30-day window, register a new model version",
    schedule="0 0 * * 1",  # Mondays 00:00 UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["kiosk", "mlops", "retrain"],
)
def model_retrain():
    @task
    def retrain() -> int:
        req = urllib.request.Request(
            f"{ML_URL}/pipeline/run?window_hours=720", method="POST"
        )
        with urllib.request.urlopen(req, timeout=900) as resp:
            return resp.status

    retrain()


model_retrain()
