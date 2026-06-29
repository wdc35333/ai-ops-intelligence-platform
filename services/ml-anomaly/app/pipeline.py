"""MLflow training/scoring pipeline — the MLOps (L3) layer.

Run by Airflow (via ``POST /pipeline/run``) on a schedule. For each detector it:
  * fits / scores on a window,
  * evaluates recall against synthetic ground truth (when source=synthetic),
  * computes a PSI (Population Stability Index) drift indicator per feature,
  * logs params + metrics + the fitted model to MLflow, and registers the
    system-anomaly model (when the tracking store supports a registry).

Tracking URI comes from ``MLFLOW_TRACKING_URI`` (default local sqlite, which is
registry-capable). The rest of the service does not import mlflow — it is pulled
in here, lazily, so L2 stays lightweight.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from .config import Settings, get_settings
from .detectors import detect_isolation_forest, detect_seasonal_zscore
from .features import SYSTEM_FEATURES, add_system_metric_features, aggregate_bookings_hourly
from .sources import get_source, make_synthetic_system_metrics

# Local default is a file store (no SQL deps). The compose stack overrides this
# with the mlflow server (http://mlflow:5000), which is registry-capable.
DEFAULT_TRACKING_URI = "file:./mlruns"
EXPERIMENT = os.environ.get("MLA_MLFLOW_EXPERIMENT", "kiosk-anomaly")


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two samples (a drift indicator).

    ~0 = no shift, >0.1 = moderate, >0.25 = significant distribution drift.
    """
    qs = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, qs))
    if len(cuts) < 3:
        return 0.0
    e = np.histogram(expected, bins=cuts)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins=cuts)[0] / max(len(actual), 1)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def run_pipeline(window_hours: Optional[int] = None, settings: Optional[Settings] = None) -> dict:
    import mlflow
    import mlflow.sklearn

    settings = settings or get_settings()
    window = window_hours or settings.default_window_hours
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(EXPERIMENT)
    source = get_source(settings)

    summary: dict = {
        "experiment": EXPERIMENT,
        "window_hours": window,
        "source": source.name,
        "runs": {},
    }

    # ── system metrics: IsolationForest (registered model) ──
    with mlflow.start_run(run_name="system-anomaly") as run:
        feats = add_system_metric_features(source.system_metrics(window))
        scored = detect_isolation_forest(
            feats, SYSTEM_FEATURES, contamination=settings.if_contamination
        )
        n_anom = int(scored["is_anomaly"].sum())
        mlflow.log_params(
            {
                "detector": "isolation_forest",
                "window_hours": window,
                "contamination": settings.if_contamination,
                "features": ",".join(SYSTEM_FEATURES),
                "source": source.name,
            }
        )
        mlflow.log_metrics(
            {
                "n_points": len(scored),
                "n_anomalies": n_anom,
                "anomaly_rate": n_anom / max(len(scored), 1),
            }
        )
        if source.name == "synthetic":
            _, injected = make_synthetic_system_metrics(window, settings.synthetic_seed)
            flagged = set(scored.index[scored["is_anomaly"]].tolist())
            recall = len(set(injected) & flagged) / len(injected) if injected else 0.0
            mlflow.log_metric("recall_synthetic", recall)
        # drift: PSI of first half vs second half of the window, per feature
        half = max(len(feats) // 2, 1)
        for col in SYSTEM_FEATURES:
            vals = feats[col].to_numpy()
            mlflow.log_metric(f"psi_{col}", _psi(vals[:half], vals[half:]))

        model = IsolationForest(
            n_estimators=200, contamination=settings.if_contamination, random_state=42
        ).fit(feats[SYSTEM_FEATURES].to_numpy())
        info = mlflow.sklearn.log_model(model, artifact_path="model")
        registered = False
        try:
            mlflow.register_model(info.model_uri, "kiosk-system-anomaly")
            registered = True
        except Exception:
            # Model Registry needs a DB-backed store; the artifact is logged regardless.
            pass
        summary["runs"]["system"] = {
            "run_id": run.info.run_id,
            "n_anomalies": n_anom,
            "registered": registered,
        }

    # ── bookings: seasonal z-score (stateless; metrics only) ──
    with mlflow.start_run(run_name="bookings-anomaly") as run:
        hourly = aggregate_bookings_hourly(source.bookings(window))
        if hourly.empty:
            mlflow.log_metric("n_points", 0)
            summary["runs"]["bookings"] = {"run_id": run.info.run_id, "n_anomalies": 0}
        else:
            scored = detect_seasonal_zscore(
                hourly, "revenue", "bucket", z_thresh=settings.booking_z_thresh
            )
            n_anom = int(scored["is_anomaly"].sum())
            mlflow.log_params(
                {
                    "detector": "seasonal_zscore",
                    "z_thresh": settings.booking_z_thresh,
                    "source": source.name,
                }
            )
            mlflow.log_metrics({"n_points": len(scored), "n_anomalies": n_anom})
            rev = scored["revenue"].to_numpy()
            half = max(len(rev) // 2, 1)
            mlflow.log_metric("psi_revenue", _psi(rev[:half], rev[half:]))
            summary["runs"]["bookings"] = {"run_id": run.info.run_id, "n_anomalies": n_anom}

    return summary
