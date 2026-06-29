"""Distribution-drift detection (PSI) — the L5 monitoring layer.

Compares a reference window (older half) against the current window (recent half)
per feature and flags when the distribution has shifted. The pipeline (L3) already
logs these PSI values to MLflow over time, so the MLflow UI doubles as the drift
dashboard; this module turns drift into a first-class, severity-classified signal
the ops agent can surface and alert on.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import Settings
from .features import SYSTEM_FEATURES, add_system_metric_features, aggregate_bookings_hourly
from .sources import DataSource

# PSI thresholds (industry-standard): <0.1 stable, 0.1–0.25 moderate, >0.25 significant.
WARN_PSI, CRIT_PSI = 0.1, 0.25
_ORDER = {"ok": 0, "warning": 1, "critical": 2}


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two samples (drift indicator)."""
    qs = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, qs))
    if len(cuts) < 3:
        return 0.0
    e = np.histogram(expected, bins=cuts)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins=cuts)[0] / max(len(actual), 1)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def classify(value: float) -> str:
    if value >= CRIT_PSI:
        return "critical"
    if value >= WARN_PSI:
        return "warning"
    return "ok"


def _drift_of(series: np.ndarray) -> dict:
    half = max(len(series) // 2, 1)
    value = psi(series[:half], series[half:])
    return {"psi": round(value, 4), "severity": classify(value)}


def compute_drift(
    source: DataSource, settings: Settings, window_hours: Optional[int] = None
) -> dict:
    window = window_hours or settings.default_window_hours
    features: dict[str, dict] = {}

    sys_df = add_system_metric_features(source.system_metrics(window))
    for col in SYSTEM_FEATURES:
        features[col] = _drift_of(sys_df[col].to_numpy())

    hourly = aggregate_bookings_hourly(source.bookings(window))
    if not hourly.empty:
        features["revenue"] = _drift_of(hourly["revenue"].to_numpy())

    overall = "ok"
    for f in features.values():
        if _ORDER[f["severity"]] > _ORDER[overall]:
            overall = f["severity"]

    return {
        "source": source.name,
        "window_hours": window,
        "features": features,
        "overall": overall,
        "drifted": overall != "ok",
    }
