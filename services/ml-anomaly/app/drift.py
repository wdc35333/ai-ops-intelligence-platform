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


def _robust_scale(x: np.ndarray) -> float:
    """MAD-based σ estimate (≈ std for normal data), never zero."""
    mad = float(np.median(np.abs(x - np.median(x))))
    if mad > 0:
        return mad * 1.4826
    s = float(np.std(x))
    return s if s > 0 else 1.0


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two samples — made scale-robust.

    Both samples are standardized by the reference's robust scale (median / MAD)
    and binned on a FIXED ±4σ grid with open (±inf) tails, so PSI measures the
    shift in units of natural variability rather than absolute value.

    Why not the textbook reference-quantile bins: a low-variance feature (disk at
    27.x% with σ≈0.006) has microscopic quantile bins, so a trivial 1%p wobble
    crosses many of them and PSI explodes — and any out-of-range point was being
    dropped by ``np.histogram``, collapsing the actual histogram to the smoothing
    floor and saturating PSI at ~11.5 (= -ln(1e-5)). Standardized fixed bins +
    open tails + Laplace smoothing keep PSI finite, bounded, and meaningful.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 2 or len(actual) < 2:
        return 0.0

    center = float(np.median(expected))
    scale = _robust_scale(expected)
    e_z = (expected - center) / scale
    a_z = (actual - center) / scale

    edges = np.linspace(-4.0, 4.0, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf  # open tails: never drop a point

    e_cnt = np.histogram(e_z, bins=edges)[0].astype(float)
    a_cnt = np.histogram(a_z, bins=edges)[0].astype(float)
    e = (e_cnt + 0.5) / (e_cnt.sum() + 0.5 * len(e_cnt))
    a = (a_cnt + 0.5) / (a_cnt.sum() + 0.5 * len(a_cnt))
    return float(np.sum((a - e) * np.log(a / e)))


def classify(value: float) -> str:
    if value >= CRIT_PSI:
        return "critical"
    if value >= WARN_PSI:
        return "warning"
    return "ok"


def _detrend(series: np.ndarray) -> np.ndarray:
    """Strip a linear trend before comparing halves.

    Normal monotonic growth — a slowly filling disk, creeping memory — is steady
    capacity drift, not a model-relevant distribution shift, yet half-vs-half PSI
    flags it every single time (the recent half sits above the older one). We fit
    and remove a least-squares line so that steady growth reads as ``ok`` while
    genuine regime changes (level steps, variance blow-ups, broken seasonality)
    survive in the residuals and still fire.
    """
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 4:
        return series
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, series, 1)
    return series - (slope * x + intercept)


def _drift_of(series: np.ndarray) -> dict:
    resid = _detrend(series)
    half = max(len(resid) // 2, 1)
    value = psi(resid[:half], resid[half:])
    return {"psi": round(value, 4), "severity": classify(value)}


def _span_hours(sys_df) -> float:
    """Hours of history actually present in the metrics frame."""
    if sys_df.empty or "collected_at" not in sys_df or len(sys_df) < 2:
        return 0.0
    ts = sys_df["collected_at"]
    return float((ts.max() - ts.min()).total_seconds()) / 3600.0


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

    # Baseline-history guard. Half-vs-half PSI is only trustworthy once enough
    # *stable* history has accrued — a reference window spanning several daily
    # cycles (default ≥ 1 week). With less, the two adjacent time-blocks differ
    # purely by time-of-day phase and PSI reports false drift. Until then we keep
    # the raw psi values for transparency but report 'building baseline' (ok)
    # instead of crying wolf.
    baseline_hours = _span_hours(sys_df)
    sufficient = baseline_hours >= settings.drift_min_baseline_hours

    if sufficient:
        overall = "ok"
        for f in features.values():
            if _ORDER[f["severity"]] > _ORDER[overall]:
                overall = f["severity"]
    else:
        for f in features.values():
            f["severity"] = "ok"
        overall = "ok"

    return {
        "source": source.name,
        "window_hours": window,
        "features": features,
        "overall": overall,
        "drifted": overall != "ok",
        "sufficient_history": sufficient,
        "baseline_hours": round(baseline_hours, 1),
    }
