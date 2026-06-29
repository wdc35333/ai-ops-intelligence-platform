"""Detector recall against synthetic ground truth (injected anomalies)."""

import pandas as pd

from app.detectors import detect_isolation_forest, detect_seasonal_zscore
from app.features import SYSTEM_FEATURES, add_system_metric_features, aggregate_bookings_hourly
from app.sources import make_synthetic_bookings, make_synthetic_system_metrics


def test_isolation_forest_flags_injected_host_anomalies():
    raw, injected = make_synthetic_system_metrics(window_hours=72, seed=42)
    feats = add_system_metric_features(raw)
    out = detect_isolation_forest(feats, SYSTEM_FEATURES, contamination=0.02)

    flagged = set(out.index[out["is_anomaly"]].tolist())
    assert set(injected).issubset(flagged), (injected, sorted(flagged))


def test_isolation_forest_quiet_on_clean_data_has_low_rate():
    raw, _ = make_synthetic_system_metrics(window_hours=72, seed=1)
    feats = add_system_metric_features(raw)
    out = detect_isolation_forest(feats, SYSTEM_FEATURES, contamination=0.02)
    # contamination=2% → flagged fraction should stay small
    assert out["is_anomaly"].mean() <= 0.05


def test_seasonal_zscore_flags_revenue_spike():
    raw, injected = make_synthetic_bookings(window_hours=24 * 14, seed=7)
    hourly = aggregate_bookings_hourly(raw)
    out = detect_seasonal_zscore(hourly, "revenue", "bucket", z_thresh=3.5)

    flagged = {pd.Timestamp(t) for t in out.loc[out["is_anomaly"], "bucket"]}
    assert injected[0] in flagged, (injected, sorted(flagged))
