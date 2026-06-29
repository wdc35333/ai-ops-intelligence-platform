"""Drift detection: thresholds + flagging a shifted distribution."""

from app.config import get_settings
from app.drift import classify, compute_drift
from app.sources import get_source


def test_classify_thresholds():
    assert classify(0.0) == "ok"
    assert classify(0.15) == "warning"
    assert classify(0.40) == "critical"


def test_compute_drift_flags_rising_disk():
    settings = get_settings()  # synthetic source
    drift = compute_drift(get_source(settings), settings, window_hours=72)

    assert {"disk_used_pct", "mem_used_pct", "load1", "revenue"} <= set(drift["features"])
    # synthetic disk usage rises monotonically over the window → strong drift
    assert drift["features"]["disk_used_pct"]["severity"] in ("warning", "critical")
    assert drift["drifted"] is True
    assert drift["overall"] in ("warning", "critical")
