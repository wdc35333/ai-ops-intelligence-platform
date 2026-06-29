"""Drift detection: scale-robust PSI, detrending, baseline guard, flagging."""

import numpy as np

from app.config import get_settings
from app.drift import _drift_of, classify, compute_drift, psi
from app.sources import get_source


def test_classify_thresholds():
    assert classify(0.0) == "ok"
    assert classify(0.15) == "warning"
    assert classify(0.40) == "critical"


def test_psi_identical_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    assert psi(x, x) < 0.01


def test_psi_bounded_for_disjoint_samples():
    # A trending feature whose recent half sits entirely outside the older half's
    # range used to saturate at exactly 11.5128 (= -ln(1e-5)). Standardized fixed
    # bins with open tails keep PSI finite and well clear of that artifact.
    older = np.linspace(0.0, 1.0, 250)
    recent = np.linspace(5.0, 6.0, 250)
    val = psi(older, recent)
    assert np.isfinite(val)
    assert val < 11.0


def test_psi_is_scale_invariant():
    # The core fix: PSI now measures shift in units of natural variability, so it
    # does NOT depend on a feature's absolute scale. The same relative shift on a
    # tiny-variance feature (disk σ≈0.006) and a huge-variance one yields the same
    # PSI — quantile bins made the low-variance case explode.
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 1.0, 400)
    shifted = rng.normal(0.5, 1.0, 400)
    small = psi(base * 0.006 + 0.27, shifted * 0.006 + 0.27)  # disk-like
    large = psi(base * 100 + 5000, shifted * 100 + 5000)  # huge scale
    assert abs(small - large) < 1e-6
    assert small < 1.0  # a 0.5σ shift is modest, nowhere near the old 11.5


def test_monotonic_trend_is_not_drift():
    # A slowly filling disk (linear ramp + small noise) is steady capacity growth,
    # not a regime change → 'ok' after detrending.
    rng = np.random.default_rng(1)
    ramp = np.linspace(0.27, 0.29, 500) + rng.normal(0, 0.0005, 500)
    assert _drift_of(ramp)["severity"] == "ok"


def test_regime_change_is_flagged():
    # Same mean, but the recent half's variance blows up — a genuine regime change
    # detrending cannot remove, so drift must still fire.
    rng = np.random.default_rng(2)
    calm = rng.normal(0.5, 0.01, 250)
    volatile = rng.normal(0.5, 0.2, 250)
    assert _drift_of(np.concatenate([calm, volatile]))["severity"] in ("warning", "critical")


def test_insufficient_history_suppresses_drift():
    # Only 72h of synthetic history (< the 168h baseline) → must report
    # sufficient_history=False and stay 'ok' instead of crying wolf, even though
    # the raw per-feature psi values are still reported for transparency.
    settings = get_settings()  # synthetic source, drift_min_baseline_hours=168
    drift = compute_drift(get_source(settings), settings, window_hours=72)
    assert drift["sufficient_history"] is False
    assert drift["baseline_hours"] < 168
    assert drift["overall"] == "ok"
    assert all(f["severity"] == "ok" for f in drift["features"].values())


def test_sufficient_history_is_evaluated():
    # Two weeks of synthetic history clears the baseline guard → severities are
    # evaluated normally (synthetic injects anomalies, so it may flag).
    settings = get_settings()
    drift = compute_drift(get_source(settings), settings, window_hours=336)
    assert drift["sufficient_history"] is True
    assert drift["baseline_hours"] >= 168
    assert {"disk_used_pct", "mem_used_pct", "load1", "revenue"} <= set(drift["features"])
    for f in drift["features"].values():
        assert np.isfinite(f["psi"])
        assert f["psi"] < 11.0  # no feature may hit the old saturation artifact
