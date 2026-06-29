"""Anomaly detectors.

Two complementary unsupervised techniques:

* ``detect_isolation_forest`` — multivariate outliers in host metrics
  (disk/mem/load move together; a joint model catches combinations a single
  threshold would miss).
* ``detect_seasonal_zscore`` — robust, season-aware deviation for a univariate
  series such as hourly revenue, where "normal" depends on the hour of day.

Both return the input frame with three added columns: ``score`` (higher = more
anomalous), ``is_anomaly`` (bool) and ``reason`` (a short Korean explanation the
LLM agent can relay verbatim).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

_MAD_SCALE = 1.4826  # makes MAD a consistent estimator of std for normal data


def _robust_z(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (median, robust-z) for a 1-D array using median/MAD."""
    med = np.median(values, axis=0)
    mad = np.median(np.abs(values - med), axis=0)
    mad = np.where(mad == 0, 1e-9, mad)
    return med, np.abs(values - med) / (_MAD_SCALE * mad)


def detect_isolation_forest(
    df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.02,
    random_state: int = 42,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    if len(out) < 12:
        out["score"], out["is_anomaly"], out["reason"] = 0.0, False, None
        return out

    X = out[feature_cols].astype(float).to_numpy()
    model = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=random_state
    )
    model.fit(X)
    # decision_function: higher = more normal → negate so higher = more anomalous.
    out["score"] = (-model.decision_function(X)).round(4)
    out["is_anomaly"] = model.predict(X) == -1

    _, z = _robust_z(X)
    top = z.argmax(axis=1)
    reasons: list[str | None] = []
    for r in range(len(out)):
        if out["is_anomaly"].iloc[r]:
            col = feature_cols[top[r]]
            reasons.append(f"{col} 이상 (robust-z={z[r, top[r]]:.1f})")
        else:
            reasons.append(None)
    out["reason"] = reasons
    return out


def detect_seasonal_zscore(
    df: pd.DataFrame,
    value_col: str,
    time_col: str,
    z_thresh: float = 3.5,
    season: str = "hour",
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    if len(out) < 24:
        out["score"], out["is_anomaly"], out["reason"] = 0.0, False, None
        return out

    t = pd.to_datetime(out[time_col])
    if season == "dow_hour":
        key = t.dt.dayofweek * 24 + t.dt.hour
    else:  # "hour"
        key = t.dt.hour
    out["_key"] = key.to_numpy()

    vals = out[value_col].astype(float)
    grp = vals.groupby(out["_key"])
    med = grp.transform("median")
    mad = grp.transform(lambda s: np.median(np.abs(s - np.median(s)))).replace(0, np.nan)
    global_mad = np.median(np.abs(vals - vals.median())) or 1e-9
    mad = mad.fillna(global_mad)

    z = (vals - med).abs() / (_MAD_SCALE * mad)
    out["score"] = z.round(3)
    out["is_anomaly"] = z >= z_thresh

    reasons: list[str | None] = []
    for r in range(len(out)):
        if out["is_anomaly"].iloc[r]:
            direction = "급증" if vals.iloc[r] > med.iloc[r] else "급감"
            reasons.append(
                f"{value_col} {direction} (z={z.iloc[r]:.1f}, 기대≈{med.iloc[r]:.0f})"
            )
        else:
            reasons.append(None)
    out["reason"] = reasons
    return out.drop(columns=["_key"])
