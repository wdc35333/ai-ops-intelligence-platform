"""Feature engineering — turn raw rows into model-ready frames."""

from __future__ import annotations

import numpy as np
import pandas as pd

SYSTEM_FEATURES = ["disk_used_pct", "mem_used_pct", "load1"]


def add_system_metric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive utilisation ratios from the raw ``system_metrics`` columns."""
    out = df.copy()
    out["collected_at"] = pd.to_datetime(out["collected_at"])
    out = out.sort_values("collected_at").reset_index(drop=True)

    disk_total = out["disk_total_bytes"].astype(float).replace(0, np.nan)
    mem_total = out["mem_total_bytes"].astype(float).replace(0, np.nan)
    out["disk_used_pct"] = (1 - out["disk_free_bytes"].astype(float) / disk_total).clip(0, 1)
    out["mem_used_pct"] = (1 - out["mem_free_bytes"].astype(float) / mem_total).clip(0, 1)
    out["load1"] = out["load1"].astype(float)
    out[SYSTEM_FEATURES] = out[SYSTEM_FEATURES].ffill().fillna(0.0)
    return out


def aggregate_bookings_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw payment rows into an hourly revenue series.

    Returns one row per hour with gross booking ``revenue`` (KRW), booking
    ``count`` and ``refund_count``. Empty hours inside the window are filled
    with zero so the seasonal detector sees a regular series.
    """
    out = df.copy()
    if out.empty:
        return pd.DataFrame(columns=["bucket", "revenue", "count", "refund_count"])

    out["created_at"] = pd.to_datetime(out["created_at"])
    out["bucket"] = out["created_at"].dt.floor("h")
    out["amount"] = out["amount"].astype(float)
    refunded = pd.to_datetime(out.get("refunded_at"), errors="coerce")

    grouped = out.groupby("bucket").agg(
        revenue=("amount", "sum"),
        count=("amount", "size"),
    )
    refund_counts = (
        refunded.dt.floor("h").value_counts().rename("refund_count")
        if refunded.notna().any()
        else pd.Series(dtype="int64", name="refund_count")
    )
    grouped = grouped.join(refund_counts).fillna({"refund_count": 0})

    full_index = pd.date_range(out["bucket"].min(), out["bucket"].max(), freq="h")
    grouped = grouped.reindex(full_index, fill_value=0)
    grouped.index.name = "bucket"
    return grouped.reset_index()
