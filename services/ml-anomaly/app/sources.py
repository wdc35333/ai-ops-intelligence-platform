"""Data sources — a read-only abstraction over the kiosk operational data.

Two implementations:

* ``SyntheticSource`` — generates realistic, seasonally-varying series with a
  handful of injected anomalies. Lets the service (and reviewers) run with zero
  infrastructure, and gives the tests deterministic ground truth.
* ``DbSource`` — issues read-only ``SELECT``s against the live MariaDB.

Both return plain ``pandas`` frames with a stable column contract so the rest of
the pipeline is source-agnostic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol

import numpy as np
import pandas as pd

from .config import Settings


class DataSource(Protocol):
    name: str

    def system_metrics(self, window_hours: int) -> pd.DataFrame: ...
    def bookings(self, window_hours: int) -> pd.DataFrame: ...


# ──────────────────────────────────────────────────────────────────────────
# Synthetic source (offline / CI / portfolio demo)
# ──────────────────────────────────────────────────────────────────────────


def make_synthetic_system_metrics(
    window_hours: int = 168, seed: int = 42
) -> tuple[pd.DataFrame, list[int]]:
    """5-minute host metrics with daily seasonality + 3 injected anomalies.

    Returns ``(df, anomaly_row_positions)`` for use as test ground truth.
    """
    rng = np.random.default_rng(seed)
    n = max(window_hours * 12, 60)  # 12 samples/hour
    end = pd.Timestamp.now(tz=timezone.utc).floor("min").tz_localize(None)
    idx = pd.date_range(end=end, periods=n, freq="5min")
    hour = idx.hour + idx.minute / 60.0

    # Load average: higher during the day, noisy.
    load = 0.6 + 0.9 * np.clip(np.sin((hour - 6) / 24 * 2 * np.pi), 0, None)
    load = np.abs(load + rng.normal(0, 0.08, n))

    disk_total = 100 * 1e9
    disk_used = np.linspace(40e9, 56e9, n) + rng.normal(0, 0.3e9, n)
    mem_total = 4 * 1e9
    mem_used = 1.5e9 + 0.5e9 * (np.sin(hour / 24 * 2 * np.pi) + 1) / 2 + rng.normal(0, 0.08e9, n)

    df = pd.DataFrame(
        {
            "collected_at": idx,
            "disk_total_bytes": disk_total,
            "disk_free_bytes": disk_total - disk_used,
            "mem_total_bytes": mem_total,
            "mem_free_bytes": mem_total - mem_used,
            "load1": load,
        }
    )

    # Inject anomalies near the end of the window.
    anomalies = [n - 3, n - 25, n - 120]
    df.loc[anomalies[0], "load1"] = 7.2  # CPU load spike
    df.loc[anomalies[1], "disk_free_bytes"] = disk_total * 0.03  # disk nearly full
    df.loc[anomalies[2], "mem_free_bytes"] = mem_total * 0.02  # memory exhaustion
    return df, sorted(anomalies)


def make_synthetic_bookings(
    window_hours: int = 168, seed: int = 7
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Per-booking rows whose hourly revenue has daily/weekly seasonality.

    Returns ``(df, anomalous_hour_buckets)``.
    """
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now(tz=timezone.utc).floor("h").tz_localize(None)
    hours = pd.date_range(end=end, periods=window_hours, freq="h")

    rows: list[dict] = []
    anomaly_buckets: list[pd.Timestamp] = []
    for h in hours:
        # Expected bookings/hour: busy midday, quiet overnight, weekday lift.
        base = 1.0 + 3.0 * np.clip(np.sin((h.hour - 6) / 24 * 2 * np.pi), 0, None)
        if h.dayofweek < 5:
            base *= 1.3
        lam = max(base, 0.05)
        count = rng.poisson(lam)
        for _ in range(count):
            minute = int(rng.integers(0, 60))
            amount = int(rng.choice([2000, 3000, 4000, 6000], p=[0.4, 0.3, 0.2, 0.1]))
            rows.append(
                {
                    "created_at": h + pd.Timedelta(minutes=minute),
                    "amount": amount,
                    "status": "paid",
                    "refunded_at": pd.NaT,
                }
            )

    # Inject a revenue spike and a revenue drought.
    spike = hours[-6]
    for _ in range(40):
        rows.append(
            {
                "created_at": spike + pd.Timedelta(minutes=int(rng.integers(0, 60))),
                "amount": 6000,
                "status": "paid",
                "refunded_at": pd.NaT,
            }
        )
    anomaly_buckets.append(spike)
    # Drought: remove a busy midday hour entirely.
    drought_candidates = [h for h in hours if h.hour == 13 and h.dayofweek < 5]
    if drought_candidates:
        drought = drought_candidates[-1]
        rows = [r for r in rows if r["created_at"].floor("h") != drought]
        anomaly_buckets.append(drought)

    df = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
    return df, anomaly_buckets


class SyntheticSource:
    name = "synthetic"

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def system_metrics(self, window_hours: int) -> pd.DataFrame:
        return make_synthetic_system_metrics(window_hours, self._seed)[0]

    def bookings(self, window_hours: int) -> pd.DataFrame:
        return make_synthetic_bookings(window_hours, self._seed)[0]


# ──────────────────────────────────────────────────────────────────────────
# DB source (live, read-only)
# ──────────────────────────────────────────────────────────────────────────

_SYSTEM_METRICS_SQL = """
    SELECT collected_at, disk_total_bytes, disk_free_bytes,
           mem_total_bytes, mem_free_bytes, load1
      FROM system_metrics
     WHERE collected_at >= %(since)s
     ORDER BY collected_at
"""

# Real kiosk revenue (success payments) — the "A" component of the canonical
# A+B−C revenue model (see features/delivery/queries/revenueComponents.ts). A is
# ~99% of revenue (B/C are a few hundred rows all-time), a close-enough proxy for
# anomaly/drift detection. NOT online_booking, which is the sparse new online-
# prepayment channel. use_log_create_date is an epoch (seconds), so we filter on
# UNIX_TIMESTAMP and project it back to a datetime to keep the column contract.
_BOOKINGS_SQL = """
    SELECT FROM_UNIXTIME(use_log_create_date) AS created_at,
           amount,
           'paid' AS status,
           NULL   AS refunded_at
      FROM use_log_payment
     WHERE payment_status = 1 AND amount > 0
       AND use_log_create_date >= UNIX_TIMESTAMP(%(since)s)
     ORDER BY use_log_create_date
"""


class DbSource:
    name = "db"

    def __init__(self, database_url: str) -> None:
        # Imported lazily so the synthetic path needs no DB drivers.
        from sqlalchemy import create_engine

        self._engine = create_engine(database_url, pool_pre_ping=True)

    def _since(self, window_hours: int) -> datetime:
        return datetime.now() - timedelta(hours=window_hours)

    def system_metrics(self, window_hours: int) -> pd.DataFrame:
        return pd.read_sql(
            _SYSTEM_METRICS_SQL, self._engine, params={"since": self._since(window_hours)}
        )

    def bookings(self, window_hours: int) -> pd.DataFrame:
        return pd.read_sql(
            _BOOKINGS_SQL, self._engine, params={"since": self._since(window_hours)}
        )


def get_source(settings: Settings) -> DataSource:
    if settings.data_source == "db":
        if not settings.database_url:
            raise RuntimeError("MLA_DATA_SOURCE=db requires MLA_DATABASE_URL")
        return DbSource(settings.database_url)
    return SyntheticSource(settings.synthetic_seed)
