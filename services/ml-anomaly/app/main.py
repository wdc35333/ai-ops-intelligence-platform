"""FastAPI entrypoint.

Endpoints
---------
GET /health                      liveness + active data source
GET /detect/system-metrics       detailed host-metric anomalies (IsolationForest)
GET /detect/bookings             detailed revenue anomalies (seasonal z-score)
GET /anomalies                   compact, severity-ranked feed (LLM-agent facing)

Everything is read-only. ``/anomalies`` is the contract the ops agent's
``get_ml_anomalies`` tool calls.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, Query

from . import __version__
from .config import Settings, get_settings
from .detectors import detect_isolation_forest, detect_seasonal_zscore
from .drift import compute_drift
from .features import SYSTEM_FEATURES, add_system_metric_features, aggregate_bookings_hourly
from .schemas import (
    AnomalyFeed,
    AnomalyPoint,
    DetectResponse,
    DriftFeature,
    DriftResponse,
    FeedItem,
)
from .sources import DataSource, get_source

app = FastAPI(title="kiosk ML anomaly service", version=__version__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── detection runners ─────────────────────────────────────────────────────


def _run_system(source: DataSource, settings: Settings, window_hours: int) -> pd.DataFrame:
    raw = source.system_metrics(window_hours)
    feats = add_system_metric_features(raw)
    return detect_isolation_forest(feats, SYSTEM_FEATURES, contamination=settings.if_contamination)


def _run_bookings(source: DataSource, settings: Settings, window_hours: int) -> pd.DataFrame:
    raw = source.bookings(window_hours)
    hourly = aggregate_bookings_hourly(raw)
    if hourly.empty:
        return hourly.assign(score=[], is_anomaly=[], reason=[])
    return detect_seasonal_zscore(
        hourly, value_col="revenue", time_col="bucket", z_thresh=settings.booking_z_thresh
    )


def _system_value(row: pd.Series) -> float:
    metric = (row["reason"] or "load1").split()[0]
    return float(row.get(metric, row["load1"]))


# ── endpoints ─────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {"status": "ok", "version": __version__, "data_source": settings.data_source}


@app.get("/detect/system-metrics", response_model=DetectResponse)
def detect_system(window_hours: int = Query(default=None, ge=6, le=2160)) -> DetectResponse:
    settings = get_settings()
    window = window_hours or settings.default_window_hours
    source = get_source(settings)
    df = _run_system(source, settings, window)
    flagged = df[df["is_anomaly"]]
    return DetectResponse(
        source=source.name,
        detector="isolation_forest",
        domain="system",
        window_hours=window,
        total_points=len(df),
        anomaly_count=int(len(flagged)),
        anomalies=[
            AnomalyPoint(
                timestamp=row["collected_at"].to_pydatetime(),
                value=round(_system_value(row), 4),
                score=float(row["score"]),
                is_anomaly=True,
                reason=row["reason"],
            )
            for _, row in flagged.iterrows()
        ],
        generated_at=_now(),
    )


@app.get("/detect/bookings", response_model=DetectResponse)
def detect_bookings(window_hours: int = Query(default=None, ge=24, le=2160)) -> DetectResponse:
    settings = get_settings()
    window = window_hours or settings.default_window_hours
    source = get_source(settings)
    df = _run_bookings(source, settings, window)
    flagged = df[df["is_anomaly"]] if not df.empty else df
    return DetectResponse(
        source=source.name,
        detector="seasonal_zscore",
        domain="bookings",
        window_hours=window,
        total_points=len(df),
        anomaly_count=int(len(flagged)),
        anomalies=[
            AnomalyPoint(
                timestamp=row["bucket"].to_pydatetime(),
                value=float(row["revenue"]),
                score=float(row["score"]),
                is_anomaly=True,
                reason=row["reason"],
            )
            for _, row in flagged.iterrows()
        ],
        generated_at=_now(),
    )


@app.get("/anomalies", response_model=AnomalyFeed)
def anomalies(window_hours: int = Query(default=None, ge=24, le=2160)) -> AnomalyFeed:
    """Compact, ranked feed for the LLM ops agent."""
    settings = get_settings()
    window = window_hours or settings.default_window_hours
    source = get_source(settings)

    items: list[FeedItem] = []

    sys_df = _run_system(source, settings, window)
    for _, row in sys_df[sys_df["is_anomaly"]].iterrows():
        metric = (row["reason"] or "load1").split()[0]
        severity = (
            "critical"
            if (row["disk_used_pct"] >= 0.9 or row["mem_used_pct"] >= 0.95)
            else "warning"
        )
        items.append(
            FeedItem(
                timestamp=row["collected_at"].to_pydatetime(),
                domain="system",
                severity=severity,
                metric=metric,
                value=round(_system_value(row), 4),
                score=float(row["score"]),
                summary=f"[시스템] {row['reason']}",
            )
        )

    bk_df = _run_bookings(source, settings, window)
    if not bk_df.empty:
        for _, row in bk_df[bk_df["is_anomaly"]].iterrows():
            items.append(
                FeedItem(
                    timestamp=row["bucket"].to_pydatetime(),
                    domain="bookings",
                    severity="critical" if row["score"] >= 6 else "warning",
                    metric="revenue",
                    value=float(row["revenue"]),
                    score=float(row["score"]),
                    summary=f"[예약/매출] {row['reason']}",
                )
            )

    rank = {"critical": 0, "warning": 1, "info": 2}
    items.sort(key=lambda i: (rank[i.severity], -i.score))

    counts = {
        "total": len(items),
        "critical": sum(i.severity == "critical" for i in items),
        "warning": sum(i.severity == "warning" for i in items),
    }
    return AnomalyFeed(
        generated_at=_now(),
        window_hours=window,
        source=source.name,
        counts=counts,
        items=items,
    )


@app.post("/pipeline/run")
def pipeline_run(window_hours: int = Query(default=None, ge=6, le=2160)) -> dict:
    """Train/score + log to MLflow (called by Airflow on a schedule). MLOps L3."""
    from .pipeline import run_pipeline

    return run_pipeline(window_hours)


@app.get("/drift", response_model=DriftResponse)
def drift(window_hours: int = Query(default=None, ge=12, le=2160)) -> DriftResponse:
    """Per-feature distribution drift (PSI, reference vs current). MLOps L5."""
    settings = get_settings()
    window = window_hours or settings.default_window_hours
    source = get_source(settings)
    d = compute_drift(source, settings, window)
    return DriftResponse(
        generated_at=_now(),
        source=d["source"],
        window_hours=d["window_hours"],
        overall=d["overall"],
        drifted=d["drifted"],
        features={k: DriftFeature(**v) for k, v in d["features"].items()},
    )
