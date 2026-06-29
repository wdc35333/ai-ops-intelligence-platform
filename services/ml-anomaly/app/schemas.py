"""Pydantic response models — the service's public contract.

``/detect/*`` returns the detailed per-point view (for the dashboard / MLflow),
while ``/anomalies`` returns a compact, severity-ranked feed designed to be
consumed by the LLM ops agent as a tool result.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

Severity = Literal["info", "warning", "critical"]
DriftSeverity = Literal["ok", "warning", "critical"]
Domain = Literal["system", "bookings"]


class AnomalyPoint(BaseModel):
    timestamp: datetime
    value: float
    score: float
    is_anomaly: bool
    reason: Optional[str] = None


class DetectResponse(BaseModel):
    source: str
    detector: str
    domain: Domain
    window_hours: int
    total_points: int
    anomaly_count: int
    anomalies: list[AnomalyPoint]
    generated_at: datetime


class FeedItem(BaseModel):
    timestamp: datetime
    domain: Domain
    severity: Severity
    metric: str
    value: float
    score: float
    summary: str


class AnomalyFeed(BaseModel):
    """Agent-facing summary. Small, ranked, and human-readable (Korean)."""

    generated_at: datetime
    window_hours: int
    source: str
    counts: dict[str, int]
    items: list[FeedItem]


class DriftFeature(BaseModel):
    psi: float
    severity: DriftSeverity


class DriftResponse(BaseModel):
    generated_at: datetime
    source: str
    window_hours: int
    overall: DriftSeverity
    drifted: bool
    features: dict[str, DriftFeature]
