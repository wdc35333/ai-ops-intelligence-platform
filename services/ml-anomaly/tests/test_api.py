"""API contract tests (synthetic data source)."""

import pandas as pd
from fastapi.testclient import TestClient

from app.main import _system_feed_items, app

client = TestClient(app)


def _sys_frame(disk: float, mem: float, *, isolated: bool = False) -> pd.DataFrame:
    n = 20
    return pd.DataFrame(
        {
            "collected_at": pd.date_range("2026-06-01", periods=n, freq="5min"),
            "disk_used_pct": [disk] * n,
            "mem_used_pct": [mem] * n,
            "load1": [0.1] * n,
            "is_anomaly": [False] * (n - 1) + [isolated],
            "score": [0.1] * n,
        }
    )


def test_system_feed_silent_when_resources_healthy():
    # disk 28% / mem 60% is fine — even with an isolated point the agent feed
    # must stay empty (no daily false 'warning').
    assert _system_feed_items(_sys_frame(0.28, 0.60, isolated=True)) == []


def test_system_feed_flags_full_disk_even_if_steady():
    # A disk steadily at 93% never looks 'unusual' to IsolationForest, but it is
    # operationally critical → must surface on level alone.
    items = _system_feed_items(_sys_frame(0.93, 0.60))
    assert len(items) == 1
    assert items[0].metric == "disk_used_pct" and items[0].severity == "critical"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["data_source"] == "synthetic"


def test_detect_system_returns_anomalies():
    r = client.get("/detect/system-metrics", params={"window_hours": 72})
    assert r.status_code == 200
    assert r.json()["anomaly_count"] >= 1


def test_anomalies_feed_ranked_with_critical():
    r = client.get("/anomalies", params={"window_hours": 72})
    assert r.status_code == 200
    body = r.json()
    assert body["counts"]["total"] >= 1
    # the injected disk-full / memory-exhaustion rows must surface as critical
    assert any(i["severity"] == "critical" for i in body["items"])
    # feed must be severity-ranked (critical before warning)
    severities = [i["severity"] for i in body["items"]]
    assert severities == sorted(severities, key={"critical": 0, "warning": 1, "info": 2}.get)
