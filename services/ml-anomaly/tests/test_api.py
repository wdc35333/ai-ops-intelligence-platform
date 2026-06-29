"""API contract tests (synthetic data source)."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


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
