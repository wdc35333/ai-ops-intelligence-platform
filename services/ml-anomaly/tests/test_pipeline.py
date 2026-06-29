"""The MLflow pipeline logs runs + metrics (recall, drift) and a model artifact."""


def test_run_pipeline_logs_runs_metrics_and_drift(tmp_path, monkeypatch):
    uri = (tmp_path / "mlruns").as_uri()  # isolated file store
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    from app.pipeline import EXPERIMENT, run_pipeline

    summary = run_pipeline(window_hours=72)
    assert {"system", "bookings"} <= set(summary["runs"])
    assert isinstance(summary["runs"]["system"]["registered"], bool)

    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name(EXPERIMENT)
    assert exp is not None

    runs = client.search_runs([exp.experiment_id])
    names = {r.data.tags.get("mlflow.runName") for r in runs}
    assert {"system-anomaly", "bookings-anomaly"} <= names

    sys_run = next(r for r in runs if r.data.tags.get("mlflow.runName") == "system-anomaly")
    assert "recall_synthetic" in sys_run.data.metrics  # measured against ground truth
    assert any(k.startswith("psi_") for k in sys_run.data.metrics)  # drift indicator
    assert sys_run.data.params["detector"] == "isolation_forest"
