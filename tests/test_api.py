from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import app
from train_stoich_ode import audit_dataset_errors_and_null_recovery


client = TestClient(app)


def test_health_and_model_list_are_ready():
    health = client.get("/api/health")
    models = client.get("/api/models")

    assert health.status_code == 200
    assert health.json()["status"] == "ready"
    assert models.status_code == 200
    assert {item["id"] for item in models.json()["models"]} == {"M1", "M2", "M3", "M4"}


def test_sample_csv_is_served_for_browser_data_preview():
    response = client.get("/kaggle_synthetic_reaction_data/completion/test.csv")

    assert response.status_code == 200
    assert response.text.splitlines()[0] == "#,mass A,mass B,mass C,mass D"
    assert len(response.text.splitlines()) == 201


def test_default_prediction_uses_small_euler_step_and_conserves_mass():
    response = client.post(
        "/api/predict",
        json={"model": "M4", "initial_state": [1, 1, 0, 0], "end_time": 20},
    )

    assert response.status_code == 200
    payload = response.json()
    rows = payload["predictions"]
    assert len(rows) == 201
    assert rows[0]["t"] == 0
    assert rows[-1]["t"] == pytest.approx(20)
    assert rows[1]["t"] - rows[0]["t"] == pytest.approx(0.1)
    totals = np.array([row["total_mass"] for row in rows])
    assert np.max(np.abs(totals - 2.0)) < 1e-12
    assert payload["summary"]["warnings"] == []


def test_invalid_initial_state_returns_422():
    response = client.post(
        "/api/predict",
        json={"model": "M4", "initial_state": [1, 1, 0], "end_time": 10},
    )

    assert response.status_code == 422


def test_negative_model_prediction_returns_explicit_warning():
    response = client.post(
        "/api/predict",
        json={
            "model": "M1",
            "initial_state": [1_000_000, 0, 0, 0],
            "end_time": 100,
            "step_size": 1,
        },
    )

    assert response.status_code == 200
    warnings = response.json()["summary"]["warnings"]
    assert warnings
    assert "negative species mass" in warnings[0]
    assert response.json()["summary"]["minimum_predicted_species_mass_g"] < 0


def test_sensor_dropout_audit_measures_values_returned_by_csv_loader():
    data_dir = Path(__file__).resolve().parents[1] / "kaggle_synthetic_reaction_data"
    audit = audit_dataset_errors_and_null_recovery(data_dir)

    assert audit["null_diagnostics"]["mass_balance_imputed"] == 8
    assert audit["null_diagnostics"]["irregular_gap_rows_dropped"] == 2
    assert audit["max_mass_balance_imputation_error_g"] < 1e-15
