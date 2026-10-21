#!/usr/bin/env python3
"""Local FastAPI service for predictions from the fitted STOICH-ODE models."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from train_stoich_ode import (
    BaselineUnconstrainedNeuralODE,
    ICLR2023TwoStageNeuralODE,
    StoichiometricFluxNeuralODE,
    SymbolicMechanisticNeuralODE,
)


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "stoich_ode_models.joblib"

MODEL_NAMES = {
    "M1": "Unconstrained Neural ODE",
    "M2": "Simplified Two-Stage ODE",
    "M3": "Stoichiometric Flux Neural ODE",
    "M4": "Sparse Mechanistic ODE",
}

app = FastAPI(
    title="STOICH-ODE Local Model API",
    description=(
        "Runs the fitted reaction-dynamics models saved by train_stoich_ode.py. "
        "The browser console is served by this same local process."
    ),
    version="1.0.0",
)
app.mount(
    "/kaggle_synthetic_reaction_data",
    StaticFiles(directory=ROOT / "kaggle_synthetic_reaction_data"),
    name="benchmark-data",
)


class PredictionRequest(BaseModel):
    model: Literal["M1", "M2", "M3", "M4"] = "M4"
    initial_state: list[float] = Field(default=[1.0, 1.0, 0.0, 0.0])
    start_time: float = Field(default=0.0, ge=0.0)
    end_time: float = Field(default=20.0, gt=0.0, le=200.0)
    step_size: float = Field(default=0.1, gt=0.0, le=10.0, description="Maximum time spacing between returned prediction rows.")


@lru_cache(maxsize=1)
def load_model_bundle():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Fitted model artifact not found at {MODEL_PATH}. "
            "Run `python train_stoich_ode.py` from the project folder first."
        )

    artifact = joblib.load(MODEL_PATH)
    if artifact.get("format_version") != 1:
        raise ValueError("Unsupported model artifact format version.")

    states = artifact.get("models", {})
    required = {"M1", "M2", "M3", "M4"}
    if set(states) != required:
        raise ValueError("The model artifact must contain fitted states for M1, M2, M3, and M4.")

    m1_state = states["M1"]
    m1 = BaselineUnconstrainedNeuralODE(hidden_dim=int(m1_state["hidden_dim"]))
    m1.p = np.asarray(m1_state["p"], dtype=np.float64)

    m2_state = states["M2"]
    m2 = ICLR2023TwoStageNeuralODE(hidden_dim=int(m2_state["hidden_dim"]))
    m2.p = np.asarray(m2_state["p"], dtype=np.float64)

    m3_state = states["M3"]
    m3 = StoichiometricFluxNeuralODE(hidden_dim=int(m3_state["hidden_dim"]))
    m3.p = np.asarray(m3_state["p"], dtype=np.float64)

    m4_state = states["M4"]
    m4 = SymbolicMechanisticNeuralODE()
    m4.K = np.asarray(m4_state["K"], dtype=np.float64)
    m4.feature_names = list(m4_state.get("feature_names", m4.feature_names))

    models = {"M1": m1, "M2": m2, "M3": m3, "M4": m4}
    return artifact, models


def api_error_message(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return str(exc)
    return f"Could not load fitted model artifact: {exc}"


@app.get("/", include_in_schema=False)
def serve_console():
    return FileResponse(ROOT / "index.html")


@app.get("/api/health")
def health():
    try:
        artifact, _ = load_model_bundle()
    except Exception as exc:
        return {
            "status": "model_not_ready",
            "artifact_exists": MODEL_PATH.is_file(),
            "model_artifact": str(MODEL_PATH),
            "message": api_error_message(exc),
        }
    return {
        "status": "ready",
        "artifact_exists": True,
        "model_artifact": "models/stoich_ode_models.joblib",
        "trained_at_utc": artifact.get("created_at_utc"),
        "training_split": artifact.get("training_split"),
        "training_rows": artifact.get("training_rows"),
        "training_end_s": artifact.get("training_end_s"),
        "models": list(MODEL_NAMES),
    }


@app.get("/api/models")
def list_models():
    try:
        artifact, _ = load_model_bundle()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=api_error_message(exc)) from exc
    return {
        "models": [
            {"id": key, "name": name}
            for key, name in MODEL_NAMES.items()
        ],
        "trained_at_utc": artifact.get("created_at_utc"),
        "training_split": artifact.get("training_split"),
        "training_rows": artifact.get("training_rows"),
        "training_end_s": artifact.get("training_end_s"),
    }


@app.post("/api/predict")
def predict(request: PredictionRequest):
    if len(request.initial_state) != 4:
        raise HTTPException(
            status_code=422,
            detail="initial_state must contain [A, B, C, D] values.",
        )
    y0 = np.asarray(request.initial_state, dtype=np.float64)
    if not np.all(np.isfinite(y0)) or np.any(y0 < 0.0):
        raise HTTPException(status_code=422, detail="Initial species masses must be finite and non-negative.")
    if request.end_time <= request.start_time:
        raise HTTPException(status_code=422, detail="end_time must be greater than start_time.")

    interval_count = int(math.ceil((request.end_time - request.start_time) / request.step_size))
    if interval_count > 2000:
        raise HTTPException(status_code=422, detail="Choose a larger step_size; at most 2,000 intervals are allowed.")

    try:
        artifact, models = load_model_bundle()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=api_error_message(exc)) from exc

    times = np.linspace(request.start_time, request.end_time, interval_count + 1, dtype=np.float64)
    try:
        predicted = np.asarray(models[request.model].integrate(y0, times), dtype=np.float64)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Model integration failed: {exc}") from exc
    if predicted.shape != (len(times), 4) or not np.all(np.isfinite(predicted)):
        raise HTTPException(
            status_code=422,
            detail=f"{request.model} became numerically unstable over this horizon. Try a shorter horizon or a different model.",
        )

    rows = [
        {
            "t": float(t),
            "A": float(state[0]),
            "B": float(state[1]),
            "C": float(state[2]),
            "D": float(state[3]),
            "total_mass": float(np.sum(state)),
        }
        for t, state in zip(times, predicted)
    ]
    initial_total = float(np.sum(y0))
    max_mass_drift = float(np.max(np.abs(np.sum(predicted, axis=1) - initial_total)))
    minimum_mass = float(np.min(predicted))
    negative_mass_tolerance = 1e-10 * max(1.0, abs(initial_total))
    warnings = []
    if minimum_mass < -negative_mass_tolerance:
        warnings.append(
            "The model predicted a negative species mass "
            f"(minimum {minimum_mass:.6g} g). This trajectory is outside the non-negative mass domain; "
            "try a smaller step, a shorter horizon, or a different model."
        )

    result = {
        "model": request.model,
        "model_name": MODEL_NAMES[request.model],
        "source": "fitted Python model loaded from models/stoich_ode_models.joblib",
        "training": {
            "training_split": artifact.get("training_split"),
            "training_rows": artifact.get("training_rows"),
            "training_end_s": artifact.get("training_end_s"),
            "trained_at_utc": artifact.get("created_at_utc"),
        },
        "summary": {
            "rows": len(rows),
            "initial_total_mass_g": initial_total,
            "max_mass_drift_g": max_mass_drift,
            "minimum_predicted_species_mass_g": minimum_mass,
            "warnings": warnings,
        },
        "predictions": rows,
    }
    if request.model == "M4":
        rates = models["M4"].K
        result["discovered_rates"] = {
            "k1_for_A_plus_B_to_C_plus_D": float(rates[0, 0]),
            "k2_for_C_to_B": float(rates[1, 1]),
            "feature_names": list(models["M4"].feature_names),
        }
    return result
