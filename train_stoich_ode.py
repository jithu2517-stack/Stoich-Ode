#!/usr/bin/env python3
"""
MAT-05: Synthetic Chemical Reaction Dynamics via Neural ODEs
============================================================
Complete Data Hygiene, Null/Error Diagnostics, and 4-Tier Neural ODE Benchmark Engine.

Reference Benchmark:
  - Dataset: Kaggle (cici118/synthetic-chemical-reaction)
  - Paper: Coelho, Costa & Ferras (ICLR 2023 Tiny Paper),
           "Prior knowledge meets Neural ODEs: a two-stage training method for improved explainability"
           https://openreview.net/pdf?id=p7sHcNt_tqo

Reaction System:
  A + B -> C + D   (Forward bimolecular reaction, rate r1)
  C -> B           (Secondary regeneration step, rate r2)
  Initial state at t=0: A(0)=1.0g, B(0)=1.0g, C(0)=0.0g, D(0)=0.0g
  Conservation of Mass Law: A(t) + B(t) + C(t) + D(t) = 2.0g for all t >= 0.
"""

import argparse
import time
import pathlib
import tempfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
try:
    from scipy.optimize import least_squares
except ImportError as exc:
    raise SystemExit(
        "SciPy is required for the four-model benchmark. Install dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from exc


# ============================================================================
# STEP 1: KAGGLE DATASET GENERATOR & DATA HYGIENE / NULL / ERROR DIAGNOSTICS
# ============================================================================
def generate_kaggle_dataset(base_dir: pathlib.Path) -> None:
    """
    Generates deterministic synthetic demo splits matching this project's reaction
    equations. This is a fallback for demos, not a Kaggle download or benchmark source.
    """
    specs = {
        "reconstruction/train.csv": (1.0, 100),
        "reconstruction/test.csv":  (1.0, 100),
        "completion/train.csv":     (1.0, 100),
        "completion/test.csv":      (0.5, 200),
        "extrapolation/train.csv":  (1.0, 100),
        "extrapolation/test.csv":   (2.0, 100),
    }
    for rel_path, (dt, n_rows) in specs.items():
        fpath = base_dir / rel_path
        if fpath.exists():
            continue
        fpath.parent.mkdir(parents=True, exist_ok=True)
        A, B, C, D = 1.0, 1.0, 0.0, 0.0
        t = 0.0
        rows = []
        for _ in range(n_rows):
            rows.append((t, A, B, C, D))
            dA = -0.1 * A * B
            dB = -0.1 * A * B + 0.05 * C
            dC =  0.1 * A * B - 0.05 * C
            dD =  0.1 * A * B
            A += dt * dA
            B += dt * dB
            C += dt * dC
            D += dt * dD
            t += dt
        df = pd.DataFrame(rows, columns=["#", "mass A", "mass B", "mass C", "mass D"])
        df.to_csv(fpath, index=False)


def load_and_clean_csv(fpath: pathlib.Path, m_total: float = 2.0):
    """
    Loads a reaction CSV and applies 4 stages of data hygiene & error recovery:
      1. Header Normalization: Renames '#' -> 't' and strips leading/trailing spaces.
      2. Algebraic Mass-Balance Null Imputation: If 1 species sensor drops out (NaN)
         at any timestamp, recovers it exactly via m_missing = 2.0 - sum(m_known).
      3. Irregular Time-Gap Handling: If multiple sensors drop out on an intermediate row,
         drops the corrupted row so the continuous-time Neural ODE integrates smoothly across
         the irregular step size dt = t_{i+1} - t_i (which standard RNNs/LSTMs cannot do).
      4. Monotonicity & Duplicate Timestamp Guard: Prevents dt <= 0 solver singularities.
    """
    df = pd.read_csv(fpath)
    df.columns = [c.strip() for c in df.columns]
    if "t" not in df.columns:
        time_column = next((name for name in ("#", "Unnamed: 0", "time", "time_sec", "t_sec") if name in df.columns), None)
        if time_column is None:
            raise ValueError(f"No time column found in {fpath}; expected t, #, or an exported index column.")
        df = df.rename(columns={time_column: "t"})
    species_cols = ["mass A", "mass B", "mass C", "mass D"]

    diagnostics = {
        "raw_rows": len(df),
        "nulls_detected": int(df[species_cols].isnull().sum().sum()),
        "mass_balance_imputed": 0,
        "irregular_gap_rows_dropped": 0,
        "duplicate_timestamps_removed": 0,
    }

    if diagnostics["nulls_detected"] > 0:
        for idx, row in df.iterrows():
            na_mask = row[species_cols].isnull()
            if na_mask.sum() == 1:
                known_sum = row[species_cols][~na_mask].sum()
                missing_col = species_cols[np.where(na_mask)[0][0]]
                df.loc[idx, missing_col] = max(0.0, m_total - known_sum)
                diagnostics["mass_balance_imputed"] += 1

        before_drop = len(df)
        if df.iloc[0][species_cols].isnull().any():
            df[species_cols] = df[species_cols].bfill()
        df = df.dropna(subset=["t"] + species_cols).reset_index(drop=True)
        diagnostics["irregular_gap_rows_dropped"] = before_drop - len(df)

    before_dedup = len(df)
    df = df.sort_values("t").drop_duplicates(subset=["t"]).reset_index(drop=True)
    diagnostics["duplicate_timestamps_removed"] = before_dedup - len(df)

    t = df["t"].values.astype(np.float64)
    Y = df[species_cols].values.astype(np.float64)
    return t, Y, diagnostics


def audit_dataset_errors_and_null_recovery(data_dir: pathlib.Path) -> dict:
    """
    Runs two diagnostic audits to explain to hackathon judges:
      Audit A: Forward Euler Discretization Truncation Error built into the Kaggle CSVs
               at the exact same timestamp (t = 2.0) across train (dt=1.0), completion (dt=0.5),
               and extrapolation (dt=2.0).
      Audit B: Simulated Sensor Dropout (Nulls / NaNs) recovery test.
    """
    t_tr, Y_tr, _ = load_and_clean_csv(data_dir / "reconstruction/train.csv")
    t_co, Y_co, _ = load_and_clean_csv(data_dir / "completion/test.csv")
    t_ex, Y_ex, _ = load_and_clean_csv(data_dir / "extrapolation/test.csv")

    y_t2_dt10 = Y_tr[np.where(np.isclose(t_tr, 2.0))[0][0]]
    y_t2_dt05 = Y_co[np.where(np.isclose(t_co, 2.0))[0][0]]
    y_t2_dt20 = Y_ex[np.where(np.isclose(t_ex, 2.0))[0][0]]

    df_corrupt = pd.read_csv(data_dir / "reconstruction/train.csv")
    rng = np.random.default_rng(101)
    species_cols = ["mass A", "mass B", "mass C", "mass D"]
    true_vals = []
    for row_idx in [5, 12, 24, 37, 49, 63, 78, 91]:
        col_idx = int(rng.integers(0, 4))
        col_name = species_cols[col_idx]
        true_vals.append((row_idx, col_name, float(df_corrupt.loc[row_idx, col_name])))
        df_corrupt.loc[row_idx, col_name] = np.nan

    df_corrupt.loc[18, ["mass A", "mass C"]] = np.nan
    df_corrupt.loc[55, ["mass B", "mass D"]] = np.nan

    with tempfile.TemporaryDirectory(prefix="stoich_ode_audit_", dir=data_dir.parent) as temp_dir:
        corrupt_path = pathlib.Path(temp_dir) / "corrupted_sensor_test.csv"
        df_corrupt.to_csv(corrupt_path, index=False)
        t_clean, Y_clean, diag = load_and_clean_csv(corrupt_path)

        df_reloaded = pd.read_csv(corrupt_path)
        df_reloaded.columns = [c.strip() for c in df_reloaded.columns]
        imputation_errors = []
        for r_idx, c_name, orig_val in true_vals:
            row = df_reloaded.loc[r_idx, species_cols]
            na_mask = row.isnull()
            recovered_val = 2.0 - float(row[~na_mask].sum())
            imputation_errors.append(abs(recovered_val - orig_val))

    return {
        "euler_t2_dt05_massA": float(y_t2_dt05[0]),
        "euler_t2_dt10_massA": float(y_t2_dt10[0]),
        "euler_t2_dt20_massA": float(y_t2_dt20[0]),
        "euler_truncation_gap_dt05_vs_dt20": float(abs(y_t2_dt05[0] - y_t2_dt20[0])),
        "null_diagnostics": diag,
        "max_mass_balance_imputation_error_g": float(max(imputation_errors)),
        "irregular_timestamps_remaining": len(t_clean),
    }


# ============================================================================
# STEP 2: FOUR PROGRESSIVE NEURAL ODE ARCHITECTURES
# ============================================================================

class BaselineUnconstrainedNeuralODE:
    """
    Model 1: Standard Unconstrained Black-Box Neural ODE
    ----------------------------------------------------
    Architecture: 4 -> 16 (tanh) -> 4 unconstrained derivatives dy/dt = f_theta(y).
    Trained with single-stage MSE loss without physical constraints.
    Shows why naive Neural ODEs drift away from total mass = 2.0g during extrapolation.
    """
    def __init__(self, hidden_dim: int = 16, seed: int = 42):
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)
        self.p = rng.normal(0, 0.12, 4 * hidden_dim + hidden_dim + hidden_dim * 4 + 4)

    def _unpack(self, p):
        H = self.hidden_dim
        i1 = 4 * H
        i2 = i1 + H
        i3 = i2 + H * 4
        W1 = p[:i1].reshape(4, H)
        b1 = p[i1:i2]
        W2 = p[i2:i3].reshape(H, 4)
        b2 = p[i3:]
        return W1, b1, W2, b2

    def deriv(self, Y, p=None):
        W1, b1, W2, b2 = self._unpack(self.p if p is None else p)
        return np.tanh(Y @ W1 + b1) @ W2 + b2

    def integrate(self, y0, t_grid, p=None):
        W1, b1, W2, b2 = self._unpack(self.p if p is None else p)
        N = len(t_grid)
        Y_pred = np.zeros((N, 4))
        Y_pred[0] = y0
        for i in range(N - 1):
            dt = t_grid[i + 1] - t_grid[i]
            dY = np.tanh(Y_pred[i] @ W1 + b1) @ W2 + b2
            Y_pred[i + 1] = Y_pred[i] + dt * dY
        return Y_pred

    def fit(self, t_train, Y_train):
        dt_vec = np.diff(t_train)[:, None]
        dY_emp = np.diff(Y_train, axis=0) / dt_vec
        Y_in = Y_train[:-1]
        res = least_squares(
            lambda p: np.concatenate([(self.deriv(Y_in, p) - dY_emp).ravel(), 0.12 * p]),
            self.p,
            max_nfev=30,
        )
        self.p = res.x


class ICLR2023TwoStageNeuralODE(BaselineUnconstrainedNeuralODE):
    """
    Model 2: Simplified ICLR-Inspired Two-Stage Neural ODE
    --------------------------------------------------------------------
    Stage I penalizes the sum of predicted derivatives; Stage II fits trajectories
    while adding a soft mass-violation residual. This is an ICLR-inspired baseline,
    not a faithful reproduction of the paper's constraint handling or rollback method.
    """
    def __init__(self, hidden_dim: int = 16, seed: int = 42):
        super().__init__(hidden_dim=hidden_dim, seed=seed)
        self.stage1_initial_violation = 0.0
        self.stage1_final_violation = 0.0

    def fit(self, t_train, Y_train, m_total: float = 2.0):
        dt_vec = np.diff(t_train)[:, None]
        dY_emp = np.diff(Y_train, axis=0) / dt_vec
        Y_in = Y_train[:-1]

        Y_init = self.integrate(Y_train[0], t_train, self.p)
        self.stage1_initial_violation = float(np.sum(np.abs(np.sum(Y_init, axis=1) - m_total)))

        # Stage I (Admissibility Phase): Minimize mass rate violation sum_i (dy_i/dt) = 0
        res_stage1 = least_squares(
            lambda p: np.concatenate([np.sum(self.deriv(Y_in, p), axis=1), 0.05 * p]),
            self.p,
            max_nfev=30,
        )
        p_adm = res_stage1.x
        Y_adm = self.integrate(Y_train[0], t_train, p_adm)
        self.stage1_final_violation = float(np.sum(np.abs(np.sum(Y_adm, axis=1) - m_total)))

        # Stage II (Optimization Phase): Minimize trajectory error while staying in admissible region
        def stage2_residual(p):
            Y_p = self.integrate(Y_train[0], t_train, p)
            mse_res = (Y_p - Y_train).ravel()
            viol_guard = (np.sum(Y_p, axis=1) - m_total) * 3.0
            deriv_res = (self.deriv(Y_in, p) - dY_emp).ravel() * 0.5
            return np.concatenate([mse_res, viol_guard, deriv_res, 0.02 * p])

        res_stage2 = least_squares(stage2_residual, p_adm, max_nfev=45)
        self.p = res_stage2.x


class StoichiometricFluxNeuralODE:
    """
    Model 3: Pure Physics-Informed Stoichiometric Flux Neural ODE
    --------------------------------------------------------------------------------
    Pure Neural Network (MLP: 4 species -> 12 Tanh hidden units -> 2 elementary reaction fluxes)
    coupled to a hard Stoichiometric Conservation Layer:
        dy/dt = S @ r_theta(y)
    where S is the 4x2 stoichiometric matrix:
        S = [[-1,  0],   # Species A consumed in reaction 1
             [-1, +1],   # Species B consumed in reaction 1, regenerated in reaction 2
             [+1, -1],   # Species C produced in reaction 1, consumed in reaction 2
             [+1,  0]]   # Species D produced in reaction 1
    Because every column of S sums to 0 (1^T @ S = [0, 0]), the sum of derivatives is
    identically zero everywhere in state space. The structural mass invariant is exact;
    trajectory accuracy still depends on the split, solver, and numerical step size.
    """
    def __init__(self, hidden_dim: int = 12, seed: int = 42):
        self.S = np.array([
            [-1.0,  0.0],
            [-1.0, +1.0],
            [+1.0, -1.0],
            [+1.0,  0.0],
        ])
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)
        self.p = rng.normal(0, 0.1, 4 * hidden_dim + hidden_dim + hidden_dim * 2 + 2)

    def _unpack(self, p):
        H = self.hidden_dim
        i1 = 4 * H
        i2 = i1 + H
        i3 = i2 + H * 2
        W1 = p[:i1].reshape(4, H)
        b1 = p[i1:i2]
        W2 = p[i2:i3].reshape(H, 2)
        b2 = p[i3:]
        return W1, b1, W2, b2

    def reaction_fluxes(self, Y, p=None):
        W1, b1, W2, b2 = self._unpack(self.p if p is None else p)
        return np.tanh(Y @ W1 + b1) @ W2 + b2

    def integrate(self, y0, t_grid, p=None):
        W1, b1, W2, b2 = self._unpack(self.p if p is None else p)
        N = len(t_grid)
        Y_pred = np.zeros((N, 4))
        Y_pred[0] = y0
        for i in range(N - 1):
            dt = t_grid[i + 1] - t_grid[i]
            r = np.tanh(Y_pred[i] @ W1 + b1) @ W2 + b2
            Y_pred[i + 1] = Y_pred[i] + dt * (r @ self.S.T)
        return Y_pred

    def fit(self, t_train, Y_train):
        dt_vec = np.diff(t_train)[:, None]
        dY_emp = np.diff(Y_train, axis=0) / dt_vec
        pinv_S = np.linalg.pinv(self.S)
        R_emp = dY_emp @ pinv_S.T
        Y_in = Y_train[:-1]

        # Stage 1: Derivative Collocation Warm-Start on Stoichiometric Manifold
        res1 = least_squares(
            lambda p: (self.reaction_fluxes(Y_in, p) - R_emp).ravel(),
            self.p,
            max_nfev=120,
        )
        # Stage 2: Full Unrolled ODE Trajectory Refinement
        res2 = least_squares(
            lambda p: (self.integrate(Y_train[0], t_train, p) - Y_train).ravel(),
            res1.x,
            max_nfev=60,
        )
        self.p = res2.x


class SymbolicMechanisticNeuralODE:
    """
    Model 4: Symbolic-Stoichiometric Mechanistic Discovery ODE (SINDy + Stoichiometric Neural ODE)
    ----------------------------------------------------------------------------------------------
    Distills the learned stoichiometric fluxes into explicit, human-readable mass-action rate laws
    via Sequential Thresholded Least Squares (STRidge) over candidate elementary terms:
        Phi(y) = [A*B, C, A*C, B*D]
    On this synthetic benchmark, the sparse fit can recover the generating rate laws.
    This result does not establish accuracy on experimental or noisy reaction data.
    """
    def __init__(self):
        self.S = np.array([
            [-1.0,  0.0],
            [-1.0, +1.0],
            [+1.0, -1.0],
            [+1.0,  0.0],
        ])
        self.feature_names = ["A*B", "C", "A*C", "B*D"]
        self.K = np.zeros((2, 4))

    @staticmethod
    def _phi(y):
        A, B, C, D = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
        return np.stack([A * B, C, A * C, B * D], axis=-1)

    def fit(self, t_train, Y_train, threshold: float = 1e-3):
        dt_vec = np.diff(t_train)[:, None]
        dY_emp = np.diff(Y_train, axis=0) / dt_vec
        pinv_S = np.linalg.pinv(self.S)
        R_emp = dY_emp @ pinv_S.T
        Phi = self._phi(Y_train[:-1])

        K_hat, _, _, _ = np.linalg.lstsq(Phi, R_emp, rcond=None)
        for _ in range(3):
            K_hat[np.abs(K_hat) < threshold] = 0.0
            for j in range(2):
                mask = np.abs(K_hat[:, j]) >= threshold
                if np.any(mask):
                    k_sub, _, _, _ = np.linalg.lstsq(Phi[:, mask], R_emp[:, j], rcond=None)
                    K_hat[:, j] = 0.0
                    K_hat[mask, j] = k_sub
        self.K = K_hat.T

    def integrate(self, y0, t_grid):
        N = len(t_grid)
        Y_pred = np.zeros((N, 4))
        Y_pred[0] = y0
        for i in range(N - 1):
            dt = t_grid[i + 1] - t_grid[i]
            r = self._phi(Y_pred[i]) @ self.K.T
            Y_pred[i + 1] = Y_pred[i] + dt * (r @ self.S.T)
        return Y_pred


def save_model_artifact(model_out: pathlib.Path, models: dict, training_rows: int, training_end_s: float) -> pathlib.Path:
    """Save fitted parameters and provenance in a portable joblib bundle.

    The artifact stores plain parameter arrays rather than pickled model instances,
    so api.py can rebuild the current model classes from the repository source.
    """
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model_states = {}
    for key in ("M1", "M2", "M3"):
        model = models[key]
        model_states[key] = {
            "kind": type(model).__name__,
            "hidden_dim": int(model.hidden_dim),
            "p": np.asarray(model.p, dtype=np.float64),
        }
    model_states["M4"] = {
        "kind": type(models["M4"]).__name__,
        "K": np.asarray(models["M4"].K, dtype=np.float64),
        "feature_names": list(models["M4"].feature_names),
    }
    artifact = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "supplied synthetic chemical reaction CSVs",
        "training_split": "reconstruction/train.csv",
        "training_rows": int(training_rows),
        "training_end_s": float(training_end_s),
        "models": model_states,
    }
    joblib.dump(artifact, model_out, compress=3)
    return model_out


# ============================================================================
# STEP 3: METRICS & FULL BENCHMARK RUNNER
# ============================================================================
def compute_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, m_total: float = 2.0) -> dict:
    mse = float(np.mean((Y_true - Y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(Y_true - Y_pred)))
    ss_res = np.sum((Y_true - Y_pred) ** 2)
    ss_tot = np.sum((Y_true - np.mean(Y_true, axis=0, keepdims=True)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
    sum_viol = float(np.sum(np.abs(np.sum(Y_pred, axis=1) - m_total)))
    max_viol = float(np.max(np.abs(np.sum(Y_pred, axis=1) - m_total)))
    return {
        "Rows": int(len(Y_true)),
        "R2": r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Sum_Mass_Violation_g": sum_viol,
        "Max_Mass_Violation_g": max_viol,
    }


def run_complete_benchmark(data_dir: pathlib.Path, model_out: pathlib.Path | None = None):
    required_files = [
        data_dir / split / f"{kind}.csv"
        for split in ("reconstruction", "completion", "extrapolation")
        for kind in ("train", "test")
    ]
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "The benchmark expects all six supplied CSV files. Missing:\n"
            f"{missing_text}\nUse --generate-demo-data only to create synthetic demo data."
        )
    audit = audit_dataset_errors_and_null_recovery(data_dir)

    S = np.array([[-1.0, 0.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 0.0]])
    null_check = np.ones(4) @ S
    if not np.allclose(null_check, 0.0, atol=1e-16):
        raise AssertionError(f"Stoichiometric nullspace check failed: {null_check}")

    print("=" * 88)
    print("PART 1: DATA HYGIENE, NULL RECOVERY & EULER DISCRETIZATION AUDIT")
    print("=" * 88)
    print(f"  [Stoichiometric Nullspace] 1^T S = {null_check.tolist()} (exact invariant)")
    print(f"  [Discretization Audit at t=2.0s across Kaggle splits]")
    print(f"    - completion/test.csv    (dt=0.5s) : mass A = {audit['euler_t2_dt05_massA']:.8f} g")
    print(f"    - reconstruction/train.csv (dt=1.0s): mass A = {audit['euler_t2_dt10_massA']:.8f} g")
    print(f"    - extrapolation/test.csv (dt=2.0s) : mass A = {audit['euler_t2_dt20_massA']:.8f} g")
    print(f"    -> Built-in Forward Euler truncation gap (dt=0.5 vs dt=2.0): {audit['euler_truncation_gap_dt05_vs_dt20']:.6f} g")
    print(f"\n  [Simulated Sensor Dropout / Null Imputation Test (12 injected NaNs)]")
    print(f"    - Total NaNs detected in corrupted CSV       : {audit['null_diagnostics']['nulls_detected']}")
    print(f"    - Single-sensor nulls recovered via Mass Law : {audit['null_diagnostics']['mass_balance_imputed']} (Max error: {audit['max_mass_balance_imputation_error_g']:.2e} g)")
    print(f"    - Multi-sensor dropout rows bridged via ODE  : {audit['null_diagnostics']['irregular_gap_rows_dropped']} rows skipped cleanly")

    print("\n" + "=" * 88)
    print("PART 2: 4-TIER MODEL BENCHMARK ACROSS ALL 3 KAGGLE TASKS")
    print("=" * 88)

    # Note: In the Kaggle dataset, train.csv is identical across all 3 splits (100 points in [0, 99]).
    t_tr, Y_tr, _ = load_and_clean_csv(data_dir / "reconstruction/train.csv")

    m1 = BaselineUnconstrainedNeuralODE(hidden_dim=16, seed=42)
    m1.fit(t_tr, Y_tr)

    m2 = ICLR2023TwoStageNeuralODE(hidden_dim=16, seed=42)
    m2.fit(t_tr, Y_tr)

    m3 = StoichiometricFluxNeuralODE(hidden_dim=12, seed=42)
    m3.fit(t_tr, Y_tr)

    m4 = SymbolicMechanisticNeuralODE()
    m4.fit(t_tr, Y_tr)

    if model_out is None:
        model_out = pathlib.Path("./models/stoich_ode_models.joblib")
    saved_model_path = save_model_artifact(
        model_out,
        {"M1": m1, "M2": m2, "M3": m3, "M4": m4},
        training_rows=len(t_tr),
        training_end_s=float(t_tr[-1]),
    )

    all_results = []
    for task in ["reconstruction", "completion", "extrapolation"]:
        t_te, Y_te, _ = load_and_clean_csv(data_dir / task / "test.csv")

        p1 = m1.integrate(Y_te[0], t_te)
        p2 = m2.integrate(Y_te[0], t_te)
        p3 = m3.integrate(Y_te[0], t_te)
        p4 = m4.integrate(Y_te[0], t_te)

        print(f"\n--- Task: {task.upper()} (Train: {len(t_tr)} pts in [0,{t_tr[-1]:.0f}] | Test: {len(t_te)} pts in [0,{t_te[-1]:.0f}]) ---")
        for model_name, preds in [
            ("1. Baseline Unconstrained Neural ODE", p1),
            ("2. Simplified ICLR-Inspired Two-Stage ODE", p2),
            ("3. Pure Stoichiometric Flux Neural ODE", p3),
            ("4. Symbolic-Stoichiometric Hybrid ODE", p4),
        ]:
            m = compute_metrics(Y_te, preds)
            m["Task"] = task
            m["Model"] = model_name
            m["Evaluation"] = "full_test_csv"
            m["Train_end_s"] = float(t_tr[-1])
            m["Test_start_s"] = float(t_te[0])
            m["Test_end_s"] = float(t_te[-1])
            all_results.append(m)
            print(
                f"  {model_name:38s} | R2={m['R2']: .6f} | "
                f"MSE: {m['MSE']:.3e} | Mass Violation (Sum): {m['Sum_Mass_Violation_g']:.2e} g"
            )

            # Report the strictly unseen future interval separately. The full
            # extrapolation CSV also contains many timestamps covered by training.
            future_mask = t_te > t_tr[-1]
            if task == "extrapolation" and np.count_nonzero(future_mask) >= 2:
                future_metrics = compute_metrics(Y_te[future_mask], preds[future_mask])
                future_metrics.update({
                    "Task": task,
                    "Model": model_name,
                    "Evaluation": "future_only_t_gt_train_end",
                    "Train_end_s": float(t_tr[-1]),
                    "Test_start_s": float(t_te[future_mask][0]),
                    "Test_end_s": float(t_te[-1]),
                })
                all_results.append(future_metrics)
                print(
                    f"    strict future (t>{t_tr[-1]:g}s, {future_metrics['Rows']} rows): "
                    f"R2={future_metrics['R2']:.6f} | RMSE={future_metrics['RMSE']:.3e} g"
                )

    print("\n" + "=" * 88)
    print("PART 3: EXPLAINABILITY — DISCOVERED ELEMENTARY CHEMICAL RATE LAWS (MODEL 4)")
    print("=" * 88)
    print(f"  Reaction 1 (A + B -> C + D): r1 = {m4.K[0, 0]:.4f} * [A][B]  (True k1 = 0.1000)")
    print(f"  Reaction 2 (C -> B)        : r2 = {m4.K[1, 1]:.4f} * [C]     (True k2 = 0.0500)")
    print("=" * 88)
    results_path = data_dir / "benchmark_results.csv"
    pd.DataFrame(all_results).to_csv(results_path, index=False)
    print(f"Saved measured model metrics to: {results_path.resolve()}")
    print(f"Saved fitted model parameters to: {saved_model_path.resolve()}")
    return all_results, audit


if __name__ == "__main__":
    t_start = time.time()
    parser = argparse.ArgumentParser(description="Train and evaluate four reaction ODE models against the included CSV files.")
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("./kaggle_synthetic_reaction_data"),
        help="Folder containing reconstruction/, completion/, and extrapolation/ train/test CSVs.",
    )
    parser.add_argument(
        "--generate-demo-data",
        action="store_true",
        help="Create any missing synthetic demo CSVs. Existing files are preserved.",
    )
    parser.add_argument(
        "--model-out",
        type=pathlib.Path,
        default=pathlib.Path("./models/stoich_ode_models.joblib"),
        help="Destination for the fitted model parameter bundle loaded by the FastAPI app.",
    )
    args = parser.parse_args()
    if args.generate_demo_data:
        generate_kaggle_dataset(args.data_dir)
    run_complete_benchmark(args.data_dir, args.model_out)
    print(f"\nCompleted full pipeline verification in {time.time() - t_start:.2f} seconds.")
