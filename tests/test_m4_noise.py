from pathlib import Path

import numpy as np

from train_stoich_ode import SymbolicMechanisticNeuralODE, load_and_clean_csv


DATA_DIR = Path(__file__).resolve().parents[1] / "kaggle_synthetic_reaction_data"


def test_m4_one_percent_measurement_noise_worsens_future_prediction():
    train_t, clean_train, _ = load_and_clean_csv(DATA_DIR / "reconstruction/train.csv")
    test_t, clean_test, _ = load_and_clean_csv(DATA_DIR / "extrapolation/test.csv")

    clean_model = SymbolicMechanisticNeuralODE()
    clean_model.fit(train_t, clean_train)
    clean_prediction = clean_model.integrate(clean_test[0], test_t)

    rng = np.random.default_rng(20261021)
    noisy_train = clean_train + rng.normal(0.0, 0.02, clean_train.shape)
    noisy_model = SymbolicMechanisticNeuralODE()
    noisy_model.fit(train_t, noisy_train)
    noisy_prediction = noisy_model.integrate(clean_test[0], test_t)

    future = test_t > 99.0
    clean_rmse = float(np.sqrt(np.mean((clean_prediction[future] - clean_test[future]) ** 2)))
    noisy_rmse = float(np.sqrt(np.mean((noisy_prediction[future] - clean_test[future]) ** 2)))

    assert noisy_rmse > clean_rmse + 1e-3
    assert noisy_rmse < 0.1
    assert np.all(np.isfinite(noisy_prediction))
