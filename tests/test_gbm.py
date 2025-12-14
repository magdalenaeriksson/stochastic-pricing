import numpy as np
from src.simulation import simulate_gbm
from src.models import GBMParams

def test_gbm_shapes():
    params = GBMParams(mu=0.05, sigma=0.2, S0=100, T=1.0, dt=1/252, n_paths=10)
    t, S = simulate_gbm(params, seed=42)
    assert S.shape == (len(t), params.n_paths)
