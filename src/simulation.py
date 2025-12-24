"""
Monte Carlo simulation of stochastic price processes.
"""

import numpy as np

def simulate_gbm(params, seed:int=None):
    mu, sigma, S0, T, dt, n_paths = (params.mu, params.sigma, params.S0, params.T, params.dt, params.n_paths)
    rng = np.random.default_rng(seed)

    n_steps = int(T / dt) 
    times = np.linspace(0, T, n_steps+1)
    S = np.zeros((n_steps+1, n_paths))
    S[0] = S0

    for t in range(1, n_steps+1):
        z = rng.standard_normal(n_paths)
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    return times, S

def simulate_ou(params, seed:int=None):
    theta, mu, sigma, X0, T, dt, n_paths = (params.theta, params.mu, params.sigma, params.X0, params.T, params.dt, params.n_paths)
    rng = np.random.default_rng(seed)

    n_steps = int(T / dt)
    times = np.linspace(0, T, n_steps+1)
    X = np.zeros((n_steps+1, n_paths))
    X[0] = X0

    for t in range(1, n_steps+1):
        z = rng.standard_normal(n_paths)
        X[t] = X[t-1] + (mu - X[t-1]) * theta * dt + sigma * np.sqrt(dt) * z

    return times, X

def simulate_heston(params, seed:int=None):
    mu, kappa, theta, xi, rho, S0, v0, T, dt, n_paths = (params.mu, params.kappa, params.theta, params.xi, params.rho, params.S0, params.v0, params.T, params.dt, params.n_paths)

    rng = np.random.default_rng(seed)

    n_steps = int(T / dt) 
    times = np.linspace(0, T, n_steps+1)
    S = np.zeros((n_steps+1, n_paths))
    v = np.zeros_like(S)

    S[0] = S0
    v[0] = v0

    for t in range(1, n_steps+1):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        z_corr = rho * z1 + np.sqrt(1 - rho**2) * z2
        
        v_prev = np.maximum(v[t-1], 0)
        v[t] = v_prev + kappa * (theta - v_prev) * dt \
                + xi * np.sqrt(v_prev) * np.sqrt(dt) * z_corr
        v[t] = np.maximum(v[t], 0)

        S[t] = S[t-1] + mu * S[t-1] * dt \
                + np.sqrt(v_prev) * S[t-1] * np.sqrt(dt) * z1

    return times, S, v

def simulate_merton(params, seed:int=None):
    mu, sigma, S0, lambda_j, mu_j, sigma_j, T, dt, n_paths = (params.mu, params.sigma, params.S0, params.lambda_j, params.mu_j, params.sigma_j, params.T, params.dt, params.n_paths)
    rng = np.random.default_rng(seed)

    n_steps = int(T / dt) 
    times = np.linspace(0, T, n_steps)
    S = np.zeros((n_steps, n_paths))
    S[0] = S0

    for t in range(1, n_steps):
        z   = rng.standard_normal(n_paths)
        Nj  = rng.poisson(lambda_j * dt, size=n_paths)
        
        J = np.exp(mu_j + sigma_j * rng.standard_normal(n_paths))-1

        S[t] = S[t-1] + mu * S[t-1] * dt + sigma * S[t-1] * np.sqrt(dt) * z + S[t-1] * Nj * J # jump to linear order

    return times, S
