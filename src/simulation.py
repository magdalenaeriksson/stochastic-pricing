# code to run simulation

import numpy as np

# simulate Geometric Brownian Motion price paths using Euler–Maruyama method
def simulate_gbm(mu=0.05, sigma=0.2, S0=100, T=1.0, dt=1/252, n_paths=5):

    n_steps = int(T / dt)
    times = np.linspace(0, T, n_steps)
    S = np.zeros((n_steps, n_paths))
    S[0] = S0

    for t in range(1, n_steps):
        z = np.random.standard_normal(n_paths)
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    return times, S

