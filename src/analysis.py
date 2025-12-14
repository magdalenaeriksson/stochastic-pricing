# code to compute metrics or plots
from src.models import GBMParams, OUParams, HestonParams, MertonParams
from src.utils import load_prices, realized_variance
from src.calibration import calibrate_gbm, calibrate_ou, calibrate_heston, calibrate_merton
from src.simulation import simulate_gbm, simulate_ou, simulate_heston, simulate_merton

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


def get_sim_logreturns(model:str, price_data:np.ndarray, T:float=1.0, dt:float=1/252, n_paths:int=2000, seed:int=42):
    # simulate many paths for model and return 1-day log-returns as a 1D array
    price_data = np.asarray(price_data, dtype=float)
    S0 = price_data[-1]
    model = model.upper()

    if model == "GBM":
        mu_hat, sigma_hat = calibrate_gbm(price_data, dt)
        params = GBMParams(mu=mu_hat, sigma=sigma_hat, S0=S0, T=T, dt=dt, n_paths=n_paths)
        _, S_sim = simulate_gbm(params, seed)
    elif model == "OU":
        logp = np.log(price_data)
        theta_hat, mu_hat, sigma_hat = calibrate_ou(logp, dt=dt)
        params = OUParams(theta=theta_hat, mu=mu_hat, sigma=sigma_hat, 
                          X0=logp[-1], T=T, dt=dt, n_paths=n_paths)
        _, X_sim = simulate_ou(params, seed)
        S_sim = np.exp(X_sim)
    elif model == "HESTON":
        kappa, theta, xi, rho, v0 = calibrate_heston(price_data, dt)
        mu_hat, _ = calibrate_gbm(price_data, dt)
        params = HestonParams(mu=mu_hat, kappa=kappa, theta=theta, xi=xi, 
                              rho=rho, S0=S0, v0=v0, T=T, dt=dt, n_paths=n_paths)
        _, S_sim, v = simulate_heston(params, seed)
    elif model == "MERTON":
        mu_hat, sigma_hat, lambda_j, mu_j, sigma_j = calibrate_merton(price_data, dt)
        params = MertonParams(mu=mu_hat, sigma=sigma_hat, S0=S0, lambda_j=lambda_j,
                              mu_j=mu_j, sigma_j=sigma_j, T=T, dt=dt, n_paths=n_paths)
        _, S_sim = simulate_merton(params, seed)
    else:
        raise ValueError(f"unknown model: {model}")
    
    log_ret_sim = np.diff(np.log(S_sim), axis=0)

    return log_ret_sim.ravel()

def summarize_returns(r:np.ndarray):
    # compute basic distribution statistics for daily returns
    r = np.asarray(r)

    return {
        "mean": np.mean(r),
        "std": np.std(r, ddof=1),
        "skew": skew(r),
        "kurtosis": kurtosis(r, fisher=False) # Fisher=False for Pearson kurtosis=3 
        # for normal dist, True -> 0 for normal
    }

def qq_plot(real_returns:np.ndarray, sim_returns:np.ndarray, model:str, save_path:str):
    # make qq plot comparing real and simulated returns

    real = np.sort(real_returns)
    sim  = np. sort(sim_returns)

    # ensure equal-length comparison by interpolation
    q = np.linspace(0, 1, len(real))
    sim_q = np.quantile(sim, q)

    plt.figure(figsize=(10,5))
    plt.scatter(real, sim_q, s=10, alpha=0.5, label=f"{model} vs real")

    # draw 45 degree line
    min_line = min(real.min(), sim_q.min())
    max_line = max(real.max(), sim_q.max())
    plt.plot([min_line, max_line], [min_line, max_line], "--r", label="perfect fit")

    plt.title(f"qq plot: real vs {model} returns")
    plt.xlabel("real data quantiles")
    plt.ylabel("model quantiles")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=200)

def get_var(r:np.ndarray, alpha:float) -> float:
    # compute 1-day value at risk (VaR) at confidence level alpha from a sample of returns r
    # returns VaR as a positive number (loss)
    r = np.asarray(r)
    # left-tail quantile: 1-alpha (e.g. 5% if alpha=95%)
    q = np.quantile(r,1 - alpha)
    return -q

def get_es(r:np.ndarray, alpha:float) -> float:
    # compute 1-day expected shortfall (ES/CVaR) at confidence level alpha from returns r
    r = np.asarray(r)
    q = np.quantile(r, 1 - alpha)
    tail = r[r <= q]
    if tail.size == 0:
        return 0.0
    
    return -np.mean(tail)

def plot_real_vs_simulated(price_data, S, model:str):
    import matplotlib.pyplot as plt
    import numpy as np

    S0 = S[0,0] # simulation starting price (can be == price_data[-1])
    
    price_data_rescaled = price_data / price_data[-1] * S0
    N = len(price_data_rescaled)
    S = S[:N, :] # originally, S.shape = (n_steps + 1, n_paths), so here we keep
                 # only the first N rows and all column to match with price_data

    t = np.arange(N)

    plt.figure(figsize=(12, 6))
    plt.plot(price_data_rescaled, label="price data", lw=2)

    for i in range(min(10, S.shape[1])):
        plt.plot(t, S[:, i], color="C1", alpha=0.4)
    
    plt.title(f"real vs simulated {model} paths (calibrated)")
    plt.xlabel("time steps")
    plt.ylabel("price")
    plt.grid(True)
    plt.legend()

    plt.savefig(f"data/data_vs_{model.lower()}.png", dpi=200)
    print(f"✅ {model} comparison plot saved to data/real_vs_{model.lower()}.png")


