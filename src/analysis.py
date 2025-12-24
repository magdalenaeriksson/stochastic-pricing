"""
Risk analysis utilities: tail probabilities, VaR, and expected shortfall.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from src.models import GBMParams, OUParams, HestonParams, MertonParams
from src.calibration import calibrate_gbm, calibrate_ou, calibrate_heston, calibrate_merton
from src.simulation import simulate_gbm, simulate_ou, simulate_heston, simulate_merton

from pathlib import Path

def get_sim_logreturns(model:str, prices:np.ndarray, T:float=1.0, dt:float=1/252, n_paths:int=2000, seed:int=None):
    """ simulate many paths for given model and return 1-day log-returns as a 1D array """
    prices = np.asarray(prices, dtype=float)
    S0 = prices[-1] # start simulation from most recent observed price -> simulate future price paths

    if model == "GBM":
        mu_hat, sigma_hat = calibrate_gbm(prices, dt)
        params = GBMParams(mu=mu_hat, sigma=sigma_hat, S0=S0, T=T, dt=dt, n_paths=n_paths)
        _, S_sim = simulate_gbm(params, seed)
    elif model == "OU":
        logp = np.log(prices)
        theta_hat, mu_hat, sigma_hat = calibrate_ou(logp, dt=dt)
        params = OUParams(theta=theta_hat, mu=mu_hat, sigma=sigma_hat, 
                          X0=logp[-1], T=T, dt=dt, n_paths=n_paths)
        _, X_sim = simulate_ou(params, seed)
        S_sim = np.exp(X_sim)
    elif model == "Heston":
        kappa, theta, xi, rho, v0 = calibrate_heston(prices, dt)
        mu_hat, _ = calibrate_gbm(prices, dt)
        params = HestonParams(mu=mu_hat, kappa=kappa, theta=theta, xi=xi, 
                              rho=rho, S0=S0, v0=v0, T=T, dt=dt, n_paths=n_paths)
        _, S_sim, v = simulate_heston(params, seed)
    elif model == "Merton":
        mu_hat, sigma_hat, lambda_j, mu_j, sigma_j = calibrate_merton(prices, dt)
        params = MertonParams(mu=mu_hat, sigma=sigma_hat, S0=S0, lambda_j=lambda_j,
                              mu_j=mu_j, sigma_j=sigma_j, T=T, dt=dt, n_paths=n_paths)
        _, S_sim = simulate_merton(params, seed)
    
    log_ret_sim = np.diff(np.log(S_sim), axis=0)

    return log_ret_sim.ravel() # ravel() : flatten to 1D to build histogram of daily returns

def summarize_returns(r:np.ndarray):
    """" compute basic distribution statistics for daily returns """
    r = np.asarray(r)

    return {
        "mean": np.mean(r),
        "std": np.std(r, ddof=1),
        "skew": skew(r),
        "kurtosis": kurtosis(r, fisher=False) # Fisher=False for Pearson kurtosis = 3
    }

def qq_plot(real_returns:np.ndarray, sim_returns:np.ndarray, model:str, save_path:str=None):
    """ make qq plot comparing real and simulated returns """

    real = np.sort(real_returns)
    sim  = np. sort(sim_returns)

    # ensure equal-length comparison by interpolation
    q = np.linspace(0, 1, len(real))
    sim_q = np.quantile(sim, q)

    plt.figure(figsize=(8,4))
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

    if save_path:
        plt.savefig(save_path, dpi=200)
        #print("saved to:", Path(save_path).resolve())

def get_var(r:np.ndarray, alpha:float) -> float:
    """ compute 1-day value at risk (VaR) at confidence level alpha from 
        a sample of returns r, returns VaR as a positive number (loss) """
    r = np.asarray(r)
    q = np.quantile(r,1 - alpha) # left-tail quantile: 1-alpha (e.g. 5% if alpha=95%)

    return -q

def get_es(r:np.ndarray, alpha:float) -> float:
    """ compute 1-day expected shortfall (ES/CVaR) at confidence level alpha from returns r """
    r = np.asarray(r)
    q = np.quantile(r, 1 - alpha)
    tail = r[r <= q]
    if tail.size == 0:
        return 0.0
    
    return -np.mean(tail)

def plot_real_vs_simulated(prices: np.ndarray, 
                           S: np.ndarray, 
                           model: str, 
                           window: int = None, 
                           n_show: int = 10, 
                           align: str = "start",
                           save_path: str = None):
    """ 
    plot real (observed) price paths vs simulated paths S 
    prices      : (T,) price series (or log-price series)
    S           : (n_steps + 1, n_paths) simulated paths
    window      : compare last window points (int)
    n_show      : number of simulated paths to show in the plot (default = 10)
    align       : "start" -> use S[:N] (start-aligned), "end" -> use S[-N:] (end-aligned)
    """

    prices = np.asarray(prices, dtype=float)
    S = np.asarray(S, dtype=float)

    T_real = prices.shape[0]
    T_sim  = S.shape[0]

    N = min(T_real, T_sim) if window is None else min(window, T_real, T_sim)

    real_w = prices[-N:]

    if align == "start":
        sim_w = S[:N, :]  # first N points in simulation (keeps S0 alignment)
    elif align == "end":
        sim_w = S[-N:, :] # last N points in simulation (overlap alignment)
    else:
        raise ValueError("align must be 'start' or 'end'")

    t = np.arange(N)

    plt.figure(figsize=(8, 4))
    plt.plot(t, real_w, label="real data", lw=2)

    n_plot = min(n_show, sim_w.shape[1])
    for i in range(n_plot):
        plt.plot(t, sim_w[:, i], alpha=0.35) # color="C1"
    
    plt.title(f"real vs simulated {model} paths (calibrated)")
    plt.xlabel("time steps")
    plt.ylabel("price")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=200)
        #print("saved to:", Path(save_path).resolve())


