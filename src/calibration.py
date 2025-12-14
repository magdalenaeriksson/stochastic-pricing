import numpy as np # for numerical operations (vectors, arrays, np.diff, np.log, np.var, etc)
import pandas as pd # use it here to compure rolling variance
from dataclasses import dataclass # in case we use @dataclass containers
from typing import Dict, Tuple # make functions clearer, eg "-> Dict[str, float]" means that
# this function returns a directory that maps strings to floats
from scipy import optimize, stats # for optimization routines, eg. optimize.minimize() and 
# for statistical distributions and tests
from src.utils import realized_variance

def calibrate_gbm(prices: np.ndarray, dt: float=1/252):
    # calibrate GBM parameters from historical (daily) data
    prices      = np.asarray(prices) # ensure input is a numpy array
    log_returns = np.diff(np.log(prices))

    sigma_hat   = np.std(log_returns, ddof=1) / np.sqrt(dt)
    mu_hat      = np.mean(log_returns) / dt + 0.5 * sigma_hat**2

    return mu_hat, sigma_hat

def calibrate_ou(X: np.ndarray, dt: float = 1/252):
    # calibrate OU parameters from observed data X
    X      = np.asarray(X)
    X_t    = X[:-1]
    X_next = X[1:]
    # X_t, X_next = X[:-1], X[1:]

    # estimate alpha and beta using OLS
    # bias=True alt ddof=0 means division by 1/N
    # bias=False alt ddof=1 means division by 1/(N-1) 
    beta  = np.cov(X_t, X_next, ddof=0)[0,1] / np.var(X_t)
    alpha = np.mean(X_next) - beta * np.mean(X_t)

    # compute residuals (errors)
    res = X_next - (alpha + beta * X_t)

    # convert discrete OLS parameters into continuous model 
    # parameters theta, mu and sigma
    theta_hat = (1 - beta) / dt
    mu_hat    = alpha / (theta_hat * dt)
    sigma_hat = np.std(res, ddof=1) / np.sqrt(dt)

    return theta_hat, mu_hat, sigma_hat

def fit_variance_to_ou(v_t, dt=1/252):
    # fit variance process v_t to OU params kappa, theta, xi
    v_t = np.asarray(v_t)
    v_t_prev, v_next = v_t[:-1], v_t[1:]

    # linear regression: v_next = alpha + beta * v_t_prev + residual
    beta  = np.cov(v_t_prev, v_next, bias=True)[0,1] / np.var(v_t_prev)
    alpha = np.mean(v_next) - beta * np.mean(v_t_prev)

    # residual
    res = v_next - (alpha + beta * v_t_prev)
    xi_hat = np.std(res, ddof=1) / np.sqrt(dt)

    # continous time parameters
    kappa_hat = (1 - beta) / dt
    theta_hat = alpha / (kappa_hat * dt)

    # res = shocks to the variance process, used to compute correlation
    return kappa_hat, theta_hat, xi_hat, res

def estimate_rho(log_returns, res):
    # compute correlation between log-returns and variance shocks
    r = log_returns[1:] # make same length as v_next which is one step ahead
    rho_hat = np.corrcoef(r, res)[0,1]
    return rho_hat

def calibrate_heston(S, dt=1/252):
    # calibrate Heston parameters from historical prices
    v_t, log_returns = realized_variance(S, dt)
    kappa, theta, xi, res = fit_variance_to_ou(v_t, dt)
    rho = estimate_rho(log_returns, res)
    v0 = v_t[-1]
    return kappa, theta, xi, rho, v0

def calibrate_merton(prices: np.array, dt: float =1/252):
    # moment-based calibration for the Merton jump-diffusion model

    # Model (small Δt):
    #     r_t = μ Δt + σ √Δt Z_t + sum_{i=1}^{N_t} Y_i

    # where:
    #     Z_t ~ N(0,1)
    #     N_t ~ Poisson(λ Δt)
    #     Y_i ~ N(μ_J, σ_J^2)

    # Parameters we estimate:
    #     μ       : diffusion drift (per year)
    #     σ       : diffusion volatility (per year)
    #     λ       : jump intensity (expected jumps per year)
    #     μ_J     : mean jump size in log-returns
    #     σ_J     : jump volatility

    # Returns
    # -------
    # mu, sigma, lambda_j, mu_j, sigma_j

    # 1. Compute log-returns r_t = log(S_{t+1} / S_t)
    r = np.diff(np.log(prices))

    # Sample moments of returns:
    mean_r  = np.mean(r)                    # ≈ E[r_t]
    var_r   = np.var(r, ddof=1)              # ≈ Var(r_t)
    std_r   = np.std(r, ddof=1)
    skew_r  = np.mean((r - mean_r)**3) / std_r**3       # ≈ γ_1 (skewness)
    kurt_r  = np.mean((r - mean_r)**4) / std_r**4       # ≈ γ_2 (kurtosis)

    # --- Link to theory ---
    # From the Merton model, for small Δt:
    #   E[r_t]      = (μ + λ μ_J) Δt
    #   Var(r_t)    = σ^2 Δt + λ Δt (μ_J^2 + σ_J^2)
    #   Skew(r_t)   ∝ λ Δt (μ_J^3 + 3 μ_J σ_J^2)
    #   Kurt(r_t)-3 ∝ λ Δt (μ_J^4 + 6 μ_J^2 σ_J^2 + 3 σ_J^4)
    #
    # We use these qualitatively:
    #   - mean_r     -> μ (drift)
    #   - variance   -> σ^2 + jump variance
    #   - skewness   -> sign of μ_J (direction of jumps)
    #   - kurtosis   -> magnitude of λ (jump frequency)

    # 2. Drift estimate: ignore jumps in the mean for simplicity
    #    E[r_t] ≈ μ Δt  =>  μ ≈ mean_r / Δt
    mu = mean_r / dt

    # 3. Start with a "GBM-like" diffusion volatility from total variance:
    #    Var(r_t) ≈ σ^2 Δt  =>  σ_GBM ≈ sqrt(Var(r_t) / Δt)
    sigma_gbm = np.sqrt(var_r / dt)

    # 4. Estimate jump intensity λ from excess kurtosis:
    #    Excess kurtosis (kurt_r - 3) grows with λ:
    #       Kurt(r_t) - 3 ∝ λ Δt (...)
    #    So we set λ proportional to excess kurtosis.
    excess_kurt = max(kurt_r - 3.0, 0.0)
    lambda_j = excess_kurt * 0.1         # heuristic scaling factor
    # Avoid absurdly large λ when data is weird:
    lambda_j = min(lambda_j, 1.0 / dt)   # at most ~1 jump per time step

    # 5. Estimate average jump size μ_J from skewness:
    #    Skew(r_t) ∝ λ Δt (μ_J^3 + 3 μ_J σ_J^2)
    #    We use only the SIGN of skewness to decide direction.
    if abs(skew_r) > 0.1:
        mu_j = np.sign(skew_r) * 0.02    # e.g. ±2% jumps in log-price
    else:
        mu_j = 0.0                       # no strong skewness -> symmetric jumps

    # 6. Choose a small jump volatility σ_J:
    #    Typical daily jump magnitudes a few percent; here we fix σ_J
    #    instead of solving full nonlinear moment equations.
    sigma_j = 0.05                       # e.g. 5% jump dispersion in log-space

    # 7. Adjust diffusion volatility σ to account for jump variance:
    #    From Var(r_t) = σ^2 Δt + λ Δt (μ_J^2 + σ_J^2)
    #    => σ^2 = Var(r_t)/Δt - λ (μ_J^2 + σ_J^2)
    jump_var_per_dt = lambda_j * (mu_j**2 + sigma_j**2)
    sigma_sq = var_r / dt - jump_var_per_dt
    sigma_sq = max(sigma_sq, 1e-8)       # clip to avoid negative from noise
    sigma = np.sqrt(sigma_sq)

    return mu, sigma, lambda_j, mu_j, sigma_j



