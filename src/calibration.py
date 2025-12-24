"""
Parameter calibration routines for stochastic price models.
"""
import numpy as np
from src.utils import realized_variance

def calibrate_gbm(prices:np.ndarray, dt:float):
    prices      = np.asarray(prices)
    log_returns = np.diff(np.log(prices))

    sigma_hat   = np.std(log_returns, ddof=1) / np.sqrt(dt)
    mu_hat      = np.mean(log_returns) / dt + 0.5 * sigma_hat**2

    return mu_hat, sigma_hat

def calibrate_ou(X:np.ndarray, dt:float):
    X = np.asarray(X)
    X_t, X_next = X[:-1], X[1:]

    # estimate alpha and beta using OLS
    # bias=True/ddof=0 means division by 1/N
    # bias=False/ddof=1 means division by 1/(N-1) 
    beta  = np.cov(X_t, X_next, ddof=0)[0,1] / np.var(X_t)
    alpha = np.mean(X_next) - beta * np.mean(X_t)

    # compute residuals (errors)
    res = X_next - (alpha + beta * X_t)

    # convert discrete OLS parameters into continuous model parameters theta, mu and sigma
    theta_hat = (1 - beta) / dt
    mu_hat    = alpha / (theta_hat * dt)
    sigma_hat = np.std(res, ddof=1) / np.sqrt(dt)

    return theta_hat, mu_hat, sigma_hat

def fit_variance_to_ou(v_t:np.ndarray, dt:float):
    v_t = np.asarray(v_t)
    v_t_prev, v_next = v_t[:-1], v_t[1:]

    # linear regression: v_next = alpha + beta * v_t_prev + residual
    beta  = np.cov(v_t_prev, v_next, bias=True)[0,1] / np.var(v_t_prev)
    alpha = np.mean(v_next) - beta * np.mean(v_t_prev)

    # residual
    res = v_next - (alpha + beta * v_t_prev) # shocks to the variance process, used to compute correlation
    xi_hat = np.std(res, ddof=1) / np.sqrt(dt)

    # continous-time parameters
    kappa_hat = (1 - beta) / dt
    theta_hat = alpha / (kappa_hat * dt)

    return kappa_hat, theta_hat, xi_hat, res

def estimate_rho(log_returns:np.ndarray, res:np.ndarray):
    r = log_returns[1:] # make same length as v_next which is one step ahead
    rho_hat = np.corrcoef(r, res)[0,1]
    return rho_hat

def calibrate_heston(S:np.ndarray, dt:float):
    v_t, log_returns = realized_variance(S, dt)
    kappa, theta, xi, res = fit_variance_to_ou(v_t, dt)
    rho = estimate_rho(log_returns, res)
    v0 = v_t[-1]
    return kappa, theta, xi, rho, v0

def calibrate_merton(prices:np.array, dt:float):
    r = np.diff(np.log(prices))

    mean_r  = np.mean(r)
    var_r   = np.var(r, ddof=1)
    std_r   = np.std(r, ddof=1)
    skew_r  = np.mean((r - mean_r)**3) / std_r**3
    kurt_r  = np.mean((r - mean_r)**4) / std_r**4

    # drift (ignore jump contributions in the mean)
    mu = mean_r / dt

    # jump intensity inferred from excess kurtosis
    # excess kurtosis reflects fat tails induced by jumps
    excess_kurt = max(kurt_r - 3.0, 0.0) 
    lambda_j = 0.1 * excess_kurt         # heuristic scaling
    lambda_j = min(lambda_j, 1.0 / dt)   # cap at ~1 jump per time step

    # use only sign of skewness to determine jump asymmetry
    if abs(skew_r) > 0.1:
        mu_j = np.sign(skew_r) * 0.02    # ±2% jumps in log-price
    else:
        mu_j = 0.0                       # approximately symmetric jumps

    # jump volatility fixed by heutristic choice
    sigma_j = 0.05                       # 5% jump dispersion in log-space

    # subtract jump-induced variance from total variance
    jump_var_per_dt = lambda_j * (mu_j**2 + sigma_j**2)
    sigma_sq = var_r / dt - jump_var_per_dt
    sigma_sq = max(sigma_sq, 1e-8)       # clip to avoid negative from noise
    sigma = np.sqrt(sigma_sq)

    return mu, sigma, lambda_j, mu_j, sigma_j



