# stochastic models (GBM, OU, etc. )
from dataclasses import dataclass

@dataclass
class GBMParams:
    mu      : float # drift of asset price (expected returns)
    sigma   : float # volatility
    S0      : float # initial price
    T       : float # total time horizon (in years)
    dt      : float # time step (in years, e.g. 1/252 for daily)
    n_paths : int   # number of Monte-Carlo paths

@dataclass
class OUParams:
    theta   : float # speed of mean-reversion
    mu      : float # long-term mean level
    sigma   : float # volatility
    X0      : float # initial rate/spreads
    T       : float
    dt      : float
    n_paths : int

@dataclass
class HestonParams:
    mu      : float # drift
    kappa   : float # mean-reversion speed of volatility
    theta   : float # long-run variance
    xi      : float # volatility of volatility
    rho     : float # correlation between price and volatility shocks
    S0      : float # initial price
    v0      : float # initial variance
    T       : float
    dt      : float
    n_paths : int

@dataclass
class MertonParams:
    mu       : float
    sigma    : float
    S0       : float
    lambda_j : float # jumps per year
    mu_j     : float # mean jump size (log)
    sigma_j  : float # std of jump size (log)
    T        : float
    dt       : float
    n_paths  : int
