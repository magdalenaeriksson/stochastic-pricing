# run source .venv/bin/activate to activate virtual environment in terminal from project root

from src.cli import build_parser
from src.models import GBMParams, OUParams, HestonParams, MertonParams
from src.simulation import simulate_gbm, simulate_ou, simulate_heston, simulate_merton
from src.utils import load_prices, realized_variance
from src.analysis import plot_real_vs_simulated

import matplotlib.pyplot as plt
import argparse
import numpy as np

def plot(times, data, name):
    plt.figure(figsize=(10,5))
    for i in range(data.shape[1]):
        plt.plot(times, data[:,i], lw=1.2)
    plt.title(f"{name} simulated paths")
    plt.xlabel("time [years]")
    plt.ylabel("value")
    plt.grid(True)
    #plt.tight_layout()
    plt.savefig(f"plots/{name.lower()}_paths.png", dpi=300)
    print(f"✅ {name} plot saved to plots/{name.lower()}_paths.png")

def run_gbm(args):
    params = get_gbm_params(args)
    times, S = simulate_gbm(params, seed=42)
    plot(times, S, "GBM")

def run_ou(args):
    params = get_ou_params(args)
    times, X = simulate_ou(params, seed=42)
    plot(times, X, "OU")

def run_heston(args):
    params = get_heston_params(args)
    times, S, v = simulate_heston(params, seed=42)
    plot(times, S, "Heston price")
    plot(times, v, "Heston variance")

def run_merton(args):
    params = get_merton_params(args)
    times, S = simulate_merton(params, seed=42)
    plot(times, S, "Merton")

RUNNERS = {
    "GBM"   : run_gbm,
    "OU"    : run_ou,
    "HESTON": run_heston,
    "MERTON": run_merton
}


# prices[-1] means "from the end", i.e.
# prices[0], prices[1] = first and second elements, while
# prices[-1], prices[-2] = last and second-last elements
# prices[-N] = last N elements
# and so S0=prices[-1] means that we start the simulation from 
# the most recent observed price. So we use historical prices to
# calibrate the parameters and then simulate forward, from today's
# price. The simulation then models plausable future price paths.

def get_gbm_params(args):
    if args.calibrated:
        from src.calibration import calibrate_gbm

        prices = load_prices(args.data_path, args.price_col)

        mu_hat, sigma_hat = calibrate_gbm(prices)
        return GBMParams(mu=mu_hat, 
                         sigma=sigma_hat, 
                         S0=prices[-1], 
                         T=args.T, 
                         dt=args.dt, 
                         n_paths=args.n_paths)
    else: 
        return GBMParams(args.mu, args.sigma, args.S0,
                         args.T, args.dt, args.n_paths)

def get_ou_params(args):
    if args.calibrated:
        from src.calibration import calibrate_ou

        prices = load_prices(args.data_path, args.price_col)

        theta_hat, mu_hat, sigma_hat = calibrate_ou(np.log(prices),args.dt)
        return OUParams(theta=theta_hat, 
                        mu=mu_hat, 
                        sigma=sigma_hat,
                        X0=np.log(prices[-1]), 
                        T=args.T, 
                        dt=args.dt,
                        n_paths=args.n_paths)
    else: 
        return OUParams(args.theta, args.mu, args.sigma,
                        args.X0, args.T, args.dt, args.n_paths)

def get_heston_params(args):
    if args.calibrated:
        from src.calibration import calibrate_gbm, calibrate_heston

        prices = load_prices(args.data_path, args.price_col)
        mu_hat, _ = calibrate_gbm(prices)   # ignore GBM sigma here
        kappa_hat, theta_hat, xi_hat, rho_hat, v0 = calibrate_heston(prices)
        return HestonParams(mu=mu_hat,
                            kappa=kappa_hat, 
                            theta=theta_hat,
                            xi=xi_hat, 
                            rho=rho_hat, 
                            S0=prices[-1], 
                            v0=v0, 
                            T=args.T,
                            dt=args.dt, 
                            n_paths=args.n_paths)
    else:
        return HestonParams(args.mu, args.kappa, args.theta,
                            args.xi, args.rho, args.S0,
                            args.v0, args.T, args.dt, args.n_paths)

def get_merton_params(args):
    if args.calibrated:
        from src.calibration import calibrate_merton

        prices = load_prices(args.data_path, args.price_col)
        
        mu_hat, sigma_hat, lambda_j_hat, mu_j_hat, sigma_j_hat = calibrate_merton(prices)
        return MertonParams(mu=mu_hat,
                            sigma=sigma_hat,
                            S0=prices[-1],
                            lambda_j=lambda_j_hat,
                            mu_j=mu_j_hat, 
                            sigma_j=sigma_j_hat,
                            T=args.T, 
                            dt=args.dt,
                            n_paths=args.n_paths)
    else:
        return MertonParams(args.mu, args.sigma, args.S0,
                            args.lambda_j, args.mu_j, args.sigma_j,
                            args.T, args.dt, args.n_paths)

def compare_real_vs_simulated(model, args, time_window:int): 
    price_data = load_prices(args.data_path, args.price_col)
    price_window = price_data[-time_window:] # time_window is in number of days 

    if model == "GBM":
        params = get_gbm_params(args)
        _ , S = simulate_gbm(params, seed=42)
    elif model == "OU":
        params = get_ou_params(args)
        _ , X = simulate_ou(params, seed=42)
        S = np.exp(X)
    elif model == "HESTON":
        params = get_heston_params(args)
        _ , S, v = simulate_heston(params, seed=42)
    elif model == "MERTON":
        params = get_merton_params(args)
        _ , S = simulate_merton(params, seed=42)
    
    plot_real_vs_simulated(price_window, S, model)


def main():
    
    parser = build_parser()
    args = parser.parse_args()

    model = args.model.upper()

    if model not in RUNNERS:
        #raise ValueError(f"unknown model '{args.model}' -- try GBM, OU, Heston, or Merton")
        print(f"unknown model '{args.model}' -- try GBM, OU, Heston, or Merton")
        exit()
    
    RUNNERS[model](args)
    compare_real_vs_simulated(model, args, 252)


if __name__ == "__main__":
    main()

