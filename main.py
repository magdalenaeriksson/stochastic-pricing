"""
Main file to simulate stochastic price processes and compare with observed market data.
"""
from src.cli import build_parser
from src.models import GBMParams, OUParams, HestonParams, MertonParams
from src.simulation import simulate_gbm, simulate_ou, simulate_heston, simulate_merton
from src.utils import load_prices

import matplotlib.pyplot as plt
import numpy as np

def plot(times, data, name):
    """ plot simulated price paths """
    plt.figure(figsize=(8,4))
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
    "Heston": run_heston,
    "Merton": run_merton
}

def get_gbm_params(args):
    """ set GBM parameters according to either calibrated values or manual input """
    if args.calibrated:
        from src.calibration import calibrate_gbm

        prices = load_prices(args.data_path, args.price_col)

        mu_hat, sigma_hat = calibrate_gbm(prices, args.dt)
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
    """ set OU parameters according to either calibrated values or manual input """
    if args.calibrated:
        from src.calibration import calibrate_ou

        prices = load_prices(args.data_path, args.price_col)

        theta_hat, mu_hat, sigma_hat = calibrate_ou(np.log(prices), args.dt)
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
    """ set Heston parameters according to either calibrated values or manual input """
    if args.calibrated:
        from src.calibration import calibrate_gbm, calibrate_heston

        prices = load_prices(args.data_path, args.price_col)
        mu_hat, _ = calibrate_gbm(prices, args.dt)   # ignore GBM sigma here
        kappa_hat, theta_hat, xi_hat, rho_hat, v0 = calibrate_heston(prices, args.dt)
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
    """ set Merton parameters according to either calibrated values or manual input """
    if args.calibrated:
        from src.calibration import calibrate_merton

        prices = load_prices(args.data_path, args.price_col)
        
        mu_hat, sigma_hat, lambda_j_hat, mu_j_hat, sigma_j_hat = calibrate_merton(prices, args.dt)
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


def main():
    
    parser = build_parser()
    args = parser.parse_args()
    
    RUNNERS[args.model](args)


if __name__ == "__main__":
    main()

