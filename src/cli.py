"""
Argparser for model arguments, simulation setup and calibration specifics.
"""
import argparse

def build_parser():
    parser = argparse.ArgumentParser()

    # model choice
    parser.add_argument("--model", choices=["GBM", "OU", "Heston", "Merton"], default="GBM", help="model to simulate: GBM, OU, Heston or Merton")
    
    # generic parameters
    parser.add_argument("--mu",    type=float, default=0.05,  )
    parser.add_argument("--sigma", type=float, default=0.2,   )
    parser.add_argument("--S0",    type=float, default=100,   )
    parser.add_argument("--theta", type=float)
    
    # OU-specific
    parser.add_argument("--X0", type=float)

    # Heston-specific
    parser.add_argument("--kappa", type=float)
    parser.add_argument("--xi",    type=float)
    parser.add_argument("--rho",   type=float)
    parser.add_argument("--v0",    type=float)

    # Merton-specific
    parser.add_argument("--lambda_j", type=float)
    parser.add_argument("--mu_j",     type=float)
    parser.add_argument("--sigma_j",  type=float)

    # simulation setup
    parser.add_argument("--T",       type=float, default=1.0,   )
    parser.add_argument("--dt",      type=float, default=1/252, )
    parser.add_argument("--n_paths", type=int,   default=10,    )
    parser.add_argument("--seed",    type=int,   default=None,  help="random seed (None = non-reproducible runs)")

    # calibration & data file specs
    parser.add_argument("--calibrated", action="store_true", help="use calibrated parameters instead of manually set params.")
    parser.add_argument("--data_path",  type=str, default="data/SPY.csv")
    parser.add_argument("--price_col",  type=str, default="Price", help="name of price column in data file")

    return parser
