"""
Helper functions for data loading & preprocessing, and variance.
"""
import pandas as pd
import numpy as np

def realized_variance(S:np.ndarray, dt:float=1/252):
    """ compute log-returns and realized variance """
    S = np.asarray(S)
    log_returns = np.diff(np.log(S))
    v_raw = log_returns**2 / dt # log-returns per year (convention)
    v_t = pd.Series(v_raw).rolling(5, min_periods=1).mean().to_numpy() # 5-day avrg
    return v_t, log_returns


def load_prices(path:str, price_col:str) -> np.ndarray:
    """ load historical price data from CSV file """
    df = pd.read_csv(path, thousands=",")

    if price_col not in df.columns:
        raise ValueError(f"column '{price_col}' not found in {path}"
                         f"available columns: {df.columns.tolist()}")
    
    prices = df[price_col].to_numpy(dtype=float)

    if prices.ndim != 1:
        raise ValueError("prices must be a 1D vector")
    
    return prices