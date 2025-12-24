# Stochastic asset price modelling, calibration and risk analysis

This project implements and compares four stochastic asset price models (GBM, OU, Heston and Merton jump-diffusion), which includes calibration to real market data, risk analysis via Monte Carlo simulation, tail probabilities, Value-at-Risk (VaR), and expected shortfall (ES). All results are reproducible via the provided notebooks and CLI interface.

## Motivation
Classical models such as Geometric Brownian Motion (GBM) assume constant volatility and normally distributed returns. Real market returns, however, exhibit:
- heavy tails
- negative skew
- volatility clustering
- rare but extreme jump events

The aim of this project is to study how different stochastic models perform when calibrated to real market data, and how these differences translate into tail risk and downside exposure.

## Implemented models
We consider the following price processes:
- Geometric Brownian Motion (GBM) - a constant-volatility diffusion model
- Ornstein-Uhlenbeck process (OU) - mean-reverting process applied to log-prices
- Heston stochastic volatility model - includes stochastic dynamics for both price and variance
- Merton's jump-diffusion model - a diffusion model with Poisson-driven price jumps

Each model is simulated using Euler or log-Euler discretization and supports large-scale Monte Carlo simulations. 

## Parameter calibration
All models are calibrated to historical market data using moment-based and regression-based estimators:
- GBM: drift and volatility from log-returns
- OU: linear regression on discretized OU dynamics
- Heston: variance dynamics estimated from realized volatility
- Merton: jump intensity and jump size via moment matching

Maximum likelihood estimation (MLE) is a natural extension and will be added in a future update. Full mathematical derivation of the formulas used for calibration can be found in a separate document (see docs/stochastic_models_estimation.pdf).

## Risk analysis
After calibration, each model is evaluated via Monte Carlo simulation:
- distribution of 1-day log-returns
- tail probability comparison
- QQ-plots against real market returns
- value at risk (VaR) at 95% and 99%
- expected shortfall (ES) at 95% and 99%

## Key findings
- GBM significantly underestimates extreme downside risk due to thin Gaussian tails.
- Heston improves tail behaviour via stochastic volatility but still underestimates extreme downside risk in calm regimes.
- Merton's jump-diffusion model captures left-tail risk most accurately and produces VaR and ES closest to real data. 

## Results

### Sample simulated price paths
![OU price paths](plots/ou_paths.png)

### Real data comparison
![Real vs GBM](plots/real_vs_gbm_paths.png)
![Real vs Merton](plots/real_vs_merton_paths.png)

### Return distribution comparison
![return distributions](plots/log-returnPDFs.png)

### QQ-plots
![GBM QQ](plots/qq_gbm.png)
![Heston QQ](plots/qq_heston.png)
![Merton QQ](plots/qq_merton.png)


## How to Run

### 1. Create virtual environment
```bash
python -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt
```

### 2. Run simulation
```bash
python main.py --model GBM
```
### 3. Run calibrated simulation
```bash
python main.py --model GBM --calibrated --data_path data/SPY.csv --price_col Price
```
### 4. Run simulation with input parameters
```bash
python main.py --model GBM --mu 0.1 --sigma 0.3 --S0 100 --T 1.0 --dt 1/252 --data_path data/SPY.csv --price_col Price --seed 42
```

## Project Structure
```text
stochastic-pricing/
│
├── data/           # historical data
├── notebooks/      # exploratory analysis
├── plots/          # generated figures
├── src/            # simulation, calibration, analysis
├── tests/          # test functions
└── main.py         # CLI entry point
```

## Future work
- Maximum likelihood estimation for all models
- Regime-switching volatility models
- Multivariate models and correlation modelling
- Portfolio-level VaR and ES

## Summary and conclusions
This project provides a pipeline for stochastic asset price modelling, calibration, simulation and risk analysis for four simple stochastic price models. The results demonstrate the limitations of constant-volatility models and the importance of jump and stochastic volatility dynamics for more realistic tail risk modelling. 

