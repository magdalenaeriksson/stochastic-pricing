# run source .venv/bin/activate to activate virtual environment in terminal

from src.simulation import simulate_gbm
import matplotlib.pyplot as plt

def main():
    times, S = simulate_gbm(mu=0.07, sigma=0.25, S0=100, T=1.0, dt=1/252, n_paths=10)
    
    plt.figure(figsize=(8,5))
    plt.plot(times, S)
    plt.title("Simulated GBM Price Paths")
    plt.xlabel("Time (years)")
    plt.ylabel("Price")
    plt.grid(True)
    # plt.show() # if interactive
    plt.savefig("plots/gbm_plot.png") 

if __name__ == "__main__":
    main()
