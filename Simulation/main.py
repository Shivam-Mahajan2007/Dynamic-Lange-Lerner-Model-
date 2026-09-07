#!/usr/bin/env python3
import os
from Simulation.src.model.simulation import SimulationEngine
from Simulation.src.analysis.statistics import compute_summary_statistics
from Simulation.src.analysis.visualization import plot_price_trajectories, plot_loss_convergence

def main():
    # 1. Initialize and run simulation
    engine = SimulationEngine()
    
    # Run the dynamic lange-lerner model simulation
    history = engine.run_simulation(max_iter=100)
    
    # 2. Analyze the results
    print("\n--- Computing Summary Statistics ---")
    stats_df = compute_summary_statistics(history)
    print(stats_df)
    
    # 3. Visualize the results
    print("\n--- Generating Plots ---")
    os.makedirs('results', exist_ok=True)
    
    plot_price_trajectories(history, save_path='results/price_trajectories.png')
    plot_loss_convergence(history, save_path='results/loss_convergence.png')
    
    print("Plots saved in the 'results' directory.")
    
if __name__ == "__main__":
    main()
