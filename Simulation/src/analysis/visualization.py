import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict

# Set plot style
sns.set_theme(style="whitegrid")

def plot_price_trajectories(history: Dict, save_path: str = None):
    """
    Plots the trajectory of prices across iterations.
    """
    if 'x' not in history:
        print("No price trajectory found in history.")
        return
        
    prices = np.array(history['x'])
    iterations = np.arange(len(prices))
    
    plt.figure(figsize=(10, 6))
    for i in range(prices.shape[1]):
        plt.plot(iterations, prices[:, i], label=f'Good {i+1}')
        
    plt.title('Price Trajectories Over Simulation')
    plt.xlabel('Iteration')
    plt.ylabel('Price')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_loss_convergence(history: Dict, save_path: str = None):
    """
    Plots the convergence of the objective loss function.
    """
    if 'loss' not in history or len(history['loss']) == 0:
        print("No loss history found.")
        return
        
    loss = np.array(history['loss'])
    iterations = np.arange(len(loss))
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, loss, color='red', linewidth=2)
    plt.yscale('log')
    plt.title('Convergence of System Planner Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (Log Scale)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
