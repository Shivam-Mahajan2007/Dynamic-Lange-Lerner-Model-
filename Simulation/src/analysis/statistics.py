import numpy as np
import pandas as pd
from typing import Dict

def compute_summary_statistics(history: Dict) -> pd.DataFrame:
    """
    Computes summary statistics from the simulation history.
    This analysis is designed to be scale-invariant (independent of 
    the absolute number of households/firms).
    
    Args:
        history: Dictionary containing the trajectory of states (e.g. prices, loss).
        
    Returns:
        pd.DataFrame containing summary metrics.
    """
    if not history or 'x' not in history:
        return pd.DataFrame()
        
    prices = np.array(history['x'])
    
    # Calculate price volatility (std over mean across iterations)
    # Exclude initial transient steps (e.g., first 10%)
    burn_in = max(1, len(prices) // 10)
    steady_state_prices = prices[burn_in:]
    
    price_means = np.mean(steady_state_prices, axis=0)
    price_stds = np.std(steady_state_prices, axis=0)
    price_volatility = price_stds / (price_means + 1e-9)
    
    # Final gradient norm (measure of how close to equilibrium)
    final_grad_norm = history['grad_norm'][-1] if 'grad_norm' in history and len(history['grad_norm']) > 0 else np.nan
    
    # Compile into dataframe
    stats_dict = {
        'Metric': [
            'Final Gradient Norm',
            'Mean Price Volatility (Post Burn-in)',
            'Max Price Volatility (Post Burn-in)',
            'Convergence Iterations'
        ],
        'Value': [
            final_grad_norm,
            np.mean(price_volatility),
            np.max(price_volatility),
            len(prices)
        ]
    }
    
    df = pd.DataFrame(stats_dict)
    return df
