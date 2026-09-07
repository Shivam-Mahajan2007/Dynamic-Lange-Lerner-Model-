import numpy as np
from typing import List, Dict

from Simulation.src.config import settings
from Simulation.src.optimization.gradient_descent import nesterov_gradient_descent
from Simulation.src.optimization.subproblem_solver import solve_agent_subproblem

class SimulationEngine:
    def __init__(self):
        """
        Initializes the simulation environment, loading parameters from settings.
        """
        self.num_households = settings.NUM_HOUSEHOLDS
        self.num_firms = settings.NUM_FIRMS
        self.num_goods = settings.NUM_GOODS
        
        self.A = settings.LEONTIEF_MATRIX
        self.K_req = settings.CAPITAL_ACCOUNTS_MATRIX
        self.L_req = settings.LABOR_USE_VECTOR
        
        # Endowments
        self.L_total = settings.TOTAL_LABOR_ENDOWMENT
        self.K_total = settings.TOTAL_CAPITAL_ENDOWMENT
        
        # State variables (e.g., prices)
        # Initialize prices arbitrarily, e.g., ones
        self.prices = np.ones(self.num_goods)
        
    def _aggregate_excess_demand(self, current_prices: np.ndarray) -> np.ndarray:
        """
        Computes aggregate excess demand across the economy for given prices.
        This represents the gradient for the main price adjustment process.
        
        In the Dynamic Lange-Lerner model, the central planner adjusts prices
        based on the excess demand reported by agents solving their subproblems.
        """
        # --- Stub implementation ---
        # 1. Firms solve profit maximization (subproblem_solver)
        # 2. Households solve utility maximization (subproblem_solver)
        # 3. Sum demands, subtract supply and endowments.
        # Here we just mock a gradient that pushes prices towards a hypothetical equilibrium
        
        equilibrium_prices = np.ones(self.num_goods) * 2.0
        excess_demand = equilibrium_prices - current_prices 
        
        return excess_demand
        
    def _planner_loss_function(self, current_prices: np.ndarray) -> float:
        """
        Hypothetical objective function for tracking convergence.
        (Minimizing the squared norm of excess demand).
        """
        excess_demand = self._aggregate_excess_demand(current_prices)
        return 0.5 * np.sum(excess_demand ** 2)

    def run_simulation(self, max_iter: int = 500) -> Dict:
        """
        Runs the main simulation loop.
        
        Uses Nesterov gradient descent to update prices based on excess demand.
        Returns the history of the simulation for analysis.
        """
        print("Starting Dynamic Lange-Lerner Simulation...")
        
        # We want to find prices where excess demand is zero.
        # This is equivalent to minimizing the squared norm of excess demand,
        # but gradient descent on prices typically steps in the direction of excess demand:
        # p_{t+1} = p_t + alpha * Z(p_t)
        # So we can formulate it as finding roots, or minimizing a potential function.
        # For our nesterov optimizer (which minimizes), we negate the excess demand to act as a gradient.
        
        def grad_func(p):
            # Gradient is negative excess demand to push prices up when demand is positive
            return -self._aggregate_excess_demand(p)
            
        optimal_prices, history = nesterov_gradient_descent(
            initial_x=self.prices,
            grad_func=grad_func,
            lr=0.05,
            momentum=0.9,
            max_iter=max_iter,
            tol=1e-5,
            obj_func=self._planner_loss_function
        )
        
        self.prices = optimal_prices
        print("Simulation Completed.")
        
        return history
