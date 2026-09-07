import cvxpy as cp
import numpy as np

def solve_agent_subproblem(
    objective_coeffs: np.ndarray,
    constraint_matrix: np.ndarray,
    constraint_bounds: np.ndarray,
    is_maximization: bool = True
) -> dict:
    """
    Solves a generic linear or quadratic subproblem for an agent using CVXPY.
    
    This acts as a base solver for the Dynamic Lange-Lerner subproblems 
    (e.g., household utility maximization or firm profit maximization) 
    given prices and budgets.
    
    Args:
        objective_coeffs: Coefficients for the objective function (assumes linear for now).
        constraint_matrix: Matrix A in the constraint A * x <= b.
        constraint_bounds: Vector b in the constraint A * x <= b.
        is_maximization: If True, maximizes the objective; otherwise, minimizes.
        
    Returns:
        A dictionary containing:
            'status': Optimization status (e.g., 'optimal').
            'value': Optimal value of the objective.
            'x': The optimal variable values.
    """
    n_vars = objective_coeffs.shape[0]
    
    # Define variables
    x = cp.Variable(n_vars)
    
    # Define objective (linear by default)
    objective_expr = objective_coeffs.T @ x
    if is_maximization:
        objective = cp.Maximize(objective_expr)
    else:
        objective = cp.Minimize(objective_expr)
        
    # Define constraints (A * x <= b) and positivity (x >= 0)
    constraints = [
        constraint_matrix @ x <= constraint_bounds,
        x >= 0
    ]
    
    # Define problem
    problem = cp.Problem(objective, constraints)
    
    # Solve
    try:
        problem.solve()
    except cp.error.SolverError:
        problem.solve(solver=cp.SCS) # Fallback solver
        
    result = {
        'status': problem.status,
        'value': problem.value,
        'x': x.value
    }
    
    return result

def solve_quadratic_subproblem(
    Q: np.ndarray,
    p: np.ndarray,
    A: np.ndarray = None,
    b: np.ndarray = None,
    G: np.ndarray = None,
    h: np.ndarray = None
) -> dict:
    """
    Solves a convex quadratic subproblem:
    Minimize 1/2 * x^T * Q * x + p^T * x
    Subject to:
        A * x = b
        G * x <= h
        
    Use this for strictly convex utility functions or production cost formulations.
    """
    n_vars = p.shape[0]
    x = cp.Variable(n_vars)
    
    # 1/2 x^T Q x + p^T x
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + p.T @ x)
    
    constraints = []
    if A is not None and b is not None:
        constraints.append(A @ x == b)
    if G is not None and h is not None:
        constraints.append(G @ x <= h)
        
    # Standard non-negativity
    constraints.append(x >= 0)
    
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return {
        'status': problem.status,
        'value': problem.value,
        'x': x.value
    }
