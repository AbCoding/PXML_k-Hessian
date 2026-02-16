import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import warnings

# Assuming these are available in your directory as per your import
from hessian_defs import log_comb, HessianODE


def solve_k_hessian(n, k, grid_points=50, prev_sol=None):
    """
    Solves the k-Hessian equation with a specific initial grid resolution.

    Parameters:
    - n, k: dimensions
    - grid_points (M): Number of points in the initial mesh (default 50)
    - prev_sol: (Optional) Previous solution for homotopy
    """
    if k > n or k <= 0:
        return np.nan, None

    # --- 1. Generate Initial Guess ---
    # We use 'grid_points' (M) here to define resolution
    x_grid = np.linspace(0, 1, grid_points)

    if prev_sol is not None:
        # Interpolate previous solution onto new grid size if necessary
        # (Simple homotopy often just uses the old solution object,
        # but for grid study we want to force the new mesh density)
        y_guess = prev_sol.sol(x_grid)
        lam_guess = prev_sol.p[0]
    else:
        # Standard Analytic Guess
        try:
            log_ratio = log_comb(n, k)
            log_lam_guess = np.log(2.0) + (1.0 / k) * log_ratio
            lam_guess = np.exp(log_lam_guess)
            if lam_guess > 200: lam_guess = 200.0
        except:
            lam_guess = 2.0

        # Standard Grid/Array Guess
        y_guess = np.zeros((2, x_grid.size))
        y_guess[0, :] = x_grid ** 2 - 1
        y_guess[1, :] = 2 * x_grid

    # --- 2. Initialize ODE System ---
    ode_system = HessianODE(n, k)

    # --- 3. Run Solver ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            # We relax tol slightly or rely on the initial grid density.
            # Note: solve_bvp is adaptive. To strictly test grid convergence,
            # we often provide a dense initial grid.
            res = solve_bvp(
                ode_system.fun,
                ode_system.bc,
                x_grid,
                y_guess,
                p=[lam_guess],
                tol=1e-6,  # Tighter tolerance to ensure grid error dominates
                max_nodes=10000
            )
            if res.success:
                return res.p[0], res
            else:
                return np.nan, None
        except Exception as e:
            return np.nan, None