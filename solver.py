import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

from math_utils import log_binom, asymptotic_limit
from ode_system import HessianODE


def lam_upper_bound(alpha):
    """2× the theoretical asymptotic limit — safe ceiling for the BVP solver."""
    return 2.0 * asymptotic_limit(alpha)


def solve_hessian_bvp(n, k, grid_points=100, prev_sol=None,
                      lam_upper_bound_val=100_000.0, debug=False, plot_failures=False):
    """Solve the k-Hessian BVP for a single (n, k) pair.

    Uses the previous solution as a warm start when available, falling back
    to a built-in initial guess otherwise. Tries progressively looser
    tolerances if the first attempt fails.

    Parameters
    ----------
    n, k : float
        Dimension and Hessian index. Must satisfy 0 < k <= n.
    grid_points : int
        Number of initial mesh points (ignored when warm-starting).
    prev_sol : BVP solution or None
        Warm-start solution from the previous continuation step.
    lam_upper_bound_val : float
        Hard ceiling on lambda passed to the ODE circuit-breaker.
    debug, plot_failures : bool
        Extra diagnostics on failure.

    Returns
    -------
    lam : float   — eigenvalue, or nan on failure.
    sol : object  — BVP solution object, or None on failure.
    """
    if k > n or k <= 0:
        return np.nan, None

    lam_guess = _initial_lam_guess(n, k, lam_upper_bound_val, prev_sol)
    x, y, tols = _initial_mesh(n, k, grid_points, prev_sol)
    ode = HessianODE(n, k, lam_upper_bound_val)

    res = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for attempt, tol in enumerate(tols):
            try:
                res = solve_bvp(ode.fun, ode.bc, x, y,
                                p=[lam_guess], tol=tol, max_nodes=500_000)
                if res.success:
                    if attempt > 0 and debug:
                        print(f"      -> Converged on attempt {attempt + 1} (tol={tol:.1e})")
                    return res.p[0], res
            except Exception as e:
                if debug:
                    print(f"      -> Exception on attempt {attempt + 1}: {e}")

    if debug:
        _print_failure(res)
    if plot_failures:
        _save_failure_plot(res, n, k, n / k, x, y, prev_sol, debug)

    return np.nan, None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _initial_lam_guess(n, k, lam_upper_bound_val, prev_sol):
    """Compute an initial lambda guess, preferring the warm-start value."""
    if prev_sol is not None:
        return prev_sol.p[0]
    try:
        log_guess = np.log(2.0) + (1.0 / k) * log_binom(n, k)
        guess = np.exp(log_guess)
        if guess > lam_upper_bound_val * 0.9:
            return lam_upper_bound_val * 0.9
        if guess > 200 and lam_upper_bound_val > 200:
            return 200.0
        return guess
    except Exception:
        return 2.0


def _initial_mesh(n, k, grid_points, prev_sol):
    """Build the initial grid, function values, and tolerance schedule.

    Two strategies based on the ratio k/n:
      - k/n > 0.55  (alpha < 1.818): uniform grid, tight tolerances.
      - k/n <= 0.55 (alpha >= 1.818): sine-clustered grid near r=0,
                                       relaxed tolerances for large n.
    """
    alpha = n / k

    if alpha < 1.0 / 0.55:  # k/n > 0.55 in old convention
        if prev_sol is not None:
            x = np.linspace(0, 1, grid_points)
            y = prev_sol.sol(x)
        else:
            x = np.linspace(0, 1, grid_points)
            y = np.vstack((x ** 2 - 1, 2 * x))
        tols = [1e-8, 1e-7, 1e-6]

    else:
        if prev_sol is not None:
            # Thin the mesh if it has grown very large to keep memory reasonable
            x = prev_sol.x[::2] if len(prev_sol.x) > 3000 else prev_sol.x
            y = prev_sol.sol(x) if len(prev_sol.x) > 3000 else prev_sol.y
        else:
            x = np.sin(np.linspace(0, np.pi / 2, grid_points))
            y = np.vstack((x ** 2 - 1, 2 * x))
        tols = [1e-6, 1e-6, 5e-6] if n <= 100 else [1e-4, 1e-4, 1e-4]

    return x, y, tols


def _print_failure(res):
    lam = res.p[0] if res is not None and res.p is not None else float("nan")
    max_u = np.max(np.abs(res.y[0])) if res is not None and res.y is not None else float("nan")
    status = getattr(res, "status", "unknown")
    message = getattr(res, "message", "exception occurred")
    mesh = res.x.size if res is not None and res.x is not None else 0
    print(f"      -> ALL FAILED. Status {status}: {message}")
    print(f"      -> Mesh: {mesh} nodes | lam={lam:.6f} | max|u|={max_u:.6f}")


def _save_failure_plot(res, n, k, alpha, x_guess, y_guess, prev_sol, debug):
    try:
        if res is not None and res.x is not None and res.y is not None:
            plt.figure(figsize=(8, 5))
            plt.plot(res.x, res.y[0], 'r-', linewidth=2, label='Failed $u(x)$')
            if prev_sol is not None:
                plt.plot(x_guess, y_guess[0], 'k--', alpha=0.5, label='Initial guess')
            plt.title(f"Failed: n={n}, k={k:.2f}")
            plt.grid(True, linestyle='--')
            plt.legend()
            plt.savefig(f"fail_plot_alpha{alpha:.3f}_n{n}.png", dpi=150)
            plt.close()
    except Exception as e:
        if debug:
            print(f"      -> Failure plot error: {e}")
