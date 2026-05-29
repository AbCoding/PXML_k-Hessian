import numpy as np
import pickle

from solver import solve_hessian_bvp, lam_upper_bound


def generate_and_save_data(
    n_max, alphas, output_filename,
    min_step=0.1, verbose=True, debug=False,
    plot_failures=False, num_saved_points=500,
):
    """Run adaptive BVP continuation across n for each alpha and save to pickle.

    For each alpha = n/k the solver walks from small n up to n_max, using
    each converged solution as a warm start for the next step. When a step
    fails the interval is subdivided; if the sub-step is smaller than
    min_step the run stops early.

    Parameters
    ----------
    n_max : int
        Maximum dimension to reach.
    alphas : array-like
        Values of alpha = n/k to compute (each must be > 1).
    output_filename : str
        Path for the output pickle file.
    min_step : float
        Minimum adaptive subdivision step in n.
    num_saved_points : int
        Resolution of the eigenfunction grid saved in the pickle.
    """
    n_targets_base = _build_n_targets(n_max)
    x_standard = np.linspace(0, 1, num_saved_points)

    if verbose:
        print(f"Generating '{output_filename}'  (min adaptive step: {min_step})\n")

    results = {}
    for alpha in alphas:
        result = _run_continuation(
            alpha, n_max, list(n_targets_base), x_standard,
            min_step, verbose, debug, plot_failures,
        )
        if result is not None:
            results[alpha] = result

    with open(output_filename, 'wb') as f:
        pickle.dump(results, f)

    if verbose:
        print(f"Saved to {output_filename}\n")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_n_targets(n_max):
    """Build the initial list of n values to attempt."""
    if n_max > 200:
        targets = np.unique(np.geomspace(2, n_max, num=250).astype(int))
    else:
        targets = np.arange(2, n_max + 1)

    if len(targets) == 0 or targets[-1] != n_max:
        targets = np.append(targets, n_max)

    return targets


def _run_continuation(alpha, n_max, n_targets, x_standard,
                      min_step, verbose, debug, plot_failures):
    """Walk from small n up to n_max for a single alpha.

    Returns a result dict on success, or None if no step ever converged.
    """
    lam_bound = lam_upper_bound(alpha)

    if verbose:
        print(f"--- alpha={alpha:.3f}  (lambda ceiling: {lam_bound:.2f}) ---")

    prev_sol = None
    n_history, lam_history, u_history = [], [], []
    has_started = False

    idx = 0
    while idx < len(n_targets):
        n = n_targets[idx]
        k = n / alpha

        if k < 0.99:  # k must be at least ~1 for the problem to be well-posed
            idx += 1
            continue

        lam, sol = solve_hessian_bvp(
            n, k,
            grid_points=50,
            prev_sol=prev_sol,
            lam_upper_bound_val=lam_bound,
            debug=debug,
            plot_failures=plot_failures,
        )

        if not np.isnan(lam):
            has_started = True
            prev_sol = sol
            n_history.append(n)
            lam_history.append(lam)
            u_history.append(sol.sol(x_standard)[0])

            if verbose:
                print(f"  n={n:<7.2f}  k={k:<7.2f}  lam={lam:.6f}")

            idx += 1

        else:
            if verbose:
                print(f"  n={n:<7.2f}  k={k:<7.2f}  FAILED")

            if not has_started:
                # No baseline yet — keep trying higher n with a cold start
                prev_sol = None
                idx += 1
            else:
                prev_n = n_history[-1]
                step = n - prev_n
                if step / 2.0 >= min_step:
                    mid = round(prev_n + step / 2.0, 4)
                    if verbose:
                        print(f"    -> Subdividing: inserting n={mid:.4f}")
                    n_targets.insert(idx, mid)
                else:
                    if verbose:
                        print(f"    -> Step {step:.4f} < min {min_step}. Stopping.")
                    break

    if not n_history:
        if verbose:
            print(f"  alpha={alpha:.3f}: no steps converged.\n")
        return None

    if verbose:
        reached = "reached n_max" if np.isclose(n_history[-1], n_max) else f"stopped at n={n_history[-1]:.2f}"
        print(f"  alpha={alpha:.3f}: {reached} ({len(n_history)} steps).\n")

    return {
        'n_history':         np.array(n_history),
        'lam_history':       np.array(lam_history),
        'x_standard':        x_standard,
        'u_history':         np.array(u_history),
        'final_lam':         lam_history[-1],
        'x_mesh_final':      prev_sol.x,
        'u_numerical_final': prev_sol.y[0],
    }
