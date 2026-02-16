import numpy as np
import matplotlib.pyplot as plt
from hessian_solver import solve_k_hessian


def run_grid_verification(n_val, k_val, M_values):
    """
    Runs a grid convergence study for specific n and k using the provided Grid (M) values.
    """
    print("--- Starting Grid Verification Study (with Homotopy) ---")
    print(f"Testing Case: n={n_val}, k={k_val}")
    print(f"Grid Sizes to Test: {M_values}")

    lambdas = []

    # Store the solution object to use as the guess for the next iteration
    current_sol = None

    print(f"{'M (Grid Points)':<20} | {'Lambda':<20} | {'Status'}")
    print("-" * 65)

    for M in M_values:
        # Pass 'current_sol' as the guess for the next finer grid (Homotopy Continuation)
        lam, sol = solve_k_hessian(
            n_val, k_val,
            grid_points=M,
            prev_sol=current_sol
        )

        # NOTE: Ensure solve_k_hessian is using tight tolerances (e.g., tol=1e-10)
        # and high max_nodes (e.g., 100000) for this study to work correctly.

        if not np.isnan(lam):
            lambdas.append(lam)
            current_sol = sol  # Update the "best known solution"
            print(f"{M:<20} | {lam:<20.10f} | Converged")
        else:
            lambdas.append(None)
            # If a fine grid fails, don't update current_sol; try to persist
            print(f"{M:<20} | {'N/A':<20} | Failed")

    # --- Compute Relative Errors ---
    errors = []
    plot_M = []

    for i in range(len(lambdas) - 1):
        lam_coarse = lambdas[i]
        lam_fine = lambdas[i + 1]

        if lam_coarse is not None and lam_fine is not None:
            # Error metric: |Lambda_M - Lambda_{2M}|
            diff = np.abs(lam_coarse - lam_fine)

            # Avoid log(0) if they are identical (machine precision saturation)
            if diff < 1e-16:
                diff = 1e-16

            errors.append(diff)
            plot_M.append(M_values[i])

    # --- Plotting ---
    if len(errors) > 0:
        plt.figure(figsize=(10, 7))
        plt.loglog(plot_M, errors, 'o-', linewidth=2, label='Computed Error')

        # 4th Order Reference (slope -4)
        # Anchor the reference line to the first valid error point
        ref_x = np.array(plot_M)
        c4 = errors[0] * (ref_x[0] ** 4)
        ref_y4 = c4 / (ref_x ** 4)
        plt.loglog(ref_x, ref_y4, 'r:', alpha=0.7, label='4th Order Ref ($O(h^4)$)')

        plt.xlabel('Grid Size ($M$)')
        plt.ylabel('Self-Convergence Error ($|\lambda_M - \lambda_{2M}|$)')
        plt.title(f'Grid Convergence Study (n={n_val}, k={k_val})')
        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.legend()
        plt.show()
    else:
        print("\nNot enough converged data points to plot.")


if __name__ == "__main__":
    # Define inputs here
    n_input = 100
    k_input = 10
    # Standard doubling sequence
    M_input = [200, 400, 800, 1600, 3200,6400,10000]

    run_grid_verification(n_input, k_input, M_input)