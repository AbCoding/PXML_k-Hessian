import numpy as np
from scipy.optimize import curve_fit
from hessian_solver import solve_k_hessian
from fitting_models import model_logistic_lambda
import data_utils


def run_experiment(n_max=90, num_divisors=100, div_start=1.0, div_end=20.0):
    """
    Runs the Hessian solver experiment across a range of n/k ratios.
    Uses homotopy continuation (feeding previous n solution to next n).
    """
    # Ensure folders exist
    data_utils.ensure_directories()

    n_values = np.arange(2, n_max + 1)

    # Updated to use dynamic start and end values
    divisors = np.linspace(div_start, div_end, num_divisors)

    valid_divisors = []
    calculated_asymptotes = []

    print(f"Starting simulation up to N={n_max}")
    print(f"Divisor Range: {div_start} to {div_end} ({num_divisors} steps)")
    print("-" * 60)
    print(f"{'Divisor':<10} | {'Asymptote':<15} | {'Status'}")

    for div in divisors:
        n_data = []
        lam_data = []

        # Reset continuation for new divisor
        prev_sol = None

        # 1. Run Solver Loop
        for n in n_values:
            k = n / div

            if k >= 0.99:
                # Pass prev_sol to solver for continuation
                lam, sol = solve_k_hessian(n, k, prev_sol)

                if not np.isnan(lam):
                    n_data.append(n)
                    lam_data.append(lam)
                    # Update prev_sol to be the current successful solution
                    prev_sol = sol
                else:
                    # If a step fails, reset continuation to avoid feeding garbage
                    prev_sol = None

        n_data = np.array(n_data)
        lam_data = np.array(lam_data)

        if len(n_data) == 0:
            continue

        # 2. Save Individual Run via Utils
        data_utils.save_individual_run(n_data, lam_data, div)

        # 3. Calculate Asymptote (Fit Logistic Model)
        d_asymp = np.nan
        if len(n_data) > 8:
            try:
                # Initial guess: [a, b, c, d]
                p0 = [lam_data[0], 2.0, np.mean(n_data), lam_data[-1]]
                popt, _ = curve_fit(model_logistic_lambda, n_data, lam_data, p0=p0, maxfev=10000)
                d_asymp = popt[3]

                if d_asymp > 0:
                    valid_divisors.append(div)
                    calculated_asymptotes.append(d_asymp)
                    print(f"{div:<10.2f} | {d_asymp:<15.4f} | Saved")
                else:
                    print(f"{div:<10.2f} | {'Neg/Inv':<15} | Skipped Asymp")
            except:
                print(f"{div:<10.2f} | {'Fit Fail':<15} | Saved Raw Only")
        else:
            print(f"{div:<10.2f} | {'Too Few':<15} | Saved Raw Only")

    # 4. Save Aggregate Asymptotic Data via Utils
    if valid_divisors:
        path = data_utils.save_asymptotic_data(
            np.array(valid_divisors),
            np.array(calculated_asymptotes),
            n_max
        )
        print("-" * 60)
        print(f"Aggregate data saved to: {path}")


if __name__ == "__main__":
    run_experiment(n_max=90, num_divisors=500, div_start=1.0, div_end=50.0)