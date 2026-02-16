import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

import data_utils
from fitting_models import model_logistic_lambda, model_power_constrained, model_power_fixed


def analyze_single_runs():
    """Plots N vs Lambda for all saved runs."""

    # 1. Load Data
    runs = data_utils.load_all_single_runs()
    if not runs:
        print("No individual run files found.")
        return

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = mpl.colormaps['viridis']
    colors = cmap(np.linspace(0, 1, len(runs)))

    print(f"Plotting {len(runs)} individual runs...")

    for idx, run in enumerate(runs):
        div = run['divisor']
        n_arr = run['n']
        lam_arr = run['lambda_val']
        color = colors[idx]

        # Scatter raw data
        ax.scatter(n_arr, lam_arr, color=color, s=10, alpha=0.3)

        # Plot smooth fit line
        if len(n_arr) > 8:
            try:
                p0 = [lam_arr[0], 2.0, np.mean(n_arr), lam_arr[-1]]
                popt, _ = curve_fit(model_logistic_lambda, n_arr, lam_arr, p0=p0, maxfev=5000)
                n_line = np.linspace(min(n_arr), max(n_arr), 200)
                ax.plot(n_line, model_logistic_lambda(n_line, *popt), color=color, linewidth=1.0)
            except:
                ax.plot(n_arr, lam_arr, color=color, linestyle=':', alpha=0.3)
        else:
            ax.plot(n_arr, lam_arr, color=color, linestyle=':', alpha=0.3)

    ax.set_title('Eigenvalue $\lambda$ vs Dimension $n$ (Various $n/k$ Ratios)')
    ax.set_xlabel('Dimension $n$')
    ax.set_ylabel('Eigenvalue $\lambda$')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_asymptotes():
    """Plots Asymptote vs Divisor with Power Law fits."""

    # 1. Load Data
    data = data_utils.load_latest_asymptote_data()
    if not data:
        print("No asymptotic data files found.")
        return

    x_vals = data['divisors']
    y_vals = data['asymptotes']

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot Data Points
    ax.plot(x_vals, y_vals, 'o', color='crimson', label='Calculated Limits', zorder=3, alpha=0.6)

    # Highlight Constraint Point (1, 4)
    ax.scatter([1], [4], color='blue', marker='x', s=100, linewidth=3, label='Constraint (1, 4)', zorder=4)

    # Generate trend line x-axis
    x_trend = np.linspace(1.0, max(x_vals), 1000)

    # --- FIT A: Free Exponent ---
    try:
        p0_opt = [1.0, 2.0]
        popt_opt, _ = curve_fit(model_power_constrained, x_vals, y_vals, p0=p0_opt, maxfev=10000)
        a_opt, b_opt = popt_opt
        y_trend_opt = model_power_constrained(x_trend, *popt_opt)

        label_opt = f'Free Fit: $b={b_opt:.3f}$'
        ax.plot(x_trend, y_trend_opt, 'k--', linewidth=2, label=label_opt, zorder=5)
        print(f"Free Exponent Fit -> a: {a_opt:.4f}, b: {b_opt:.4f}")
    except Exception as e:
        print(f"Free fit failed: {e}")

    # --- FIT B: Fixed Exponent (5/3) ---
    try:
        p0_fixed = [1.0]
        popt_fixed, _ = curve_fit(model_power_fixed, x_vals, y_vals, p0=p0_fixed, maxfev=10000)
        a_fixed = popt_fixed[0]
        y_trend_fixed = model_power_fixed(x_trend, *popt_fixed)

        label_fixed = f'Fixed Fit: $b=5/3$ (1.667)'
        ax.plot(x_trend, y_trend_fixed, 'r-.', linewidth=2, label=label_fixed, zorder=5)
        print(f"Fixed Exponent Fit -> a: {a_fixed:.4f}, b: 1.6667")
    except Exception as e:
        print(f"Fixed fit failed: {e}")

    ax.set_title('Scaling of Asymptotic Eigenvalue with Divisor\nConstraint $y(1)=4$ applied')
    ax.set_xlabel('Divisor ($x = n/k$)')
    ax.set_ylabel('Asymptotic Limit ($\lambda_{n \\to \infty}$)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_single_runs()
    analyze_asymptotes()