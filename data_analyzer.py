import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit, root_scalar


import data_utils
from fitting_models import model_logistic_lambda, model_power_constrained, model_power_fixed


def solve_for_p(alpha):
    """Solves the asymptotic constraint equation for p given alpha (n/k)."""
    if alpha <= 1.0:
        return np.nan

    term1 = alpha * np.log(alpha)
    term2 = (alpha - 1.0) * np.log(alpha - 1.0)
    const_part = term1 - term2

    def objective(p):
        x = alpha / p
        if x <= 0:
            return 1e9
        term3 = x * np.log(x)
        term4 = (x + 1.0) * np.log(x + 1.0)
        return const_part + term3 - term4

    try:
        res = root_scalar(objective, bracket=[1e-5, alpha * 100], method='brentq')
        if res.converged:
            return res.root
    except ValueError:
        pass
    return np.nan


def analyze_asymptotes_with_theory():
    """Plots Asymptote vs Divisor with Power Law fits AND the Theoretical Limit."""

    # 1. Load Data
    data = data_utils.load_latest_asymptote_data()
    if not data:
        print("No asymptotic data files found. Please run data_generator.py first.")
        return

    x_vals = data['divisors']
    y_vals = data['asymptotes']

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot Computational Data Points
    ax.plot(x_vals, y_vals, 'o', color='crimson', label='Computed Asymptotes', zorder=3, alpha=0.6)

    # Highlight Constraint Point (1, 4)
    ax.scatter([1], [4], color='blue', marker='x', s=100, linewidth=3, label='Constraint (1, 4)', zorder=4)

    # Generate trend line x-axis
    x_trend = np.linspace(1.0, max(x_vals), 1000)

    # --- FIT A: Empirical Free Exponent ---
    try:
        p0_opt = [1.0, 2.0]
        popt_opt, _ = curve_fit(model_power_constrained, x_vals, y_vals, p0=p0_opt, maxfev=10000)
        a_opt, b_opt = popt_opt
        label_opt = f'Empirical Free Fit: $y={a_opt:.2f}(x^{{{b_opt:.3f}}}-1)+4$'
        ax.plot(x_trend, model_power_constrained(x_trend, *popt_opt), 'k--', linewidth=2, label=label_opt, zorder=5)
    except Exception as e:
        print(f"Free fit failed: {e}")

    # --- FIT B: Empirical Fixed Exponent (5/3) ---
    try:
        p0_fixed = [1.0]
        popt_fixed, _ = curve_fit(model_power_fixed, x_vals, y_vals, p0=p0_fixed, maxfev=10000)
        a_fixed = popt_fixed[0]
        label_fixed = f'Empirical Fixed Fit: $y={a_fixed:.2f}(x^{{5/3}}-1)+4$'
        ax.plot(x_trend, model_power_fixed(x_trend, *popt_fixed), 'r-.', linewidth=2, label=label_fixed, zorder=5)
    except Exception as e:
        print(f"Fixed fit failed: {e}")

    # --- THEORETICAL LIMIT OVERLAY (Phase 5 Proof) ---
    # We evaluate for alpha > 1 to avoid NaN from solve_for_p boundary limits
    x_theory = np.linspace(1.01, max(x_vals), 1000)

    # You can compute p_values here if you want to use them for other integrals later:
    # p_theory = [solve_for_p(a) for a in x_theory]

    # As rigorously derived in Phase 5:
    y_theory = 8 * (x_theory ** (5.0 / 3.0)) - 4

    label_theory = r'Analytical Limit: $\lambda_\infty = 8\alpha^{5/3} - 4$'
    ax.plot(x_theory, y_theory, 'g-', linewidth=3, alpha=0.8, label=label_theory, zorder=6)

    # 3. Finalize Plot Details
    ax.set_title('Asymptotic Eigenvalue Scaling: Theory vs. Computation', fontsize=14)
    ax.set_xlabel(r'Divisor ($\alpha = n/k$)', fontsize=12)
    ax.set_ylabel(r'Asymptotic Limit ($\lambda_{n \to \infty}$)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save safely to avoid PyCharm bugs
    filename = "theoretical_overlay_plot.png"
    plt.savefig(filename, dpi=300)
    print(f"Plot saved successfully to {filename}")


if __name__ == "__main__":
    analyze_asymptotes_with_theory()
