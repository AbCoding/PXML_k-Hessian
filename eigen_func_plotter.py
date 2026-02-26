import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from hessian_solver import solve_k_hessian


def model_logistic_lambda(n, a, b, c, d):
    """
    As specified: y(n) = d + (a - d) / (1 + (n/c)^b)
    d is the horizontal asymptote (lambda_inf)
    """
    n = np.asarray(n, dtype=float)
    return d + (a - d) / (1.0 + (n / c) ** b)


def theoretical_eigenfunction(r, gamma, lam_inf):
    """
    u(r) = -exp( -gamma (1-gamma)^((1-gamma)/gamma) (lam_inf/2) r^2 )
    """
    coeff = (
            gamma
            * (1 - gamma) ** ((1 - gamma) / gamma)
            * (lam_inf / 2.0)
    )
    return -np.exp(-coeff * r ** 2)


def get_lam_inf_formula(gamma):
    """
    Calculates the theoretical lambda_inf formula.
    """
    if np.isclose(gamma, 0.5):
        return 8.0 * np.exp(1.0)
    term1 = 1.0 / (gamma ** 2)
    term2 = (2.0 * gamma) ** (2.0 * gamma / (2.0 * gamma - 1.0))
    term3 = (1.0 - gamma) ** (1.0 / gamma - 1.0)
    return term1 * term2 / term3


def plot_final_eigenfunctions(n_max, gammas, figure_title, filename, show_lines):
    n_values_all = np.arange(2, n_max + 1)
    n_gamma = len(gammas)
    fig, axes = plt.subplots(n_gamma, 1, figsize=(8, 4 * n_gamma), sharex=True)

    if n_gamma == 1:
        axes = [axes]

    print(f"\n{figure_title}")

    for ax, gamma in zip(axes, gammas):
        prev_sol = None
        final_sol = None
        final_lam = None

        n_history = []
        lam_history = []

        # --- Homotopy Continuation ---
        for n in n_values_all:
            k = n * gamma
            if k >= 0.99:
                lam, sol = solve_k_hessian(n, k, prev_sol=prev_sol)
                if not np.isnan(lam):
                    prev_sol = sol
                    n_history.append(n)
                    lam_history.append(lam)
                    if n == n_max:
                        final_sol = sol
                        final_lam = lam
                else:
                    prev_sol = None

        if final_sol is not None:
            x_mesh = final_sol.x
            r = (x_mesh - x_mesh[0]) / (x_mesh[-1] - x_mesh[0]) if x_mesh[-1] != x_mesh[0] else x_mesh

            # 1. Data (Numerical)
            if show_lines.get('data', True):
                numerical = final_sol.y[0]
                numerical /= np.max(np.abs(numerical))
                ax.plot(r, numerical, label="Data (Numerical)", color='black', linewidth=2)

            # 2. Theoretical using Logistic lambda_inf (The "Extrap" Line)
            if show_lines.get('lam_inf_extrap', True) and len(lam_history) > 4:
                try:
                    # Initial guess for [a, b, c, d]
                    p0 = [lam_history[0], 1.0, n_max / 2, lam_history[-1] * 1.05]

                    # Define bounds: ( [min_a, min_b, min_c, min_d], [max_a, max_b, max_c, max_d] )
                    # d (index 3) is constrained to be > 0.
                    # We use a very small epsilon (1e-10) to ensure it is strictly positive.
                    lower_bounds = [-np.inf, -np.inf, -np.inf, 1e-10]
                    upper_bounds = [np.inf, np.inf, np.inf, np.inf]

                    params, _ = curve_fit(
                        model_logistic_lambda,
                        n_history,
                        lam_history,
                        p0=p0,
                        bounds=(lower_bounds, upper_bounds),
                        maxfev=5000
                    )

                    lambda_inf_logistic = params[3]  # Extracting d

                    theoretical_logistic = theoretical_eigenfunction(r, gamma, lambda_inf_logistic)
                    theoretical_logistic /= np.max(np.abs(theoretical_logistic))
                    ax.plot(r, theoretical_logistic, "--", color='darkorange',
                            label=rf"Theoretical ($\lambda_{{\infty}}$ logistic $\approx {lambda_inf_logistic:.2f}$)")
                except Exception as e:
                    print(f"      Logistic fit failed for gamma={gamma}: {e}")

            # 3. Theoretical using lambda_n (final_lam)
            if show_lines.get('lam_n', True):
                theoretical_n = theoretical_eigenfunction(r, gamma, final_lam)
                theoretical_n /= np.max(np.abs(theoretical_n))
                ax.plot(r, theoretical_n, "-.", label=rf"Theoretical ($\lambda_{{{n_max}}} \approx {final_lam:.2f}$)")

            # 4. Theoretical using the specific lambda_inf formula
            if show_lines.get('lam_inf_formula', True):
                lam_inf_formula_val = get_lam_inf_formula(gamma)
                theoretical_form = theoretical_eigenfunction(r, gamma, lam_inf_formula_val)
                theoretical_form /= np.max(np.abs(theoretical_form))
                ax.plot(r, theoretical_form, ":", color='red',
                        label=rf"Theoretical ($\lambda_{{\infty}}$ formula $\approx {lam_inf_formula_val:.2f}$)")

            ax.set_title(rf"$\gamma={gamma:.3f}$")
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend(loc='best', fontsize='small')
            print(f"  -> gamma={gamma:.3f} OK")
        else:
            ax.set_title(rf"$\gamma={gamma:.3f}$ (no convergence)")
            print(f"  -> gamma={gamma:.3f} FAILED")

    axes[-1].set_xlabel("Normalized Domain Space")
    fig.suptitle(figure_title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    n_max = 200
    SHOW_LINES = {
        'data': True,
        'lam_inf_extrap': True,
        'lam_n': True,
        'lam_inf_formula': True
    }

    # Regime I: High Gamma
    plot_final_eigenfunctions(n_max, np.linspace(0.5, 0.95, 5),
                              r"Regime I: $\gamma \in [1/2,1)$",
                              "eigenfunctions_gamma_high.png", SHOW_LINES)

    # Regime II: Low Gamma
    plot_final_eigenfunctions(n_max, np.linspace(0.03, 0.5, 5),
                              r"Regime II: $\gamma \in [0.03,1/2]$",
                              "eigenfunctions_gamma_low.png", SHOW_LINES)