import numpy as np
from scipy.special import betaln, gammaln
from scipy.integrate import quad
import warnings

from math_utils import log_binom, u0, du0


# ---------------------------------------------------------------------------
# Integrand helpers (defined at module level so they aren't re-created inside
# the n-loop on every iteration)
# ---------------------------------------------------------------------------

def _log_integrand_den(r, n, k, alpha):
    """Log of the denominator integrand: r^(n-1) * |u0|^(k+1)."""
    v = abs(u0(r, alpha))
    return (n - 1) * np.log(r) + (k + 1) * np.log(v) if r > 0 and v > 0 else -np.inf


def _log_integrand_num(r, n, k, alpha):
    """Log of the numerator integrand: r^(n-k) * (u0')^(k+1)."""
    v = du0(r, alpha)
    return (n - k) * np.log(r) + (k + 1) * np.log(v) if r > 0 and v > 0 else -np.inf


def _log_lower_bound(n, k, log_common):
    """Log-scale lower bound for a single (n, k) pair.

    Three branches depending on whether 2k compares to n — each corresponds
    to a different form of the combinatorial estimate.
    """
    if 2 * k > n:
        q = (2 * k - n) / k
        return log_common + k * np.log(q) - (-np.log(q) + betaln(n / q, k + 1))
    elif 2 * k < n:
        m = (n - 2 * k) / k
        return log_common + k * np.log(m) - (-np.log(m) + betaln(n / m - k, k + 1))
    else:  # 2k == n exactly
        return log_common + (k + 1) * np.log(n) - gammaln(k + 1)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def compute_bounds(alpha, n_vals):
    """Lower and upper bounds on lambda_{k,n}^{1/k} for fixed alpha = n/k.

    Parameters
    ----------
    alpha : float
        Ratio n/k, must be > 1.
    n_vals : array-like
        Dimension values to evaluate; k = n/alpha at each point.

    Returns
    -------
    lower_bounds, upper_bounds : ndarray, ndarray
    """
    n_vals = np.asarray(n_vals, dtype=float)
    lower_bounds = np.zeros_like(n_vals)
    upper_bounds = np.zeros_like(n_vals)

    # alpha=2 is a degenerate case (sigma=0); nudge to avoid division by zero
    alpha_safe = 2.0 - 1e-7 if np.isclose(alpha, 2.0) else alpha
    sigma = 2.0 - alpha_safe
    r_alpha = (alpha_safe / 2.0) ** (1.0 / sigma)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, n in enumerate(n_vals):
            k = n / alpha
            log_common = -np.log(k) + log_binom(n - 1, k - 1)

            lower_bounds[i] = np.exp(_log_lower_bound(n, k, log_common) / k)

            # Upper bound via Rayleigh-quotient integration, scaled to avoid overflow
            M_den = _log_integrand_den(r_alpha, n, k, alpha_safe)
            M_num = _log_integrand_num(r_alpha, n, k, alpha_safe)

            int_den, _ = quad(
                lambda r: np.exp(_log_integrand_den(r, n, k, alpha_safe) - M_den),
                0, 1, points=[r_alpha]
            )
            int_num, _ = quad(
                lambda r: np.exp(_log_integrand_num(r, n, k, alpha_safe) - M_num),
                0, 1, points=[r_alpha]
            )

            log_upper = log_common + (M_num + np.log(int_num)) - (M_den + np.log(int_den))
            upper_bounds[i] = np.exp(log_upper / k)

    return lower_bounds, upper_bounds
