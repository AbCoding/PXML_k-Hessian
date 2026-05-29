import numpy as np
from scipy.special import gammaln


def log_binom(n, k):
    """Natural log of the binomial coefficient C(n, k).

    Accepts non-integer n and k via the Gamma-function generalisation.
    Returns -inf when k < 0 or k > n.
    """
    if np.ndim(n) == 0 and (k < 0 or k > n):
        return -np.inf
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def u0(r, alpha):
    """Theoretical limit eigenfunction u_0(r) for the k-Hessian problem.

    Convention: alpha = n/k in (1, inf).
    Accepts scalar or array r; returns matching type.

    The function has two pieces joined at r_alpha = (alpha/2)^(1/sigma):
      - Gaussian well for r <= r_alpha
      - Power-law tail for r > r_alpha
    where sigma = 2 - alpha.
    """
    scalar = np.ndim(r) == 0
    r = np.atleast_1d(np.asarray(r, dtype=float))

    if np.isclose(alpha, 2.0):
        alpha = 2.0 - 1e-7  # avoid sigma=0 (p=1 degenerate case)

    sigma = 2.0 - alpha
    r_alpha = (alpha / 2.0) ** (1.0 / sigma)
    c = alpha / r_alpha ** 2
    A = (2.0 / sigma) * np.exp(-alpha / 2.0)

    r_safe = np.where(r > 0, r, 1e-300)  # guard against r^sigma at r=0 when sigma<0
    out = np.where(r <= r_alpha,
                   -np.exp(-c * r ** 2 / 2.0),
                   A * (r_safe ** sigma - 1.0))
    return float(out[0]) if scalar else out


def du0(r, alpha):
    """Derivative u_0'(r). Scalar r only — used inside scipy quad integration."""
    if np.isclose(alpha, 2.0):
        alpha = 2.0 - 1e-7

    sigma = 2.0 - alpha
    r_alpha = (alpha / 2.0) ** (1.0 / sigma)
    c = alpha / r_alpha ** 2
    A = (2.0 / sigma) * np.exp(-alpha / 2.0)

    if r <= r_alpha:
        return c * r * np.exp(-c * r ** 2 / 2.0)
    else:
        return A * sigma * r ** (sigma - 1)


def asymptotic_limit(alpha):
    """Limiting eigenvalue lim_{k->inf} lambda_{k, alpha*k}^{1/k}.

    Closed form (new convention alpha = n/k > 1):
        lambda_inf = alpha^2 * (2/alpha)^(2/(2-alpha)) * (alpha/(alpha-1))^(alpha-1)

    Special case alpha=2 (p=1): limit is 8e via L'Hopital.
    """
    if np.isclose(alpha, 2.0):
        return 8.0 * np.exp(1.0)

    gamma = 1.0 / alpha
    return (
        (1.0 / gamma ** 2)
        * (2.0 * gamma) ** ((2.0 * gamma) / (2.0 * gamma - 1.0))
        / (1.0 - gamma) ** ((1.0 - gamma) / gamma)
    )
