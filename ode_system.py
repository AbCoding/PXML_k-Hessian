import numpy as np

from math_utils import log_binom


class HessianODE:
    """Defines the k-Hessian eigenvalue BVP on [0, 1].

    The problem is:
        S_k(D^2 u) = lambda * S_k(u * I)   on the unit ball
    reduced to the radial ODE on r in [0, 1] with boundary conditions:
        u'(0) = 0       (symmetry)
        u(0)  = -1      (normalisation)
        u(1)  = 0       (Dirichlet)

    Parameters
    ----------
    n : float
        Dimension.
    k : float
        Hessian index (k = n / alpha).
    lam_upper_bound : float
        Hard ceiling on lambda — prevents the solver from wandering to
        unphysical large values during continuation.
    """

    def __init__(self, n, k, lam_upper_bound=100_000.0):
        self.n = n
        self.k = k
        self.lam_upper_bound = lam_upper_bound
        self.l_Ak = log_binom(n - 1, k - 1)
        self.l_Bk = log_binom(n - 1, k)
        self.l_B0 = 0.0

    def fun(self, x, y, p):
        """ODE right-hand side: returns [u', u''] given state [u, u']."""
        lam = np.clip(np.abs(p[0]) + 1e-30, 1e-30, self.lam_upper_bound)
        u, up = y[0], y[1]
        upp = np.zeros_like(x)

        # Interior points (away from origin — avoids 1/r singularity)
        nz = x > 1e-5
        if np.any(nz):
            r = x[nz]
            P = np.abs(up[nz] / r) + 1e-12
            log_lam_u = np.log(lam * np.abs(u[nz]) + 1e-20)
            log_P = np.log(P)

            L1 = self.l_B0 + self.k * log_lam_u
            L2 = self.l_Bk + self.k * log_P
            L_den = self.l_Ak + (self.k - 1) * log_P
            L_max = np.maximum(L1, L2)

            upp[nz] = np.exp(np.clip(L_max - L_den, -700, 700)) * (
                np.exp(L1 - L_max) - np.exp(L2 - L_max)
            )

        # Origin (r ~ 0): use L'Hopital limit of the ODE
        z = ~nz
        if np.any(z):
            l_Cnk = log_binom(self.n, self.k)
            log_factor = np.clip(-l_Cnk / self.k, -700, 700)
            upp[z] = lam * np.abs(u[z]) * np.exp(log_factor)

        return np.vstack((up, upp))

    def bc(self, ya, yb, p):
        """Boundary conditions: [u'(0), u(0)+1, u(1)]."""
        return np.array([ya[1], ya[0] + 1, yb[0]])
