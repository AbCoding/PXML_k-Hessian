import numpy as np
from scipy.special import gammaln


def log_comb(n, k):
    """Computes log(nCk) safely using gamma functions."""
    if k < 0 or k > n:
        return -np.inf
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


class HessianODE:
    """Class to encapsulate the k-Hessian ODE parameters and equations."""

    def __init__(self, n, k):
        self.n = n
        self.k = k

        # Precompute coefficients
        l_Ak = log_comb(n - 1, k - 1)
        l_Bk = log_comb(n - 1, k)
        l_B0 = 0.0  # log(1)
        max_log = max(l_Ak, l_Bk, l_B0)

        self.A_k = np.exp(l_Ak - max_log)
        self.B_k = np.exp(l_Bk - max_log)
        self.B_0 = np.exp(l_B0 - max_log)

    def fun(self, x, y, p):
        lam = p[0]
        u = y[0]
        up = y[1]
        upp = np.zeros_like(x)

        # Handle non-singular region
        mask_nz = x > 1e-5
        if np.any(mask_nz):
            r = x[mask_nz]
            u_val = np.abs(u[mask_nz])
            up_val = up[mask_nz]

            P = np.abs(up_val / r) + 1e-12
            rhs = (lam * u_val) ** self.k

            num = self.B_0 * rhs - self.B_k * (P ** self.k)
            den = self.A_k * (P ** (self.k - 1))
            upp[mask_nz] = num / (den + 1e-15)

        # Handle singularity at x=0
        mask_z = ~mask_nz
        if np.any(mask_z):
            l_Cnk = log_comb(self.n, self.k)
            l_Cnm = 0.0
            log_factor = (l_Cnm - l_Cnk) / self.k
            log_factor = np.clip(log_factor, -700, 700)
            factor = np.exp(log_factor)
            upp[mask_z] = lam * np.abs(u[mask_z]) * factor

        return np.vstack((up, upp))

    def bc(self, ya, yb, p):
        # Boundary conditions: u'(0)=0 (implicit), u(0)=-1, u(1)=0
        return np.array([ya[1], ya[0] + 1, yb[0]])