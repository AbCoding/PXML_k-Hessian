# Asymptotic Eigenvalue Bounds for the $k$-Hessian Operator

Numerical verification of sharp asymptotic bounds on the first $k$-Hessian
eigenvalue $\lambda_{k,n}^{1/k}$ as $n, k \to \infty$ with $\alpha = n/k$ fixed,
supporting an upcoming paper.

> **Preprint:** *coming soon* <!-- replace with arXiv link once posted -->

---

## Problem Statement

### The Operator

The $k$-Hessian operator $S_k(D^2 u)$ is the $k$-th elementary symmetric
polynomial of the eigenvalues of the Hessian $D^2 u$. If $\lambda_1, \ldots, \lambda_n$
denote those eigenvalues,

$$S_k(D^2 u) = \sum_{1 \le i_1 < \cdots < i_k \le n} \lambda_{i_1} \cdots \lambda_{i_k}.$$

This family includes the Laplacian ($k = 1$) and the Monge–Ampère operator ($k = n$)
as endpoint cases.

### The Eigenvalue Problem

We consider radially symmetric solutions $u = u(r)$ on the unit ball $B \subset \mathbb{R}^n$
satisfying

$$S_k(D^2 u) = \lambda \, S_k(u \, I), \quad u\big|_{\partial B} = 0, \quad u(0) = -1,$$

where $I$ is the identity and $\lambda > 0$. For radial $u$ this reduces to
$S_k(u\,I) = \binom{n}{k}|u|^k$, making the problem nonlinear in $\lambda$.
The first eigenvalue is denoted $\lambda_{k,n}$.

### The Scaling Parameter

The asymptotic regime is parametrised by

$$\alpha = \frac{n}{k} \in (1, \infty).$$

With $\alpha$ fixed and $k \to \infty$, the quantity $\lambda_{k,\alpha k}^{1/k}$
converges to a finite limit. The $1/k$ exponent is the correct normalisation:
$\lambda_{k,n}$ itself grows super-exponentially.

---

## Theoretical Results

### Asymptotic Limit

$$\lim_{k \to \infty} \lambda_{k,\alpha k}^{1/k} = \lambda_\infty(\alpha) := \alpha^2 \left(\frac{2}{\alpha}\right)^{2/(2-\alpha)} \left(\frac{\alpha}{\alpha - 1}\right)^{\alpha - 1}.$$

At $\alpha = 2$ (the $p = 1$ regime, $k = n/2$) L'Hôpital gives $\lambda_\infty(2) = 8e$.

### Sharp Two-Sided Bounds

For all finite $n$ and $k$ we establish

$$\ell_k(n) \;\le\; \lambda_{k,n}^{1/k} \;\le\; u_k(n),$$

where both $\ell_k(n)$ and $u_k(n)$ converge to $\lambda_\infty(\alpha)$ as $k \to \infty$.
The lower bound $\ell_k$ follows from a combinatorial estimate on $S_k$; the upper
bound $u_k$ is the Rayleigh quotient evaluated at the test function $u_0$ defined below.

### Limit Eigenfunction

The radial eigenfunctions $u_{k,n}$ converge in $L^1[0,1]$ to

$$u_0(r, \alpha) = \begin{cases} -\exp\left(-\dfrac{\alpha}{2 r_\alpha^2} r^2\right) & r \le r_\alpha, \\ \dfrac{2}{\sigma} e^{-\alpha/2} \left(r^\sigma - 1\right) & r > r_\alpha, \end{cases}$$

where $\sigma = 2 - \alpha$ and $r_\alpha = (\alpha/2)^{1/\sigma}$. The function $u_0$
is $C^1$ at the junction $r_\alpha$ and serves as both the Rayleigh-quotient test
function and the predicted limiting shape.

---

## Numerical Results ($\alpha = 1.5$)

### Eigenvalue Convergence against Bounds

$\lambda_{k,n}^{1/k}$ (black dots) against $n$, with $\ell_k$ (red dashed),
$u_k$ (blue dash-dot), and $\lambda_\infty$ (grey dotted).

![Eigenvalue bounds](https://raw.githubusercontent.com/AbCoding/PXML_k-Hessian/main/figures/bounds/hessian_bounds_alpha1.50.png)

### Bound Residuals

Signed gaps $\lambda_{k,n}^{1/k} - \ell_k$ and $\lambda_{k,n}^{1/k} - u_k$,
confirming $\lambda_{k,n}^{1/k}$ lies strictly between the two bounds for all computed $n$.

![Residuals](https://raw.githubusercontent.com/AbCoding/PXML_k-Hessian/main/figures/residuals/hessian_residuals_alpha1.50.png)

### Eigenfunction Convergence to $u_0$

Radial eigenfunctions $u_{k,n}$ for all computed $n$ (grey gradient, log-scaled in $n$),
with $n = 1, 5, 20, 100$ highlighted and the limit $u_0$ overlaid (black dashed).

![Eigenfunctions](https://raw.githubusercontent.com/AbCoding/PXML_k-Hessian/main/figures/eigenfunctions/eigenfunctions_alpha1.50.png)

Selected curves only:

![Eigenfunctions highlights](https://raw.githubusercontent.com/AbCoding/PXML_k-Hessian/main/figures/eigenfunctions/eigenfunctions_highlights_alpha1.50.png)

### $L^1$ Convergence Rate

$\|u_{k,n} - u_0\|_{L^1[0,1]}$ against $n$ on a log-log scale with power-law fit.
The slope is the convergence exponent.

![L1 convergence](https://raw.githubusercontent.com/AbCoding/PXML_k-Hessian/main/figures/convergence/convergence_alpha1.50.png)

---

## Computational Method

$\lambda_{k,n}$ is computed by posing the radial ODE as a boundary value problem
on $[0,1]$ with $\lambda$ as an additional unknown, subject to
$u'(0) = 0$, $u(0) = -1$, $u(1) = 0$.

Solutions are obtained by continuation in $n$: each converged solution seeds the
solver at the next value of $n$. Failed steps trigger automatic interval subdivision,
with a prescribed minimum step size as a stopping criterion.

The bounds $\ell_k$ and $u_k$ are evaluated by Gauss quadrature of the
Rayleigh-quotient integrals using log-scale arithmetic throughout to prevent
overflow at large $n$ and $k$.

---

## Reproducing the Results

Parameters ($\alpha$ values, maximum $n$, output path) are set in `run.py`.
Executing `run.py` generates the data and all plots.
