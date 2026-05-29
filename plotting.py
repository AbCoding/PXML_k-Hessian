import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from math_utils import u0, asymptotic_limit
from bounds import compute_bounds


# ---------------------------------------------------------------------------
# Paper-quality rcParams
# ---------------------------------------------------------------------------

PAPER_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.5,
}

# n values to highlight on eigenfunction plots, sampled from the plasma colormap
HIGHLIGHT_NS = [1, 5, 20, 100]


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_numerical_vs_bounds(pkl_filename, plot_dir="eigenvalue plots"):
    """Numerical eigenvalue paths overlaid on the smooth theoretical bounds."""
    results = _load(pkl_filename)
    if results is None:
        return
    os.makedirs(plot_dir, exist_ok=True)

    with matplotlib.rc_context(PAPER_RC):
        for alpha in sorted(results):
            data = results[alpha]
            n_vals = np.array(data['n_history'])
            lam_num = np.array(data['lam_history'])

            _check_monotone(lam_num, n_vals, alpha)

            n_smooth = np.linspace(n_vals.min(), n_vals.max(), 200)
            lower, upper = compute_bounds(alpha, n_smooth)

            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            ax.plot(n_vals, lam_num, 'o', color="#222222", markersize=3,
                    markeredgewidth=0, zorder=3,
                    label=r'Numerical $\lambda_{n/\alpha,\, n}^{\alpha/n}$')
            ax.plot(n_smooth, lower, '--', color="#c0392b", linewidth=1.5,
                    label=r'Lower bound')
            ax.plot(n_smooth, upper, '-.', color="#2471a3", linewidth=1.5,
                    label=r'Upper bound')
            ax.axhline(asymptotic_limit(alpha), linestyle=':', color="#555555",
                       linewidth=1.2, label=r'$\lambda_\infty$')

            _style(ax, rf'$\alpha = {alpha:.2f}$', r'$n$',
                   r'$\lambda_{k,\alpha k}^{1/k}$')
            fig.tight_layout()
            _save(fig, plot_dir, f"hessian_bounds_alpha{alpha:.2f}")


def plot_eigenfunctions(pkl_filename, plot_dir="eigenvalue plots"):
    """Eigenfunction overlays and L1 convergence plots, one set per alpha.

    Produces two eigenfunction figures per alpha:
      1. All eigenfunctions as a grey log-scale gradient, with four
         specific n values highlighted in colour.
      2. The four highlighted curves only (no grey background).
    """
    results = _load(pkl_filename)
    if results is None:
        return
    os.makedirs(plot_dir, exist_ok=True)

    with matplotlib.rc_context(PAPER_RC):
        for alpha in sorted(results):
            data = results[alpha]
            n_history = np.array(data['n_history'])
            u_history = np.array(data['u_history'])
            x_standard = data['x_standard']

            r = (x_standard - x_standard[0]) / (x_standard[-1] - x_standard[0])
            u_theory = u0(r, alpha)
            u_theory_norm = u_theory / np.max(np.abs(u_theory))

            grey_cmap, grey_norm = _grey_log_cmap(n_history)
            highlight_map = _build_highlight_map(n_history)

            # Precompute all normalised eigenfunctions and their L1 errors once
            u_norms = {
                idx: u / np.max(np.abs(u))
                for idx, u in enumerate(u_history)
            }
            l1_errors = [
                np.trapezoid(np.abs(u_theory_norm - u_norms[idx]), x=r)
                for idx in range(len(n_history))
            ]

            _plot_eigenfunction_overlay(
                alpha, r, n_history, u_norms, highlight_map,
                u_theory_norm, grey_cmap, grey_norm, plot_dir,
            )
            _plot_eigenfunction_highlights_only(
                alpha, r, u_norms, highlight_map, u_theory_norm, plot_dir,
            )
            _plot_l1_convergence(alpha, n_history, l1_errors, plot_dir)


def plot_residuals(pkl_filename, plot_dir="eigenvalue plots"):
    """Residuals between numerical eigenvalues and each theoretical bound."""
    results = _load(pkl_filename)
    if results is None:
        return
    os.makedirs(plot_dir, exist_ok=True)

    with matplotlib.rc_context(PAPER_RC):
        for alpha in sorted(results):
            data = results[alpha]
            n_vals = np.array(data['n_history'])
            lam_num = np.array(data['lam_history'])

            print(f"Calculating residuals for alpha={alpha}...")
            lower, upper = compute_bounds(alpha, n_vals)

            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            ax.plot(n_vals, lam_num - lower, 'o', color="#c0392b", markersize=3,
                    markeredgewidth=0, zorder=3,
                    label=r'$\lambda_{n/\alpha,\, n}^{\alpha/n} - $ lower bound')
            ax.plot(n_vals, lam_num - upper, 's', color="#2471a3", markersize=3,
                    markeredgewidth=0, zorder=3,
                    label=r'$\lambda_{n/\alpha,\, n}^{\alpha/n} - $ upper bound')
            ax.axhline(0, color="#555555", linewidth=0.8, linestyle='-', zorder=1)

            _style(ax, rf'Residuals, $\alpha = {alpha:.2f}$', r'$n$', r'Residual')
            fig.tight_layout()
            _save(fig, plot_dir, f"hessian_residuals_alpha{alpha:.2f}")


# ---------------------------------------------------------------------------
# Eigenfunction sub-plots
# ---------------------------------------------------------------------------

def _plot_eigenfunction_overlay(alpha, r, n_history, u_norms, highlight_map,
                                 u_theory_norm, cmap, cnorm, plot_dir):
    """All eigenfunctions as a grey gradient with coloured highlights on top."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for idx, u_norm in u_norms.items():
        if idx not in highlight_map:
            n = n_history[idx]
            ax.plot(r, u_norm,
                    color=cmap(cnorm(max(float(n), cnorm.vmin))),
                    linewidth=0.4, alpha=0.7, zorder=2)

    _draw_highlights(ax, r, u_norms, highlight_map, u_theory_norm)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'$n$')
    cbar.ax.tick_params(labelsize=9)

    _style(ax, rf'Eigenfunctions, $\alpha = {alpha:.2f}$',
           r'$r$', r'Normalised $u(r)$')
    fig.tight_layout()
    _save(fig, plot_dir, f"eigenfunctions_alpha{alpha:.2f}")


def _plot_eigenfunction_highlights_only(alpha, r, u_norms, highlight_map,
                                         u_theory_norm, plot_dir):
    """Four highlighted eigenfunctions and the theoretical curve only."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    _draw_highlights(ax, r, u_norms, highlight_map, u_theory_norm)
    _style(ax, rf'Eigenfunctions, $\alpha = {alpha:.2f}$',
           r'$r$', r'Normalised $u(r)$')
    fig.tight_layout()
    _save(fig, plot_dir, f"eigenfunctions_highlights_alpha{alpha:.2f}")


def _plot_l1_convergence(alpha, n_history, l1_errors, plot_dir):
    """Log-log L1 error vs n with a power-law line of best fit."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    ax.plot(n_history, l1_errors, 'o', color='#2471a3',
            markersize=3, markeredgewidth=0)

    log_n = np.log(n_history.astype(float))
    log_e = np.log(np.array(l1_errors, dtype=float))
    coeffs = np.polyfit(log_n, log_e, 1)
    fit_n = np.linspace(n_history.min(), n_history.max(), 200)
    fit_e = np.exp(np.polyval(coeffs, np.log(fit_n)))
    ax.plot(fit_n, fit_e, '--', color='#e74c3c', linewidth=1.5,
            label=rf'Fit: slope $= {coeffs[0]:.2f}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    _style(ax, rf'Convergence to $u_\infty$, $\alpha = {alpha:.2f}$',
           r'$n$', r'$\int_0^1 |u_\infty - u_n|\,\mathrm{d}r$')
    fig.tight_layout()
    _save(fig, plot_dir, f"convergence_alpha{alpha:.2f}")


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def _draw_highlights(ax, r, u_norms, highlight_map, u_theory_norm):
    """Draw the coloured highlight lines and the dashed theoretical curve."""
    for idx in sorted(highlight_map):
        n_actual, color = highlight_map[idx]
        ax.plot(r, u_norms[idx], color=color, linewidth=1.5, zorder=5,
                label=rf'$n = {int(n_actual)}$')
    ax.plot(r, u_theory_norm, linestyle='--', color='black', linewidth=1.5,
            label=r'Theoretical $u_\infty$', zorder=6)


def _build_highlight_map(n_history):
    """Map array index -> (actual n, plasma color) for each highlight value."""
    plasma = matplotlib.colormaps["plasma"]
    colors = [plasma(v) for v in np.linspace(0.0, 0.9, len(HIGHLIGHT_NS))]
    idx_map = {}
    for hn, color in zip(HIGHLIGHT_NS, colors):
        idx = int(np.argmin(np.abs(n_history - hn)))
        idx_map[idx] = (n_history[idx], color)
    return idx_map


def _grey_log_cmap(n_history):
    """Truncated Greys colormap with log normalisation over n_history."""
    base = matplotlib.colormaps["Greys"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'greys_trunc', base(np.linspace(0.15, 0.75, 256))
    )
    norm = matplotlib.colors.LogNorm(
        vmin=max(float(n_history.min()), 1.0),
        vmax=float(n_history.max()),
    )
    return cmap, norm


# ---------------------------------------------------------------------------
# Axis styling and I/O utilities
# ---------------------------------------------------------------------------

def _style(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', frameon=False)
    ax.tick_params(which='both', top=False, right=False)


def _save(fig, plot_dir, stem):
    fig.savefig(os.path.join(plot_dir, f"{stem}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


def _load(pkl_filename):
    try:
        with open(pkl_filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {pkl_filename} not found.")
        return None


def _check_monotone(lam, n_vals, alpha):
    lam_sorted = lam[np.argsort(n_vals)]
    diff = np.diff(lam_sorted)
    if np.all(diff > 0):
        print(f"[alpha={alpha}] Lambdas strictly increasing.")
    elif np.all(diff >= 0):
        print(f"[alpha={alpha}] Lambdas non-decreasing (flat plateaus present).")
    else:
        print(f"WARNING [alpha={alpha}]: Lambdas NOT consistently increasing.")
