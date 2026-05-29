import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # must be set before pyplot is imported anywhere

from continuation import generate_and_save_data
from plotting import plot_numerical_vs_bounds, plot_residuals, plot_eigenfunctions

if __name__ == "__main__":
    FILENAME = "data_alpha_multi.pkl"
    PLOT_DIR = "eigenvalue plots"

    generate_and_save_data(
        n_max=100,
        alphas=np.array([1.5, 2.0, 3.0]),
        output_filename=FILENAME,
        min_step=0.1,
        verbose=True,
        debug=False,
        plot_failures=False,
    )

    plot_numerical_vs_bounds(FILENAME, PLOT_DIR)
    plot_residuals(FILENAME, PLOT_DIR)
    plot_eigenfunctions(FILENAME, PLOT_DIR)
