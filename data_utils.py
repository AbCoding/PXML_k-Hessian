import os
import glob
import numpy as np

# Define standard paths
BASE_DIR = "run_data"
SINGLE_RUNS_DIR = os.path.join(BASE_DIR, "single_runs")
ASYMPTOTES_DIR = os.path.join(BASE_DIR, "asymptotic_data")


def ensure_directories():
    """Creates the necessary folder structure if it doesn't exist."""
    os.makedirs(SINGLE_RUNS_DIR, exist_ok=True)
    os.makedirs(ASYMPTOTES_DIR, exist_ok=True)
    return SINGLE_RUNS_DIR, ASYMPTOTES_DIR


def save_individual_run(n_data, lam_data, divisor):
    """Saves a single (n, lambda) dataset to the single_runs folder."""
    if len(n_data) == 0:
        return None

    # Only using end_n as requested
    end_n = int(n_data[-1])
    ratio_str = f"{divisor:.2f}"

    # New filename format: individual_run_RATIO_ENDN.npz
    filename = f"individual_run_{ratio_str}_{end_n}.npz"
    filepath = os.path.join(SINGLE_RUNS_DIR, filename)

    np.savez(filepath, n=n_data, lambda_val=lam_data, divisor=divisor)
    return filepath


def save_asymptotic_data(divisors, asymptotes, n_max):
    """Saves the aggregated asymptote data to the asymptotic_data folder."""
    if len(divisors) == 0:
        return None

    start_ratio = divisors[0]
    end_ratio = divisors[-1]

    filename = f"asymptote_data_{start_ratio:.1f}_to_{end_ratio:.1f}_maxN{n_max}.npz"
    filepath = os.path.join(ASYMPTOTES_DIR, filename)

    np.savez(filepath, divisors=divisors, asymptotes=asymptotes)
    return filepath


def load_all_single_runs():
    """Loads all .npz files from the single_runs directory."""
    files = glob.glob(os.path.join(SINGLE_RUNS_DIR, "*.npz"))
    loaded_data = []

    for f in files:
        try:
            data = np.load(f)
            loaded_data.append({
                'divisor': float(data['divisor']),
                'n': data['n'],
                'lambda_val': data['lambda_val']
            })
        except Exception as e:
            print(f"Error loading {f}: {e}")

    # Sort by divisor for cleaner plotting
    loaded_data.sort(key=lambda x: x['divisor'])
    return loaded_data


def load_latest_asymptote_data():
    """Loads the most recent asymptote data file."""
    files = glob.glob(os.path.join(ASYMPTOTES_DIR, "*.npz"))
    if not files:
        return None

    # Sort by modification time to get the latest
    latest_file = max(files, key=os.path.getmtime)
    try:
        data = np.load(latest_file)
        return {
            'divisors': data['divisors'],
            'asymptotes': data['asymptotes']
        }
    except Exception as e:
        print(f"Error loading {latest_file}: {e}")
        return None