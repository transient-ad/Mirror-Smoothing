#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Comparison Test: Performance comparison across multiple epsilon values.

Testing Methods:
1. Mirror-Smoothing (Our proposed method)
2. SPAS
3. Naive (formerly Uniform)
4. BucOrder
5. CompOrder
6. DPI
7. PeGaSus
8. AdaPub

Testing multiple epsilon values: 0.01, 0.1, 0.2, ..., 1.0
Results are saved to CSV files in a "wide" format:
  dataset, epsilon, window_size, <method columns...>

Output layout:
  ./output/<yy.mm.dd>/<dataset_name>/results_ws<window_size>.csv
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from methods.data_process import data_reader


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ----------------------------
# Silence stdout (robust)
# ----------------------------
class _NullWriter:
    """A minimal stdout sink to silence verbose methods."""

    def write(self, _):
        pass

    def flush(self):
        pass


class silence_stdout:
    """
    Context manager to temporarily silence stdout safely.
    Always restores stdout even if exceptions happen.
    """

    def __enter__(self):
        self._old = sys.stdout
        sys.stdout = _NullWriter()
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._old
        # Do not suppress exceptions here
        return False


def run_method_safely(method_name: str, fn, *args, **kwargs):
    """
    Run a method with stdout suppressed.
    Returns:
      (error_list, elapsed_seconds, status_string)

    If it fails, returns (list_of_nan, None, "Error: ...").
    """
    start = time.time()
    try:
        with silence_stdout():
            out = fn(*args, **kwargs)
        elapsed = time.time() - start
        return out, elapsed, "Success"
    except Exception as e:
        # Return NaNs with the correct length if epsilon_list is provided
        eps_list = kwargs.get("epsilon_list", None)
        n = len(eps_list) if isinstance(eps_list, (list, tuple)) else 1
        return [float("nan")] * n, None, f"Error: {str(e)[:200]}"


# ----------------------------
# Benchmark configuration

# ----------------------------
datasets = [
    # 01 real-world datasets
    "covid19",
    "flu_deaths",
    "unemployment",
    "cab"
    "tdrive",
    "energy",
    # 02 synthetic_datasets
    "high_volatility",
    "low_volatility",
    "distribution_drift",
    "periodic_switch",
    "sparse_spike",
    "correlated_latent_factor"
]

epsilon_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

sensitivity_s = 1
sensitivity_p = 1
window_size = 120
windownum_warm = 1
windownum_updateE = 2
windownum_updateQ = 2
round_ = 3

# -----------------You can choose MAE or MRE by commenting on the following 2 lines---------
# metric = "mre"
metric = "mae"

METHODS = [
    "Mirror-Smoothing",
    "SPAS",
    "Naive",
    "BucOrder",
    "CompOrder",
    "DPI",
    "PeGaSus",
    "AdaPub",
]

# Create output directory: ./output/<yy.mm.dd>/
date_str = datetime.now().strftime("%y.%m.%d")
output_dir = os.path.join("./output_{}".format(metric), date_str)
os.makedirs(output_dir, exist_ok=True)

print(f"Output root: {output_dir}")
print(f"Datasets: {len(datasets)}")
print(f"Epsilons: {epsilon_list}")
print(f"Methods per dataset: {len(METHODS)}")
print("-" * 80)


# ----------------------------
# Main loop
# ----------------------------
for ds_idx, ds in enumerate(datasets, start=1):
    print(f"\n{'='*80}")
    print(f"Dataset [{ds_idx}/{len(datasets)}]: {ds}")
    print(f"{'='*80}")

    # Load dataset
    try:
        raw_stream = data_reader(ds)
        data_length = len(raw_stream)
        data_dim = len(raw_stream[0]) if raw_stream else 0
        print(f"Data length: {data_length}, Dimension: {data_dim}")
    except Exception as e:
        print(f"Error: Failed to read dataset '{ds}': {e}")
        continue

    # Import methods once per dataset
    from methods.mirror_smoothing import run_mirror_smoothing_gpu
    from methods.SPAS import run_SPAS_gpu
    from methods.Naive import run_naive_gpu
    from methods.BucOrder import run_bucorder_gpu
    from methods.CompOrder import run_comporder_gpu
    from methods.DPI import run_dpi_gpu
    from methods.PeGaSus import run_pegasus_gpu
    from methods.AdaPub import run_adapub_gpu

    # 1) Mirror-Smoothing
    print("  [1/8] Mirror-Smoothing...", end="", flush=True)
    err_mirror, t_mirror, st_mirror = run_method_safely(
        "Mirror-Smoothing",
        run_mirror_smoothing_gpu,
        epsilon_list=epsilon_list,
        delta_s=sensitivity_s,
        delta_p=sensitivity_p,
        raw_stream=raw_stream,
        window_size=window_size,
        windownum_warm=windownum_warm,
        windownum_updateQ=windownum_updateQ,
        rounds=round_,
        beta0=0.7,
        gamma=0.7,
        metric=metric,
        device=device,
    )
    print(f"OK! ({t_mirror:.1f}s)" if st_mirror ==
          "Success" else f" ❌ {st_mirror}")

    # 2) SPAS
    print("  [2/8] SPAS...", end="", flush=True)
    err_spas, t_spas, st_spas = run_method_safely(
        "SPAS",
        run_SPAS_gpu,
        epsilon_list=epsilon_list,
        sensitivity_s=sensitivity_s,
        sensitivity_p=sensitivity_p,
        raw_stream=raw_stream,
        window_size=window_size,
        windownum_warm=windownum_warm,
        windownum_updateE=windownum_updateE,
        round_=round_,
        metric=metric,
        device=device,
    )
    print(f" OK! ({t_spas:.1f}s)" if st_spas == "Success" else f" ❌ {st_spas}")

    # 3) Naive
    print("  [3/8] Naive...", end="", flush=True)
    err_naive, t_naive, st_naive = run_method_safely(
        "Naive",
        run_naive_gpu,
        epsilon_list=epsilon_list,
        sensitivity=sensitivity_p,
        raw_stream=raw_stream,
        window_size=window_size,
        round_=round_,
        metric=metric,
        device=device,
    )
    print(f" OK! ({t_naive:.1f}s)" if st_naive ==
          "Success" else f" ❌ {st_naive}")

    # 4) BucOrder
    print("  [4/8] BucOrder...", end="", flush=True)
    err_buc, t_buc, st_buc = run_method_safely(
        "BucOrder",
        run_bucorder_gpu,
        epsilon_list=epsilon_list,
        sensitivity=sensitivity_p,
        raw_stream=raw_stream,
        delay_time=window_size,
        buc_size=100,
        round_=round_,
        metric=metric,
        device=device,
    )
    print(f" OK! ({t_buc:.1f}s)" if st_buc == "Success" else f" ❌ {st_buc}")

    # 5) CompOrder
    print("  [5/8] CompOrder...", end="", flush=True)
    err_comp, t_comp, st_comp = run_method_safely(
        "CompOrder",
        run_comporder_gpu,
        epsilon_list=epsilon_list,
        sensitivity=sensitivity_p,
        raw_stream=raw_stream,
        delay_time=10,
        round_=round_,
        metric=metric,
        device=device,
        verbose=False,
    )
    print(f" OK! ({t_comp:.1f}s)" if st_comp == "Success" else f" ❌ {st_comp}")

    # 6) DPI
    print("  [6/8] DPI...", end="", flush=True)
    err_dpi, t_dpi, st_dpi = run_method_safely(
        "DPI",
        run_dpi_gpu,
        epsilon_list=epsilon_list,
        raw_stream=raw_stream,
        round_=round_,
        metric=metric,
        device=device,
        verbose=False,
    )
    print(f" OK! ({t_dpi:.1f}s)" if st_dpi == "Success" else f" ❌ {st_dpi}")

    # 7) PeGaSus
    print("  [7/8] PeGaSus...", end="", flush=True)
    err_pg, t_pg, st_pg = run_method_safely(
        "PeGaSus",
        run_pegasus_gpu,
        epsilon_list=epsilon_list,
        sensitivity=sensitivity_p,
        raw_stream=raw_stream,
        window_size=window_size,
        round_=round_,
        metric=metric,
        device=device,
    )
    print(f" OK! ({t_pg:.1f}s)" if st_pg == "Success" else f" ❌ {st_pg}")

    # 8) AdaPub
    print("  [8/8] AdaPub...", end="", flush=True)
    err_ap, t_ap, st_ap = run_method_safely(
        "AdaPub",
        run_adapub_gpu,
        epsilon_list=epsilon_list,
        sensitivity=sensitivity_p,
        raw_stream=raw_stream,
        window_size=window_size,
        round_=round_,
        metric=metric,
        device=device,
    )
    print(f" OK! ({t_ap:.1f}s)" if st_ap == "Success" else f" ❌ {st_ap}")

    # ----------------------------
    # Save results in wide format (one row per epsilon)
    # ----------------------------
    results_df = pd.DataFrame({
        "dataset": [ds] * len(epsilon_list),
        "epsilon": epsilon_list,
        "window_size": [window_size] * len(epsilon_list),
        "Mirror-Smoothing": err_mirror,
        "SPAS": err_spas,
        "Naive": err_naive,
        "BucOrder": err_buc,
        "CompOrder": err_comp,
        "DPI": err_dpi,
        "PeGaSus": err_pg,
        "AdaPub": err_ap,
    })

    dataset_dir = os.path.join(output_dir, ds)
    os.makedirs(dataset_dir, exist_ok=True)
    csv_path = os.path.join(dataset_dir, f"results_{ds}.csv")
    results_df.to_csv(csv_path, index=False)

    print(f"  OK! Results saved: {csv_path}")

print(f"\n{'='*80}")
print(f"All tests finished. Results saved under: {output_dir}")
print(f"{'='*80}\n")
