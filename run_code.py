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
9. DSAT
10. Fast
11. SPPA
12. NANO

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

# 01 real-world datasets
# "covid19", "Flu_Deaths", "unemp", "ilinet", "footmart", "nation", "tdv", "retail"

# 02 synthetic_datasets
# "high_volatility", "low_volatility", "distribution_drift", "periodic_switch", "sparse_spike", "correlated_latent_factor"

# ----------------------------
datasets = [
    # 01 real-world datasets
    "taxi",
    "covid19", 
    "energy",
    "Flu_Deaths", 
    "unemp", 
    # "ilinet", 
    # "footmart", 
    # "nation", 
    "tdv", 
    # "retail",
    # 02 synthetic_datasets
    "high_volatility", "low_volatility", "distribution_drift", "periodic_switch", "sparse_spike", "correlated_latent_factor"
]

epsilon_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

sensitivity_s = 1
sensitivity_p = 1
window_size = 120
windownum_warm = 1
windownum_updateE = 2
windownum_updateQ = 2
round_ = 3
metric = "mae"

METHODS = [
    # "Mirror-Smoothing",
    # "SPAS",
    # "Naive",
    # "BucOrder",
    # "CompOrder",
    # "DPI",
    # "PeGaSus",
    # "AdaPub",
    "DSAT",
    "Fast",
    "SPPA",
    "NANO",
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
# Method registry
# Each active method must define:
#   import_module, import_name, run_kwargs (mapping of kwargs loaded after import)
# Edit only the METHODS list above to toggle methods on/off.
# ----------------------------
METHOD_REGISTRY = {
    "Mirror-Smoothing": {
        "module": "methods.mirror_smoothing",
        "function": "run_mirror_smoothing_gpu",
        "kwargs": dict(
            epsilon_list=epsilon_list,
            delta_s=sensitivity_s,
            delta_p=sensitivity_p,
            raw_stream=None,          # placeholder, filled per dataset
            window_size=window_size,
            windownum_warm=windownum_warm,
            windownum_updateQ=windownum_updateQ,
            rounds=round_,
            beta0=0.7,
            gamma=0.7,
            metric=metric,
            device=device,
        ),
    },
    "SPAS": {
        "module": "methods.SPAS",
        "function": "run_SPAS_gpu",
        "kwargs": dict(
            epsilon_list=epsilon_list,
            sensitivity_s=sensitivity_s,
            sensitivity_p=sensitivity_p,
            raw_stream=None,
            window_size=window_size,
            windownum_warm=windownum_warm,
            windownum_updateE=windownum_updateE,
            round_=round_,
            metric=metric,
            device=device,
        ),
    },
    "Naive": {
        "module": "methods.Naive",
        "function": "run_naive_gpu",
        "kwargs": dict(
            epsilon_list=epsilon_list,
            sensitivity=sensitivity_p,
            raw_stream=None,
            window_size=window_size,
            round_=round_,
            metric=metric,
            device=device,
        ),
    },
    "BucOrder": {
        "module": "methods.BucOrder",
        "function": "run_bucorder_gpu",
        "kwargs": dict(
            epsilon_list=epsilon_list,
            sensitivity=sensitivity_p,
            raw_stream=None,
            delay_time=window_size,
            buc_size=100,
            round_=round_,
            metric=metric,
            device=device,
        ),
    },
    "CompOrder": {
        "module": "methods.CompOrder",
        "function": "run_comporder_gpu",
        "kwargs": dict(
            epsilon_list=epsilon_list,
            sensitivity=sensitivity_p,
            raw_stream=None,
            delay_time=10,
            round_=round_,
            metric=metric,
            device=device,
            verbose=False,
        ),
    },
    "DPI": {
        "module": "methods.DPI",
        "function": "run_dpi_gpu",
        "kwargs": dict(
            epsilon_list=epsilon_list,
            raw_stream=None,
            round_=round_,
            metric=metric,
            device=device,
            verbose=False,
        ),
    },
    "PeGaSus": {
        "module": "methods.PeGaSus",
        "function": "run_pegasus_gpu",
        "kwargs": dict(
            epsilon_list=epsilon_list,
            sensitivity=sensitivity_p,
            raw_stream=None,
            window_size=window_size,
            round_=round_,
            metric=metric,
            device=device,
        ),
    },
    "AdaPub": {
        "module": "methods.AdaPub",
        "function": "run_adapub_gpu",
        "kwargs": dict(
            epsilon_list=epsilon_list,
            sensitivity=sensitivity_p,
            raw_stream=None,
            window_size=window_size,
            round_=round_,
            metric=metric,
            device=device,
        ),
    },
    "DSAT": {
        "module": "methods.dsat",
        "function": "run_dsat_gpu",
        "kwargs": dict(
            epsilon_list=epsilon_list,
            sensitivity=sensitivity_p,
            raw_stream=None,
            window_size=window_size,
            round_=round_,
            metric=metric,
            device=device,
        ),
    },
    "Fast": {
        "module": "methods.fast_w_event",
        "function": "run_fast_gpu",
        "kwargs": dict(
            epsilon_list=epsilon_list,
            sensitivity=sensitivity_p,
            raw_stream=None,
            window_size=window_size,
            round_=round_,
            metric=metric,
            device=device,
        ),
    },
    "SPPA": {
        "module": "methods.sppa",
        "function": "run_sppa_gpu",
        "kwargs": dict(
            epsilon_list=epsilon_list,
            sensitivity=sensitivity_p,
            raw_stream=None,
            window_size=window_size,
            round_=round_,
            metric=metric,
            device=device,
            tau=1.0,          # time sensitivity τ = Ts (paper Sec 5.2)
            Ts=1.0,            # sampling period
        ),
    },
    "NANO": {
        "module": "methods.nano",
        "function": "run_nano_gpu",
        "kwargs": dict(
            epsilon_list=epsilon_list,
            sensitivity=sensitivity_p,
            raw_stream=None,
            window_size=window_size,
            round_=round_,
            metric=metric,
            device=device,
            eta=0.001,         # CUD weight-fluctuation threshold (paper: 0.001)
            mix_alpha=0.5,     # MixTD noise mixing parameter (paper: 0.5)
        ),
    },
}


def _resolve_method(method_name):
    """Lazy-import a method function from the registry. Returns (fn, kwargs_template)."""
    import importlib

    info = METHOD_REGISTRY[method_name]
    mod = importlib.import_module(info["module"])
    fn = getattr(mod, info["function"])
    return fn, info["kwargs"]


# ----------------------------
# Main loop
# ----------------------------
total_methods = len(METHODS)

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
        print(f"❌ Failed to read dataset '{ds}': {e}")
        continue

    # Run only the methods listed in METHODS, in order
    results_map = {}  # method_name -> error_list (or NaN list on failure)

    for idx, method_name in enumerate(METHODS, start=1):
        if method_name not in METHOD_REGISTRY:
            print(f"  [{idx}/{total_methods}] {method_name}... ❌ Unknown method — not in registry")
            results_map[method_name] = [float("nan")] * len(epsilon_list)
            continue

        print(f"  [{idx}/{total_methods}] {method_name}...", end="", flush=True)

        fn, kwargs_template = _resolve_method(method_name)
        kwargs = {**kwargs_template, "raw_stream": raw_stream}

        err_list, elapsed, status = run_method_safely(method_name, fn, **kwargs)
        results_map[method_name] = err_list

        print(f" ✅ ({elapsed:.1f}s)" if status == "Success" else f" ❌ {status}")

    # ----------------------------
    # Save results in wide format (one row per epsilon)
    # Columns are built dynamically from METHODS order
    # ----------------------------
    results_data = {
        "dataset": [ds] * len(epsilon_list),
        "epsilon": epsilon_list,
        "window_size": [window_size] * len(epsilon_list),
    }
    for method_name in METHODS:
        results_data[method_name] = results_map[method_name]

    results_df = pd.DataFrame(results_data)

    dataset_dir = os.path.join(output_dir, ds)
    os.makedirs(dataset_dir, exist_ok=True)
    csv_path = os.path.join(dataset_dir, f"results_{ds}.csv")
    results_df.to_csv(csv_path, index=False)

    print(f"  ✅ Results saved: {csv_path}")

print(f"\n{'='*80}")
print(f"All tests finished. Results saved under: {output_dir}")
print(f"{'='*80}\n")
