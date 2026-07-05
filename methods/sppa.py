#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SPPA: Sampling Period Perturbation Algorithm

Wang et al., "Local Differentially Private Release of Infinite Streams
With Temporal Relevance", WWW 2025.

Budget model — ALIGNED with other run_code.py baselines:
  Total ε is divided across all T points:  base = ε / T
  Each sliding window of n_perseg points gets:  ε_window = base * n_perseg
  SWP uses ε_window to perturb the sampling period via Lap(Ts, τ/ε_window).

  → Total spent = Σ ε_window = ε  (fair composition)
  → Published output is T points (full stream), so MAE is computed on the
    same length as Naive / PeGaSus / etc.

Algorithm (SPPA, Algorithm 2):
  - Sliding window:  n_perseg = w + 2,  step = w,  release = middle w points
  - First point dropped (no privacy: d'_0 ≡ d_0)
  - Last point dropped  (DFT periodic extension causes large error)
  - Unpublished positions (t=0, window boundaries, trailing tail) are
    filled with Naive-style Laplace perturbation at the base per-point budget.

SWP (Algorithm 1):
  1. Perturb Ts' ~ Lap(Ts, τ/ε_window)
  2. DFT → Fourier interpolation at perturbed timestamps
  3. Release middle n-2 points
"""

import math
import random
import numpy as np
import torch

try:
    from methods.data_process import data_reader
except Exception:
    data_reader = None


# ============================================================
# Device and seed
# ============================================================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Utility helpers
# ============================================================

def _to_2d_np(raw_stream):
    arr = np.asarray(raw_stream, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("raw_stream must be 1D or 2D.")
    return arr


# ============================================================
# GPU metric functions (same as other baselines)
# ============================================================

@torch.no_grad()
def point_mae(raw_tensor, pub_tensor):
    return torch.mean(torch.abs(raw_tensor - pub_tensor)).item()


@torch.no_grad()
def point_mre(raw_tensor, pub_tensor):
    abs_diff = torch.abs(raw_tensor - pub_tensor)
    denom = torch.abs(raw_tensor)
    mask0 = denom == 0
    rel = torch.empty_like(abs_diff)
    rel[mask0] = torch.abs(pub_tensor)[mask0]
    rel[~mask0] = abs_diff[~mask0] / denom[~mask0]
    return torch.mean(rel).item()


@torch.no_grad()
def sum_query_metric(raw, pub, query_num=100, metric="mae"):
    T, dim = raw.shape
    if query_num <= 0:
        return 0.0
    if T < 2:
        return point_mae(raw, pub) if metric == "mae" else point_mre(raw, pub)
    intervals = []
    while len(intervals) < query_num:
        a = random.randint(0, T - 1)
        b = random.randint(0, T - 1)
        if a == b:
            continue
        if a > b:
            a, b = b, a
        intervals.append((a, b))
    idx_a = torch.tensor([x[0] for x in intervals], device=raw.device, dtype=torch.long)
    idx_b = torch.tensor([x[1] for x in intervals], device=raw.device, dtype=torch.long)
    raw_ps = torch.zeros((T + 1, dim), device=raw.device, dtype=raw.dtype)
    pub_ps = torch.zeros((T + 1, dim), device=pub.device, dtype=pub.dtype)
    raw_ps[1:] = torch.cumsum(raw, dim=0)
    pub_ps[1:] = torch.cumsum(pub, dim=0)
    raw_sum = raw_ps[idx_b] - raw_ps[idx_a]
    pub_sum = pub_ps[idx_b] - pub_ps[idx_a]
    if metric == "mae":
        return torch.mean(torch.abs(raw_sum - pub_sum)).item()
    if metric == "mre":
        return point_mre(raw_sum, pub_sum)
    raise ValueError("metric must be mae or mre.")


@torch.no_grad()
def count_query_metric(raw, pub, query_num=100, metric="mae"):
    T, dim = raw.shape
    if query_num <= 0:
        return 0.0
    raw_counts_all, pub_counts_all = [], []
    for d in range(dim):
        raw_col, pub_col = raw[:, d], pub[:, d]
        minv = float(torch.min(raw_col).item())
        maxv = float(torch.max(raw_col).item())
        if minv == maxv:
            raw_cnt = torch.full((query_num,), float(T), device=raw.device, dtype=raw.dtype)
            pub_cnt = torch.sum(pub_col == minv).repeat(query_num).to(pub.dtype)
            raw_counts_all.append(raw_cnt.unsqueeze(1))
            pub_counts_all.append(pub_cnt.unsqueeze(1))
            continue
        lows, highs = [], []
        while len(lows) < query_num:
            a = random.uniform(minv, maxv)
            b = random.uniform(minv, maxv)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            lows.append(a)
            highs.append(b)
        low = torch.tensor(lows, device=raw.device, dtype=raw.dtype)
        high = torch.tensor(highs, device=raw.device, dtype=raw.dtype)
        raw_in = (raw_col.unsqueeze(1) >= low.unsqueeze(0)) & (raw_col.unsqueeze(1) < high.unsqueeze(0))
        pub_in = (pub_col.unsqueeze(1) >= low.unsqueeze(0)) & (pub_col.unsqueeze(1) < high.unsqueeze(0))
        raw_cnt = torch.sum(raw_in, dim=0).to(raw.dtype)
        pub_cnt = torch.sum(pub_in, dim=0).to(pub.dtype)
        raw_counts_all.append(raw_cnt.unsqueeze(1))
        pub_counts_all.append(pub_cnt.unsqueeze(1))
    raw_q = torch.cat(raw_counts_all, dim=1)
    pub_q = torch.cat(pub_counts_all, dim=1)
    if metric == "mae":
        return torch.mean(torch.abs(raw_q - pub_q)).item()
    if metric == "mre":
        return point_mre(raw_q, pub_q)
    raise ValueError("metric must be mae or mre.")


@torch.no_grad()
def compute_error(raw_tensor, pub_tensor, metric="mae", query_num=100):
    metric = metric.lower().strip()
    if metric == "mae":
        return point_mae(raw_tensor, pub_tensor)
    if metric == "mre":
        return point_mre(raw_tensor, pub_tensor)
    if metric == "sum_mae":
        return sum_query_metric(raw_tensor, pub_tensor, query_num, metric="mae")
    if metric == "sum_mre":
        return sum_query_metric(raw_tensor, pub_tensor, query_num, metric="mre")
    if metric == "count_mae":
        return count_query_metric(raw_tensor, pub_tensor, query_num, metric="mae")
    if metric == "count_mre":
        return count_query_metric(raw_tensor, pub_tensor, query_num, metric="mre")
    raise ValueError("metric must be one of: mae, mre, sum_mae, sum_mre, count_mae, count_mre")


# ============================================================
# SPPA core algorithm
# ============================================================

def swp_single_window(data, epsilon_window, tau=1.0, Ts=1.0):
    """
    SWP: Single Window Perturbator (Algorithm 1).

    Perturb the sampling period using the window's privacy budget,
    then resample the Fourier interpolation function.

    Args:
        data:
            np.ndarray of length n = w + 2.
        epsilon_window:
            Privacy budget for THIS window.
        tau:
            Time sensitivity.
        Ts:
            True sampling period.

    Returns:
        np.ndarray of length n-2  (the middle released points).
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)

    if n < 3:
        return np.array([], dtype=np.float64)

    eps = max(float(epsilon_window), 1e-12)

    # Step 1: perturb the sampling period  Ts' ~ Lap(Ts, τ/ε_window)
    scale = float(tau) / eps
    Ts_perturbed = float(Ts) + np.random.laplace(0.0, scale)
    if Ts_perturbed <= 1e-6:
        Ts_perturbed = 1e-6

    ratio = Ts_perturbed / float(Ts)

    # Step 2: DFT
    f = np.fft.fft(data)

    # Step 3: Fourier interpolation at perturbed timestamps (i = 1..n-2)
    k = np.arange(n, dtype=np.float64)
    i = np.arange(1, n - 1, dtype=np.float64)
    phase = (2j * np.pi / n) * np.outer(i, k) * ratio
    M = np.exp(phase) / n

    return np.real(M @ f)


def sppa_workflow(epsilon, sensitivity, raw_stream, window_size,
                  tau=1.0, Ts=1.0):
    """
    SPPA over full stream — returns T-point published list (fair budget).

    Budget (fair, matches Naive):
      Omega = ceil(T / w)           total number of windows the stream spans
      eps_window = ε / Omega        budget per window
      SWP uses eps_window for the sampling-period perturbation.
      Fill points use the SAME eps_window budget.

      → Total spent = Omega × eps_window = ε  (exact)

      Effective per-point budget ≈ ε/(T + small tail) ≈ ε/T,
      same order of magnitude as Naive.

    Output:
      published : list[list]  — shape (T, dim), every position filled.
        - SPPA-covered positions use Fourier-interpolated values
        - Unpublished positions use naive Laplace at eps_window budget

    Args:
        epsilon     : Total privacy budget.
        sensitivity : Data sensitivity (used for Laplace fill).
        raw_stream  : Shape (T, dim) or (T,).
        window_size : int w  (n_perseg = w+2, release w per window).
        tau         : Time sensitivity.
        Ts          : True sampling period.
    """
    raw_np = _to_2d_np(raw_stream)
    T, dim = raw_np.shape

    if T == 0:
        return []

    w = int(window_size)
    n_perseg = w + 2
    if w <= 0:
        raise ValueError("window_size must be positive.")

    epsilon = float(epsilon)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    # ---- Fair budget: divide ε by number of windows ----
    Omega = max(1, int(math.ceil(float(T) / float(w))))
    eps_window = epsilon / float(Omega)

    # ---- Output ----
    published = np.full((T, dim), np.nan, dtype=np.float64)

    pos = 0
    while pos + n_perseg <= T:
        window = raw_np[pos:pos + n_perseg, :]

        for d in range(dim):
            perturbed = swp_single_window(
                data=window[:, d],
                epsilon_window=eps_window,
                tau=tau,
                Ts=Ts,
            )
            # perturbed length = w, placed at pos+1 .. pos+w
            published[pos + 1:pos + 1 + w, d] = perturbed

        pos += w

    # ---- Fill unpublished positions with Laplace (same eps_window budget) ----
    lap_scale = float(sensitivity) / max(eps_window, 1e-12)

    for d in range(dim):
        col = published[:, d]
        nan_mask = np.isnan(col)
        n_nan = int(np.sum(nan_mask))
        if n_nan > 0:
            nan_fill = raw_np[nan_mask, d] + np.random.laplace(0.0, lap_scale, size=n_nan)
            published[nan_mask, d] = nan_fill

    return published.tolist()


# ============================================================
# Runner (adheres to run_code.py interface)
# ============================================================

def run_sppa_gpu(
    epsilon_list,
    sensitivity,
    raw_stream,
    window_size,
    round_,
    metric="mae",
    query_num=100,
    Flag_=0,
    device=None,
    tau=1.0,
    Ts=1.0,
):
    """
    SPPA GPU runner — algorithm on CPU, metrics on GPU.

    Parameters (aligned with run_naive_gpu):
        epsilon_list : list[float]
        sensitivity  : float
        raw_stream   : list[list] or np.ndarray, shape (T, dim)
        window_size  : int — window size w (n_perseg = w+2)
        round_       : int
        metric       : str
        query_num    : int
        Flag_        : int
        device       : torch.device or None
        tau          : float — time sensitivity
        Ts           : float — sampling period
    """
    if device is None:
        device = get_device()

    results = []

    if Flag_ == 0:
        for eps in epsilon_list:
            err_sum = 0.0
            for _ in range(int(round_)):
                pub = sppa_workflow(
                    epsilon=float(eps),
                    sensitivity=sensitivity,
                    raw_stream=raw_stream,
                    window_size=int(window_size),
                    tau=tau,
                    Ts=Ts,
                )
                raw_t = torch.tensor(raw_stream, device=device, dtype=torch.float32)
                pub_t = torch.tensor(pub, device=device, dtype=torch.float32)
                err_sum += compute_error(raw_t, pub_t, metric=metric, query_num=query_num)

            err_avg = err_sum / float(round_)
            print(f"epsilon: {eps} Done! metric={metric}, value={err_avg}")
            results.append(err_avg)

        print("SPPA DONE!")
    else:
        if isinstance(epsilon_list, (list, tuple, np.ndarray)):
            eps = float(epsilon_list[0])
        else:
            eps = float(epsilon_list)

        for w in window_size:
            err_sum = 0.0
            for _ in range(int(round_)):
                pub = sppa_workflow(
                    epsilon=eps,
                    sensitivity=sensitivity,
                    raw_stream=raw_stream,
                    window_size=int(w),
                    tau=tau,
                    Ts=Ts,
                )
                raw_t = torch.tensor(raw_stream, device=device, dtype=torch.float32)
                pub_t = torch.tensor(pub, device=device, dtype=torch.float32)
                err_sum += compute_error(raw_t, pub_t, metric=metric, query_num=query_num)

            err_avg = err_sum / float(round_)
            print(f"window size: {w} Done! metric={metric}, value={err_avg}")
            results.append(err_avg)

        print("SPPA DONE!")

    return results


# ============================================================
# Local test
# ============================================================

if __name__ == "__main__":
    set_seed(42)
    dev = get_device()
    print("Using device:", dev)

    if data_reader is not None:
        raw_stream = data_reader("Uem")
    else:
        raw_stream = [[float(i % 100)] for i in range(200)]

    epsilon_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    sensitivity = 1
    window_size = 8
    round_ = 3
    metric = "mae"

    error_list = run_sppa_gpu(
        epsilon_list=epsilon_list,
        sensitivity=sensitivity,
        raw_stream=raw_stream,
        window_size=window_size,
        round_=round_,
        metric=metric,
        device=dev,
    )

    print("SPPA result:", error_list)
