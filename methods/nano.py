#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NANO: Locally differeNtially privAte truth discovery via updatiNg
      time stamp determinatiOn  (IEEE TMC 2026)

Zhang et al. "Locally Differentially Private Truth Discovery Over
Data Streams" — IEEE Trans. on Mobile Computing, Vol. 25, No. 6, 2026.

Single-stream adaptation with FAIR budget accounting:
  - Base per-point budget = ε / T  (identical to Naive baseline)
  - CUD monitors window stability using median + scale drift
  - STABLE window: reuse previous block, save ENTIRE base budget
  - CHANGE window: perturb with base + accumulated savings / w_len
  - Total spent ≤ ε  (sequential composition holds)

Three key NANO innovations:
  1. MixTD   — mixed Laplacian (privacy) + Gaussian (inherent noise) perturbation
  2. CUD     — changing-aware detection to skip stable regions
  3. Dynamic budget — accumulated savings released at critical transition points

Paper defaults:  η = 0.001, α = 0.5, w = 7.
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


def _robust_local_stats(data_chunk):
    """
    Robust center and scale for CUD-style change detection.
    """
    arr = np.asarray(data_chunk, dtype=np.float64).reshape(-1)

    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))

    if mad < 1e-8:
        mad = float(np.std(arr))

    if mad < 1e-8:
        mad = max(float(np.mean(np.abs(arr))), 1.0)

    return med, mad


def _cud_is_stable(window_data, prev_median, prev_scale, eta):
    """
    CUD stable-window detection.

    Returns (is_stable, cur_median, cur_scale).

    Two normalised change signals:
      centre_drift = |cur_median - prev_median| / ref_scale
      scale_change = |cur_scale  - prev_scale | / prev_scale

    ref_scale = max(|prev_median|, prev_scale, 1e-8)  — no hard-coded floor
    """
    cur_median, cur_scale = _robust_local_stats(window_data)

    ref = max(abs(prev_median), prev_scale, 1e-8)
    centre_drift = abs(cur_median - prev_median) / ref
    scale_change = abs(cur_scale - prev_scale) / max(prev_scale, 1e-8)

    stable = max(centre_drift, scale_change) <= eta
    return stable, cur_median, cur_scale


def _match_block_length(prev_block, target_len):
    """
    Reuse the previous released block for a stable window.

    If the last window is shorter, crop.
    If the current window is longer than the previous block, repeat the last value.
    """
    prev_block = np.asarray(prev_block, dtype=np.float64).reshape(-1)

    if len(prev_block) == target_len:
        return prev_block.copy()

    if len(prev_block) > target_len:
        return prev_block[:target_len].copy()

    if len(prev_block) == 0:
        return np.zeros(target_len, dtype=np.float64)

    pad_len = target_len - len(prev_block)
    pad = np.full(pad_len, prev_block[-1], dtype=np.float64)
    return np.concatenate([prev_block, pad], axis=0)


def _mixTD_perturb_block(window_data, epsilon_window, sensitivity,
                         mix_alpha=0.5, local_scale=None):
    """
    MixTD perturbation for one window.

    epsilon_window is the TOTAL budget for this window.
    It is divided by the number of points n so that each point gets
    ε_per_point = epsilon_window / n  (fair composition with Naive).

    Laplace  ~ Lap(0, sensitivity / ε_per_point)   ← LDP privacy
    Gaussian ~ N(0, local_scale · α · 0.02)         ← inherent noise model
    """
    x = np.asarray(window_data, dtype=np.float64).reshape(-1)
    n = len(x)
    if n == 0:
        return x.copy()

    eps_window = max(float(epsilon_window), 1e-12)
    eps_per_point = eps_window / float(n)
    eps_per_point = max(eps_per_point, 1e-12)

    lap_scale = float(sensitivity) / eps_per_point

    if local_scale is None:
        _, local_scale = _robust_local_stats(x)

    gauss_scale = max(float(local_scale), float(sensitivity), 1e-8) * float(mix_alpha) * 0.02

    lap_noise = np.random.laplace(0.0, lap_scale, size=n)
    gauss_noise = np.random.normal(0.0, gauss_scale, size=n)

    return x + lap_noise + gauss_noise


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def point_mae(raw_tensor, pub_tensor):
    return torch.mean(torch.abs(raw_tensor - pub_tensor)).item()


@torch.no_grad()
def point_mre(raw_tensor, pub_tensor):
    abs_diff = torch.abs(raw_tensor - pub_tensor)
    denom = torch.abs(raw_tensor)

    mask_zero = denom == 0
    rel = torch.empty_like(abs_diff)

    rel[mask_zero] = torch.abs(pub_tensor)[mask_zero]
    rel[~mask_zero] = abs_diff[~mask_zero] / denom[~mask_zero]

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

    idx_a = torch.tensor([p[0] for p in intervals], device=raw.device, dtype=torch.long)
    idx_b = torch.tensor([p[1] for p in intervals], device=raw.device, dtype=torch.long)

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

    raw_counts_all = []
    pub_counts_all = []

    for d in range(dim):
        raw_col = raw[:, d]
        pub_col = pub[:, d]

        minv = float(torch.min(raw_col).item())
        maxv = float(torch.max(raw_col).item())

        if minv == maxv:
            raw_cnt = torch.full((query_num,), float(T), device=raw.device, dtype=raw.dtype)
            pub_cnt = torch.sum(pub_col == minv).repeat(query_num).to(pub.dtype)
            raw_counts_all.append(raw_cnt.unsqueeze(1))
            pub_counts_all.append(pub_cnt.unsqueeze(1))
            continue

        lows = []
        highs = []

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
        return sum_query_metric(raw_tensor, pub_tensor, query_num=query_num, metric="mae")

    if metric == "sum_mre":
        return sum_query_metric(raw_tensor, pub_tensor, query_num=query_num, metric="mre")

    if metric == "count_mae":
        return count_query_metric(raw_tensor, pub_tensor, query_num=query_num, metric="mae")

    if metric == "count_mre":
        return count_query_metric(raw_tensor, pub_tensor, query_num=query_num, metric="mre")

    raise ValueError(
        "metric must be one of: mae, mre, sum_mae, sum_mre, count_mae, count_mre"
    )


# ============================================================
# NANO-style workflow
# ============================================================

def nano_workflow(epsilon, sensitivity, raw_stream, window_size,
                  eta=0.001, mix_alpha=0.5, return_info=False):
    """
    NANO workflow with fair budget accounting.

    Base per-point budget = ε / T  (same as Naive).
    CUD monitors windows; stable windows save their entire budget slice,
    which is then redistributed to the next change window.

    Key invariants:
      - Every perturbed point ALWAYS gets no more than ε / T on average
      - Change windows get: base + (accumulated savings) / w_len per point
      - Stable windows: 0 budget spent (previous block reused exactly)
      - Total spent ≤ ε (sequential composition)

    Args:
        epsilon     : Total privacy budget.
        sensitivity : Data sensitivity.
        raw_stream  : Shape (T, dim).
        window_size : CUD monitoring window size.
        eta         : Stability threshold.
        mix_alpha   : Gaussian/Lapacian mixing (0.5 per paper).
        return_info : If True, also return (published, info_dict).

    Returns:
        published_list, or (published_list, info)
    """
    raw_np = _to_2d_np(raw_stream)
    T, dim = raw_np.shape

    if T == 0:
        return [] if not return_info else ([], {})

    w = int(window_size)
    if w <= 0:
        raise ValueError("window_size must be positive.")

    epsilon = float(epsilon)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    # Fair baseline: same per-point budget as Naive
    base_per_point = epsilon / float(T)

    published = np.zeros_like(raw_np, dtype=np.float64)

    total_points_perturbed = 0
    total_points_reused = 0
    total_updates = 0
    total_reuses = 0

    for d in range(dim):
        stream_d = raw_np[:, d]

        prev_median = None
        prev_scale = None
        prev_pub_block = None

        saved_budget = 0.0  # accumulated from skipped stable windows

        pos = 0
        while pos < T:
            end = min(pos + w, T)
            window = stream_d[pos:end]
            w_len = len(window)

            if prev_median is None:
                update_required = True
                cur_median, cur_scale = _robust_local_stats(window)
            else:
                stable, cur_median, cur_scale = _cud_is_stable(
                    window, prev_median, prev_scale, eta
                )
                update_required = not stable

            # Base budget this window would get if it updated normally:
            # base_per_point * w_len
            window_base = base_per_point * w_len

            if update_required:
                # Spend base + any accumulated savings, divided across points
                eps_window = window_base + saved_budget
                saved_budget = 0.0

                pub_block = _mixTD_perturb_block(
                    window_data=window,
                    epsilon_window=eps_window,
                    sensitivity=sensitivity,
                    mix_alpha=mix_alpha,
                    local_scale=cur_scale,
                )

                prev_pub_block = pub_block.copy()
                prev_median = cur_median
                prev_scale = cur_scale

                total_updates += 1
                total_points_perturbed += w_len

            else:
                # Stable: save the ENTIRE window budget, reuse previous block
                saved_budget += window_base

                pub_block = _match_block_length(prev_pub_block, w_len)

                total_reuses += 1
                total_points_reused += w_len

            published[pos:end, d] = pub_block
            pos += w

    published_list = published.tolist()

    if return_info:
        info = {
            "T": T,
            "dim": dim,
            "window_size": w,
            "base_per_point": base_per_point,
            "updates": total_updates,
            "reuses": total_reuses,
            "pts_perturbed": total_points_perturbed,
            "pts_reused": total_points_reused,
            "eta": eta,
            "mix_alpha": mix_alpha,
        }
        return published_list, info

    return published_list


# ============================================================
# Runner
# ============================================================

def run_nano_gpu(
    epsilon_list,
    sensitivity,
    raw_stream,
    window_size,
    round_,
    metric="mae",
    query_num=100,
    Flag_=0,
    device=None,
    eta=0.001,
    mix_alpha=0.5,
):
    """
    NANO GPU wrapper — algorithm runs on CPU, metrics on GPU.

    Parameters (aligned with run_naive_gpu):
        epsilon_list : list[float]
        sensitivity  : float
        raw_stream   : list[list] or np.ndarray, shape (T, dim)
        window_size  : int — CUD monitoring window size
        round_       : int
        metric       : str  — "mae", "mre", "sum_mae", "sum_mre", "count_mae", "count_mre"
        query_num    : int
        Flag_        : int  — 0=iterate epsilons, 1=iterate window_sizes
        device       : torch.device or None
        eta          : float — CUD threshold
        mix_alpha    : float — MixTD mixing weight
    """
    if device is None:
        device = get_device()

    results = []

    raw_np = _to_2d_np(raw_stream)
    raw_t_full = torch.tensor(raw_np, device=device, dtype=torch.float32)

    if Flag_ == 0:
        for eps in epsilon_list:
            err_sum = 0.0
            for _ in range(int(round_)):
                pub = nano_workflow(
                    epsilon=float(eps),
                    sensitivity=sensitivity,
                    raw_stream=raw_np,
                    window_size=window_size,
                    eta=eta,
                    mix_alpha=mix_alpha,
                    return_info=False,
                )
                pub_t = torch.tensor(pub, device=device, dtype=torch.float32)
                err_sum += compute_error(raw_t_full, pub_t, metric=metric, query_num=query_num)

            err_avg = err_sum / float(round_)
            print(f"epsilon: {eps} Done! metric={metric}, value={err_avg}")
            results.append(err_avg)

        print("NANO DONE!")

    else:
        if isinstance(epsilon_list, (list, tuple, np.ndarray)):
            eps = float(epsilon_list[0])
        else:
            eps = float(epsilon_list)

        for w in window_size:
            err_sum = 0.0
            for _ in range(int(round_)):
                pub = nano_workflow(
                    epsilon=eps,
                    sensitivity=sensitivity,
                    raw_stream=raw_np,
                    window_size=int(w),
                    eta=eta,
                    mix_alpha=mix_alpha,
                    return_info=False,
                )
                pub_t = torch.tensor(pub, device=device, dtype=torch.float32)
                err_sum += compute_error(raw_t_full, pub_t, metric=metric, query_num=query_num)

            err_avg = err_sum / float(round_)
            print(f"window size: {w} Done! metric={metric}, value={err_avg}")
            results.append(err_avg)

        print("NANO DONE!")

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
        raw_stream = [[float(i % 100)] for i in range(1000)]

    epsilon_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    sensitivity = 1
    window_size = 7
    round_ = 3
    metric = "mae"

    errors = run_nano_gpu(
        epsilon_list=epsilon_list,
        sensitivity=sensitivity,
        raw_stream=raw_stream,
        window_size=window_size,
        round_=round_,
        metric=metric,
        query_num=100,
        Flag_=0,
        device=dev,
        eta=0.001,
        mix_alpha=0.5,
    )

    print("NANO result:", errors)

    # Optional debug: inspect update/reuse statistics.
    pub, info = nano_workflow(
        epsilon=1.0,
        sensitivity=sensitivity,
        raw_stream=raw_stream,
        window_size=window_size,
        eta=0.001,
        mix_alpha=0.5,
        return_info=True,
    )

    print("NANO info:", info)