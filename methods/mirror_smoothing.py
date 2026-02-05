#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mirror-Smoothing: Paper-aligned implementation (open-source ready)

Three coupled mechanisms:

1) Sample-Mirror:
   - ΔR = |R_curr - R_prev|, ΔN = |N_curr - N_prev|
   - D = Var(ΔR) - Var(ΔN)
   - Q_curr = max{(ε_p / Δ_p) · sqrt(D), 1}
   - Q = γ · Q_prev + (1-γ) · Q_curr   (EMA smoothing)

2) Budget-Mirror:
   - ε_remain = ε_p - Σ ε_{c,t}   (within the current sliding window)
   - svt_remain = ε_{s2} - Σ svt_{c,t} (within the current sliding window)
   - ε_{c,t} = ε_remain / (w - i)
   - svt_{c,t} = svt_remain / Q

3) Noise-Mirror:
   - β = β_min + (β_max - β_min) · (1 - ε_remain / (2·Δ_p))
   - N_t = β·N_curr + (1-β)·N_prev
"""

import math
import numpy as np
import torch
from methods.data_process import data_reader


# ----------------------------
# Device setup
# ----------------------------
def get_device():
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 0):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Laplace noise helpers
# ----------------------------
def laplace_sample(shape, scale, device):
    """Sample Laplace(0, scale) noise with a given tensor shape."""
    dist = torch.distributions.Laplace(
        loc=torch.tensor(0.0, device=device),
        scale=torch.tensor(float(scale), device=device),
    )
    return dist.sample(shape)


def add_noise_mirror_torch(x_t, delta_p, eps_eff, prev_noise, beta, device):
    """
    Noise-Mirror (paper equation):
      n ~ Laplace(0, Δ_p / ε_eff)
      N_t = β·n + (1-β)·N_{t-1}
      R_t = X_t + N_t

    Returns:
      (noisy_x, curr_noise)
    """
    if eps_eff <= 0:
        return x_t, prev_noise

    scale = float(delta_p) / max(float(eps_eff), 1e-12)
    n = laplace_sample(x_t.shape, scale, device)
    curr_noise = float(beta) * n + (1.0 - float(beta)) * prev_noise
    return x_t + curr_noise, curr_noise


# ----------------------------
# Sample-Mirror: compute adaptive sampling count Q
# ----------------------------
def compute_sample_count_Q(epsilon_p, delta_p, history_windows, window_size,
                           R_stream, N_stream, Q_prev, gamma):
    """
    Sample-Mirror (paper equations):

    1) ΔR = |R_t - R_{t-1}|
    2) ΔN = |N_t - N_{t-1}|
    3) D = Var(ΔR) - Var(ΔN)
    4) Q_curr = max{(ε_p/Δ_p) · sqrt(D), 1}
    5) Q = γ·Q_prev + (1-γ)·Q_curr   (EMA smoothing)

    Notes:
    - We compute the statistic over the most recent L consecutive differences,
      where L = min(window_size * history_windows, T-1).
    """
    T = R_stream.shape[0]
    total_history = int(window_size) * int(history_windows)

    if T < 2:
        return 1.0

    # Use as much history as available
    L = min(total_history, T - 1)
    if L < 1:
        return 1.0

    # Take the most recent (L+1) steps
    seg_R = R_stream[-(L + 1):]   # (L+1, dim)
    seg_N = N_stream[-(L + 1):]   # (L+1, dim)

    # ΔR_t = |R_t - R_{t-1}|
    delta_R = torch.abs(seg_R[1:] - seg_R[:-1])           # (L, dim)
    delta_R_mean = torch.mean(delta_R, dim=1)             # (L,)

    # ΔN_t = |N_t - N_{t-1}|
    delta_N = torch.abs(seg_N[1:] - seg_N[:-1])           # (L, dim)
    delta_N_mean = torch.mean(delta_N, dim=1)             # (L,)

    # D = Var(ΔR) - Var(ΔN)
    var_delta_R = torch.var(delta_R_mean, unbiased=False)
    var_delta_N = torch.var(delta_N_mean, unbiased=False)
    D = var_delta_R - var_delta_N
    D = torch.clamp(D, min=1e-9)  # avoid non-positive due to noise/finite samples

    # Q_curr = max{(ε_p/Δ_p) · sqrt(D), 1}
    Q_curr = (float(epsilon_p) / float(delta_p)) * math.sqrt(float(D.item()))
    Q_curr = max(1.0, Q_curr)

    # EMA smoothing
    Q = float(gamma) * float(Q_prev) + (1.0 - float(gamma)) * float(Q_curr)

    # Clamp to a reasonable range
    Q = max(1.0, min(Q, float(window_size)))
    return Q


# ----------------------------
# Budget-Mirror: remaining budget within a sliding window
# ----------------------------
def compute_remaining_budget(epsilon_budget, consume_ledger, window_size):
    """
    Compute remaining budget within the current sliding window:

      remain = epsilon_budget - sum(consumption in the most recent window)

    consume_ledger is a per-timestep ledger where non-action steps store 0.
    """
    consumed = 0.0
    length_ = len(consume_ledger)
    back = min(int(window_size) - 1, length_)
    for k in range(back):
        consumed += consume_ledger[length_ - 1 - k]
    return float(epsilon_budget) - consumed


def find_first_release_in_window(eps_c_ledger, window_size):
    """
    Find the first release timestamp inside the current window,
    based on the release-consumption ledger (eps_c_ledger).
    """
    current = len(eps_c_ledger)
    start = max(0, current - int(window_size) + 1)
    for t in range(start, current):
        if eps_c_ledger[t] > 0:
            return t
    return current


# ----------------------------
# Noise-Mirror: dynamic beta
# ----------------------------
def compute_dynamic_beta(eps_remain, delta_p, beta_max=0.9, beta_min=0.5):
    """
    Noise-Mirror (paper equation):

      β = β_min + (β_max - β_min) · (1 - ε_remain / (2·Δ_p))

    Interpretation (matches this equation and implementation):
    - smaller ε_remain  -> larger β -> more weight on fresh noise
    - larger  ε_remain  -> smaller β -> more weight on historical noise
    """
    denom = 2.0 * float(delta_p)
    r = 1.0 - float(eps_remain) / float(max(denom, 1e-12))
    r = max(0.0, min(1.0, r))
    beta = float(beta_min) + (float(beta_max) - float(beta_min)) * r
    return float(np.clip(beta, beta_min, beta_max))


# ----------------------------
# Error metrics
# ----------------------------
def compute_error(X, R, metric="mae"):
    """Compute utility error between raw stream X and released stream R."""
    metric = metric.lower().strip()

    if metric == "mae":
        return torch.mean(torch.abs(X - R)).item()

    if metric == "mre":
        abs_diff = torch.abs(X - R)
        denom = torch.abs(X)
        mask0 = denom == 0
        rel = torch.empty_like(abs_diff)
        rel[mask0] = torch.abs(R)[mask0]
        rel[~mask0] = abs_diff[~mask0] / denom[~mask0]
        return torch.mean(rel).item()

    if metric == "sum_mae":
        X_cum = torch.cumsum(X, dim=0)
        R_cum = torch.cumsum(R, dim=0)
        return torch.mean(torch.abs(X_cum - R_cum)).item()

    if metric == "sum_mre":
        X_cum = torch.cumsum(X, dim=0)
        R_cum = torch.cumsum(R, dim=0)
        abs_diff = torch.abs(X_cum - R_cum)
        denom = torch.abs(X_cum)
        mask0 = denom == 0
        rel = torch.empty_like(abs_diff)
        rel[mask0] = torch.abs(R_cum)[mask0]
        rel[~mask0] = abs_diff[~mask0] / denom[~mask0]
        return torch.mean(rel).item()

    raise ValueError(f"Unknown metric: {metric}")


# ----------------------------
# Warm-up stage
# ----------------------------
def warm_up_stage_mirror(epsilon_p, delta_p, X, window_size, window_num, beta, device):
    """
    Warm-up stage: release with a fixed interval and Noise-Mirror.

    We follow a simple warm-up schedule:
      c_init = w / 20
      sample_interval = w / c_init
      epsilon_warmup = ε_p / c_init
    """
    c_init = window_size / 20
    sample_interval = int(window_size / c_init)
    total_for_warmup = int(window_size) * int(window_num)
    epsilon_warmup = float(epsilon_p) / float(c_init)

    dim = X.shape[1]
    R = torch.zeros((total_for_warmup, dim), device=device, dtype=torch.float32)
    N = torch.zeros((total_for_warmup, dim), device=device, dtype=torch.float32)

    eps_c_ledger = []   # release-consumption ledger ε_{c,t}
    svt_c_ledger = []   # SVT-consumption ledger svt_{c,t}

    last_R = None
    prev_noise = torch.zeros((dim,), device=device, dtype=torch.float32)

    for t in range(total_for_warmup):
        if t % sample_interval == 0:
            noisy, curr_noise = add_noise_mirror_torch(
                X[t], delta_p, epsilon_warmup, prev_noise, beta, device
            )
            R[t] = noisy
            N[t] = curr_noise
            last_R = noisy
            prev_noise = curr_noise
            eps_c_ledger.append(float(epsilon_warmup))
            svt_c_ledger.append(0.0)
        else:
            R[t] = last_R
            N[t] = prev_noise
            eps_c_ledger.append(0.0)
            svt_c_ledger.append(0.0)

    return R, N, eps_c_ledger, svt_c_ledger, prev_noise


# ----------------------------
# Main workflow
# ----------------------------
def mirror_smoothing_workflow(
    epsilon, delta_s, delta_p, raw_stream, window_size,
    windownum_warm, windownum_updateQ,
    beta0=0.7, beta_min=0.5, beta_max=0.9, gamma=0.7,
    metric="mae", device=None
):
    """
    Mirror-Smoothing main pipeline (paper-aligned notation).

    Args:
      epsilon: total privacy budget ε
      delta_s: SVT sensitivity Δ_s
      delta_p: publish sensitivity Δ_p
      raw_stream: input stream (list/ndarray), shape (T, dim)
      window_size: sliding window size w
      windownum_warm: number of warm-up windows
      windownum_updateQ: number of historical windows used to update Q
      beta0: initial beta
      beta_min/beta_max: beta bounds
      gamma: EMA coefficient for Q
      metric: error metric
      device: torch device
    """
    if device is None:
        device = get_device()

    X = torch.tensor(raw_stream, device=device, dtype=torch.float32)
    T, dim = X.shape

    # Budget split (as in your paper text): ε_s = ε/2, ε_p = ε/2
    epsilon_s = float(epsilon) / 2.0      # sampling budget for SVT
    epsilon_p = float(epsilon) / 2.0      # publishing (release) budget
    epsilon_s1 = epsilon_s / 2.0          # SVT threshold budget
    epsilon_s2 = epsilon_s / 2.0          # SVT verification budget

    # Warm-up
    R_warm, N_warm, eps_c_ledger, svt_c_ledger, prev_noise = warm_up_stage_mirror(
        epsilon_p, delta_p, X, window_size, windownum_warm, beta0, device
    )

    # Initialize Q via Sample-Mirror
    Q = compute_sample_count_Q(
        epsilon_p, delta_p, windownum_updateQ, window_size,
        R_warm, N_warm, Q_prev=window_size / 20, gamma=gamma
    )

    # Allocate full-length tensors
    R = torch.zeros((T, dim), device=device, dtype=torch.float32)
    N = torch.zeros((T, dim), device=device, dtype=torch.float32)
    warm_len = int(windownum_warm) * int(window_size)
    R[:warm_len] = R_warm[:warm_len]
    N[:warm_len] = N_warm[:warm_len]

    # Privatized SVT threshold noise (rho)
    rho = laplace_sample(
        (1,), float(delta_s) / max(float(epsilon_s1), 1e-12), device
    )[0].item()

    beta = float(beta0)

    # Online stage
    for t in range(warm_len, T):
        # Budget-Mirror: remaining budgets in the current sliding window
        eps_remain = compute_remaining_budget(epsilon_p, eps_c_ledger, window_size)
        svt_remain = compute_remaining_budget(epsilon_s2, svt_c_ledger, window_size)

        # Data change magnitude (mean absolute change)
        diff = torch.mean(torch.abs(X[t] - X[t - 1])).item()

        # SVT threshold parameter (depends on Q)
        T_svt = float(Q) * float(delta_p) / max(float(epsilon_p), 1e-12)

        # SVT verification noise v
        v = laplace_sample(
            (1,),
            float(delta_s) / max((float(epsilon_s2) / (2.0 * float(Q))), 1e-12),
            device
        )[0].item()

        first_release_in_window = find_first_release_in_window(eps_c_ledger, window_size)

        # Decision branch 1: must publish when nearing window end (feasibility)
        remaining_slots = max((int(window_size) - (t - first_release_in_window)), 1)
        budget_per_slot = epsilon_p / float(Q)

        if remaining_slots <= int(eps_remain / max(budget_per_slot, 1e-12)):
            # Allocate budgets for this release point
            eps_alloc = eps_remain / float(remaining_slots)     # ε_{c,t}
            svt_alloc = svt_remain / float(Q)                   # svt_{c,t}

            # Effective perturbation budget (matches your implementation)
            eps_eff = max(float(eps_alloc + svt_alloc), 1e-12)

            noisy, curr_noise = add_noise_mirror_torch(
                X[t], delta_p, eps_eff, prev_noise, beta, device
            )

            R[t] = noisy
            N[t] = curr_noise
            prev_noise = curr_noise
            eps_c_ledger.append(float(eps_alloc))
            svt_c_ledger.append(float(svt_alloc))

        # Decision branch 2: SVT triggers a release (significant change)
        elif (diff + v) > (T_svt + rho) and eps_remain >= budget_per_slot:
            # Per-trigger SVT verification cost in the ledger
            svt_alloc = float(epsilon_s2) / float(Q)
            svt_c_ledger.append(float(svt_alloc))

            # Re-compute remaining SVT budget after accounting this verification
            svt_remain_new = compute_remaining_budget(epsilon_s2, svt_c_ledger, window_size)

            # Effective perturbation budget (matches your implementation)
            eps_eff = max(float(budget_per_slot + max(float(svt_remain_new), 0.0)), 1e-12)

            noisy, curr_noise = add_noise_mirror_torch(
                X[t], delta_p, eps_eff, prev_noise, beta, device
            )

            R[t] = noisy
            N[t] = curr_noise
            prev_noise = curr_noise
            eps_c_ledger.append(float(budget_per_slot))

        # Decision branch 3: reuse the last released value
        else:
            R[t] = R[t - 1]
            N[t] = prev_noise
            eps_c_ledger.append(0.0)
            svt_c_ledger.append(0.0)

        # Sample-Mirror: update Q (EMA-smoothed)
        Q = compute_sample_count_Q(
            epsilon_p, delta_p, windownum_updateQ, window_size,
            R[:t + 1], N[:t + 1], Q_prev=Q, gamma=gamma
        )

        # Noise-Mirror: update beta
        beta = compute_dynamic_beta(eps_remain, delta_p, beta_max=beta_max, beta_min=beta_min)

    err = compute_error(X, R, metric=metric)
    R_list = R.detach().cpu().numpy().tolist()
    return R_list, X.detach(), R.detach(), err


# ----------------------------
# Runner
# ----------------------------
def run_mirror_smoothing_gpu(
    epsilon_list, delta_s, delta_p,
    raw_stream, window_size,
    windownum_warm, windownum_updateQ,
    rounds, beta0=0.7, gamma=0.7, metric="mae", device=None
):
    """Run multiple eps values and average over multiple rounds."""
    if device is None:
        device = get_device()

    err_list = []

    for eps in epsilon_list:
        err_sum = 0.0
        for _ in range(int(rounds)):
            _, _, _, err = mirror_smoothing_workflow(
                eps, delta_s, delta_p,
                raw_stream, window_size,
                windownum_warm, windownum_updateQ,
                beta0=beta0,
                gamma=gamma,
                metric=metric,
                device=device
            )
            err_sum += err

        err_avg = err_sum / float(rounds)
        print(f"epsilon: {eps} Done! metric={metric}, value={err_avg}")
        err_list.append(err_avg)

    print("Mirror-Smoothing DONE!")
    return err_list


# --------------------------------------------------------------------------------------------------------------
# ablation study windows-size
# --------------------------------------------------------------------------------------------------------------
# import os
# import csv
# if __name__ == "__main__":
#     # 1. Create output directory (create if not exists)
#     output_dir = "output_ablation_study"
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 2. Initialize experiment configurations
#     set_seed(42)
#     device = get_device()
#     print("Using device:", device)
#     print("\n" + "=" * 70)
#     print("Mirror-Smoothing test (paper-aligned implementation)")
#     print("=" * 70)

#     # Experiment configurations
#     datasets = ["Flu_Deaths", "unemp", "tdv"]
#     window_sizes = [40, 60, 80, 100, 120, 140, 160]
#     epsilon_list = [0.1]
#     sensitivity_s = 1
#     sensitivity_p = 1
#     windownum_warm = 1
#     windownum_updateE = 2
#     windownum_updateQ = 2
#     rounds = 3
#     metric = "mae"
#     gamma = 0.7

#     # 3. Define result file path
#     result_file = os.path.join(output_dir, "window_size_ablation_results.csv")
    

#     with open(result_file, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         # Write header first
#         writer.writerow(["dataset", "window_size", "metric", "error", "device", "seed"])
        
#         # Nested for loops: traverse datasets → window sizes (more structured)
#         for ds in datasets:
#             print(f"\nProcessing dataset: {ds}")
#             raw_stream = data_reader(ds)
            
#             # Inner loop: traverse all window sizes for ablation study
#             for window_size in window_sizes:
#                 # Run experiment for current window size
#                 error_list = run_mirror_smoothing_gpu(
#                     epsilon_list, sensitivity_s, sensitivity_p,
#                     raw_stream, window_size,
#                     windownum_warm, windownum_updateQ,
#                     rounds, beta0=0.7, gamma=gamma, metric=metric, device=device
#                 )
#                 error = float(error_list[0]) 
#                 # Print real-time result
#                 print(f"  Window size {window_size} | {metric}: {error:.4f}")
                
#                 # Write result to CSV (single writer, no repeated open/close)
#                 writer.writerow([ds, window_size, metric, round(error, 4), str(device), 42])

#     print(f"\nAll experimental results saved to: {os.path.abspath(result_file)}")
# --------------------------------------------------------------------------------------------------------------
# ablation study windows-size
# --------------------------------------------------------------------------------------------------------------







# --------------------------------------------------------------------------------------------------------------
# ablation study gamma
# --------------------------------------------------------------------------------------------------------------
import os
import csv

if __name__ == "__main__":
    # 1. Create output directory (create if not exists)
    output_dir = "output_ablation_study"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Initialize experiment configurations
    set_seed(42)
    device = get_device()
    print("Using device:", device)
    print("\n" + "=" * 70)
    print("Mirror-Smoothing gamma ablation (paper-aligned implementation)")
    print("=" * 70)

    # Experiment configurations
    datasets = ["Flu_Deaths", "unemp", "tdv"]
    gamma_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]   # gamma ablation
    epsilon_list = [0.1]

    sensitivity_s = 1
    sensitivity_p = 1
    windownum_warm = 1
    windownum_updateQ = 2
    rounds = 3
    metric = "mre"
    window_size = 120  # 

    # 3. Define result file path
    result_file = os.path.join(output_dir, "gamma_ablation_results.csv")

    with open(result_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "gamma", "window_size", "epsilon", "metric", "error", "device", "seed"])

        for ds in datasets:
            print(f"\nProcessing dataset: {ds}")
            raw_stream = data_reader(ds)

            for gamma in gamma_list:
                error_list = run_mirror_smoothing_gpu(
                    epsilon_list, sensitivity_s, sensitivity_p,
                    raw_stream, window_size,
                    windownum_warm, windownum_updateQ,
                    rounds, beta0=0.7, gamma=gamma, metric=metric, device=device
                )
                error = float(error_list[0])  # epsilon_list只有一个值

                print(f"  gamma {gamma:.1f} | window {window_size} | eps {epsilon_list[0]} | {metric}: {error:.4f}")

                writer.writerow([
                    ds, gamma, window_size, epsilon_list[0],
                    metric, round(error, 4), str(device), 42
                ])

    print(f"\nAll experimental results saved to: {os.path.abspath(result_file)}")
# --------------------------------------------------------------------------------------------------------------
# ablation study gamma
# --------------------------------------------------------------------------------------------------------------