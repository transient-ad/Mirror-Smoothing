import sys
import os
import math
import numpy as np

# GPU (optional)
import torch

from methods.data_process import data_reader


ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)


# ----------------------------
# Device setup (GPU if available)
# ----------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Laplace noise on GPU/CPU via torch
# ----------------------------
def laplace_noise(shape, scale, device):
    # scale can be float or tensor
    dist = torch.distributions.Laplace(loc=torch.tensor(0.0, device=device),
                                       scale=torch.tensor(scale, device=device))
    return dist.sample(shape)


def add_noise_torch(histo_vec, sensitivity, eps, device):
    """
    histo_vec: torch.Tensor shape (dim,)
    returns: torch.Tensor shape (dim,)
    """
    if eps <= 0:
        # no budget, no noise
        return histo_vec
    scale = sensitivity / eps
    noise = laplace_noise(histo_vec.shape, scale, device)
    return histo_vec + noise


# ----------------------------
# Distance / variance utilities (vectorized)
# ----------------------------
def count_dis_torch(h1, h2):
    # mean L1 distance per-dimension
    return torch.mean(torch.abs(h1 - h2))


def count_vardis_torch(h1, h2):
    # mean squared diff, mean abs diff
    diff = h1 - h2
    vardis = torch.mean(diff * diff)
    varexp = torch.mean(torch.abs(diff))
    return vardis, varexp


def agv_vardis_torch(compute_num, window_size, published_stream):
    """
    published_stream: torch.Tensor shape (T, dim)
    returns scalar tensor (variance-like estimate)
    """
    T = published_stream.shape[0]
    total_ = window_size * compute_num  # e.g., 120*2=240

    # We need last (total_ - 1) consecutive pairs, but require at least total_+1 samples as in your original code
    if T >= total_ + 1:
        L = total_  # window length to look back
    else:
        L = T - 1   # use what we have
        if L <= 1:
            return torch.tensor(0.0, device=published_stream.device)

    # take last (L+1) points to form L pairs
    seg = published_stream[-(L+1):]  # shape (L+1, dim)
    prev = seg[:-1]                  # shape (L, dim)
    curr = seg[1:]                   # shape (L, dim)

    diff = curr - prev
    vardis = torch.mean(diff * diff, dim=1)       # shape (L,)
    varexp = torch.mean(torch.abs(diff), dim=1)   # shape (L,)

    # keep only vardis > 0 (same as your code)
    mask = vardis > 0
    if torch.sum(mask) == 0:
        return torch.tensor(0.0, device=published_stream.device)

    vardis_mean = torch.mean(vardis[mask])
    varexp_mean = torch.mean(varexp[mask])

    # var = E[x^2] - (E[|x|])^2 (your original form)
    return vardis_mean - varexp_mean * varexp_mean


def update_optimalc_torch(epsilon_p, sensitivity_p, compute_num, window_size, published_stream):
    """
    published_stream: torch.Tensor (T, dim)
    returns python int optimal_c
    """
    E_dis = agv_vardis_torch(compute_num, window_size, published_stream)
    E_dis_val = float(E_dis.item())

    theta_ = 1 / 2
    inside = (1 - 2 * theta_)**2 * (window_size**2) + (3 * (theta_**2)
                                                       * E_dis_val * (epsilon_p**2)) / (sensitivity_p**2)
    inside = max(inside, 0.0)
    optimal_c = max(
        int((math.sqrt(inside) - (1 - 2 * theta_) * window_size) / (6 * theta_)), 1)
    return optimal_c


# ----------------------------
# Remaining epsilon utilities (keep as python list: tiny cost)
# ----------------------------
def compute_epsremain(epsilon_p, eps_consume, window_size):
    eps_con = 0.0
    length_ = len(eps_consume)
    # only look back window_size-1
    back = min(window_size - 1, length_)
    for i in range(back):
        eps_con += eps_consume[length_ - 1 - i]
    return epsilon_p - eps_con


def find_firstsample(eps_con, window_size):
    current = len(eps_con)
    start = max(0, current - window_size + 1)
    for i in range(start, current):
        if eps_con[i] > 0:
            return i
    return current


# ----------------------------
# Error metric (MAE / MRE switch)
# ----------------------------
def compute_error(raw_tensor, pub_tensor, metric="mae"):
    """
    raw_tensor, pub_tensor: torch.Tensor (T, dim)
    metric: "mae" or "mre"
    """
    if raw_tensor.shape != pub_tensor.shape:
        raise ValueError("raw and published shapes mismatch")

    if metric.lower() == "mae":
        return torch.mean(torch.abs(raw_tensor - pub_tensor)).item()

    if metric.lower() == "mre":
        # mean relative error with 0-handling like your commented version:
        # if raw==0 => abs(pub)
        abs_diff = torch.abs(raw_tensor - pub_tensor)
        raw_abs = torch.abs(raw_tensor)

        # where raw==0 -> use abs(pub), else abs_diff/raw
        mask0 = raw_abs == 0
        rel = torch.empty_like(abs_diff)
        rel[mask0] = torch.abs(pub_tensor)[mask0]
        rel[~mask0] = abs_diff[~mask0] / raw_abs[~mask0]

        return torch.mean(rel).item()

    raise ValueError("metric must be 'mae' or 'mre'")


# ----------------------------
# Warm-up stage (GPU)
# ----------------------------
def warm_up_stage(epsilon, sensitivity_p, raw_stream_tensor, window_size, window_num, device):
    """
    raw_stream_tensor: torch.Tensor (T, dim)
    returns:
      published_stream_tensor: torch.Tensor (total_for_warmup, dim)
      eps_consumed: list[float]
      svt_consumed: list[float]
    """

    c_init = window_size / 20
    sample_interval = int(window_size / c_init)  # e.g. 20
    total_for_warmup = window_size * window_num
    epsilon_warmup = epsilon / c_init

    dim = raw_stream_tensor.shape[1]
    published = torch.zeros((total_for_warmup, dim),
                            device=device, dtype=torch.float32)

    eps_consumed = []
    svt_consumed = []

    last_pub = None
    for i in range(total_for_warmup):
        if i % sample_interval == 0:
            noisy = add_noise_torch(
                raw_stream_tensor[i], sensitivity_p, epsilon_warmup, device)
            published[i] = noisy
            last_pub = noisy
            eps_consumed.append(float(epsilon_warmup))
            svt_consumed.append(0.0)
        else:
            published[i] = last_pub
            eps_consumed.append(0.0)
            svt_consumed.append(0.0)

    return published, eps_consumed, svt_consumed


# ----------------------------
# SPAS main workflow (GPU)
# ----------------------------
def SPAS_workflow(epsilon, sensitivity_s, sensitivity_p, raw_stream, window_size,
                  windownum_warm, windownum_updateE, metric="mae", device=None):
    """
    raw_stream: list[list[int]] or np.ndarray
    returns: published_stream as list[list[float]] (for compatibility)
    """
    if device is None:
        device = get_device()

    raw_tensor = torch.tensor(
        raw_stream, device=device, dtype=torch.float32)  # (T, dim)
    T, dim = raw_tensor.shape

    # ---- budget split (keep your original SPAS split) ----
    epsilon_s = epsilon / 2
    epsilon_p = epsilon - epsilon_s
    eps_1 = epsilon_s / 2
    eps_2 = epsilon_s - eps_1

    # ---- warm-up ----
    pub_warm, eps_consumed, svt_consumed = warm_up_stage(
        epsilon, sensitivity_p, raw_tensor, window_size, windownum_warm, device
    )

    # ---- init optimal_c ----
    optimal_c = update_optimalc_torch(
        epsilon_p, sensitivity_p, windownum_updateE, window_size, pub_warm
    )

    # ---- allocate published tensor for whole stream ----
    published = torch.zeros((T, dim), device=device, dtype=torch.float32)
    # fill warm-up part
    warm_len = windownum_warm * window_size
    published[:warm_len] = pub_warm[:warm_len]

    # rho_ is a noisy threshold (scalar)
    rho_ = add_noise_torch(torch.tensor(
        [0.0], device=device), sensitivity_s, eps_1, device)[0]

    # ---- online stage ----
    for i in range(warm_len, T):
        eps_remain = compute_epsremain(epsilon_p, eps_consumed, window_size)
        T_ = optimal_c * sensitivity_p / epsilon_p

        diff = count_dis_torch(raw_tensor[i], raw_tensor[i - 1])
        v_ = add_noise_torch(torch.tensor([0.0], device=device),
                             sensitivity_s, eps_2 / (2 * optimal_c), device)[0]

        first_sample_inwindow = find_firstsample(eps_consumed, window_size)

        # ---- force publish (avoid future budget deficit) ----
        if window_size - (i - first_sample_inwindow) <= int(eps_remain / (epsilon_p / optimal_c)):
            eps_svt_remain = compute_epsremain(
                eps_2, svt_consumed, (i - first_sample_inwindow + 1)
            )

            eff_eps = eps_remain / \
                (window_size - (i - first_sample_inwindow)) + \
                eps_svt_remain / optimal_c
            noisy = add_noise_torch(
                raw_tensor[i], sensitivity_p, eff_eps, device)

            svt_consumed.append(eps_svt_remain / optimal_c)
            published[i] = noisy
            eps_consumed.append(
                eps_remain / (window_size - (i - first_sample_inwindow)))

        # ---- SVT trigger publish ----
        elif (diff + v_) > (T_ + rho_) and eps_remain >= (epsilon_p / optimal_c):
            svt_consumed.append(eps_2 / optimal_c)
            eps_svt_remain = compute_epsremain(
                eps_2, svt_consumed, window_size)

            eff_eps = epsilon_p / optimal_c + max(eps_svt_remain, 0.0)
            noisy = add_noise_torch(
                raw_tensor[i], sensitivity_p, eff_eps, device)

            published[i] = noisy
            eps_consumed.append(epsilon_p / optimal_c)

        # ---- reuse last published ----
        else:
            published[i] = published[i - 1]
            eps_consumed.append(0.0)
            svt_consumed.append(0.0)

        # update optimal_c using latest published prefix
        # (use published[:i+1] to match “history up to now”)
        optimal_c = update_optimalc_torch(
            epsilon_p, sensitivity_p, windownum_updateE, window_size, published[:i+1]
        )

    # return published as python list of lists for compatibility with your other code
    published_list = published.detach().cpu().numpy().tolist()
    return published_list, raw_tensor.detach(), published.detach()


# ----------------------------
# Runner (MAE / MRE switch)
# ----------------------------
def run_SPAS_gpu(epsilon_list, sensitivity_s, sensitivity_p,
                 raw_stream, window_size,
                 windownum_warm, windownum_updateE,
                 round_, Flag_=0, metric="mae", device=None):

    if device is None:
        device = get_device()

    MAE_list = []

    if Flag_ == 0:  # varying epsilon
        for eps in epsilon_list:
            err_sum = 0.0
            for _ in range(round_):
                pub_list, raw_t, pub_t = SPAS_workflow(
                    eps, sensitivity_s, sensitivity_p,
                    raw_stream, window_size,
                    windownum_warm, windownum_updateE,
                    metric=metric, device=device
                )
                err_sum += compute_error(raw_t, pub_t, metric=metric)

            err_avg = err_sum / round_
            print(f"epsilon: {eps} done. {metric}={err_avg}")
            MAE_list.append(err_avg)

        print("SPAS DONE!")
    else:  # varying window size
        for w in window_size:
            err_sum = 0.0
            for _ in range(round_):
                pub_list, raw_t, pub_t = SPAS_workflow(
                    epsilon_list, sensitivity_s, sensitivity_p,
                    raw_stream, w,
                    windownum_warm, windownum_updateE,
                    metric=metric, device=device
                )
                err_sum += compute_error(raw_t, pub_t, metric=metric)

            err_avg = err_sum / round_
            print(f"window size: {w} done. {metric}={err_avg}")
            MAE_list.append(err_avg)

        print("SPAS DONE!")

    return MAE_list


# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    device = get_device()
    print("Using device:", device)

    datasets = ["covid19", ]

    for i in datasets:

        raw_stream = data_reader(i)
        epsilon = [1]
        sensitivity_s = 1
        sensitivity_p = 1
        window_size = 120
        windownum_warm = 1
        windownum_updateE = 2
        round_ = 5

        metric = "mae"

        error_ = run_SPAS_gpu(
            epsilon, sensitivity_s, sensitivity_p,
            raw_stream, window_size,
            windownum_warm, windownum_updateE,
            round_, Flag_=0, metric=metric, device=device
        )
        print("Result:", error_)
