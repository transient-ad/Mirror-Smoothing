#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import random
import numpy as np
import torch
from methods.data_process import data_reader


# ----------------------------
# Device setup
# ----------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Laplace noise
# ----------------------------
def laplace_sample(shape, scale, device):
    dist = torch.distributions.Laplace(
        loc=torch.tensor(0.0, device=device),
        scale=torch.tensor(float(scale), device=device)
    )
    return dist.sample(shape)


def add_noise_torch(value, sensitivity, eps, device):

    if eps <= 0:
        return value

    scale = float(sensitivity) / max(float(eps), 1e-12)
    if isinstance(value, torch.Tensor):
        noise = laplace_sample(value.shape, scale, device)
        return value + noise
    else:
        noise = laplace_sample((1,), scale, device)
        return value + noise[0].item()


# ----------------------------
# Error metrics
# ----------------------------
def compute_error(raw_tensor, pub_tensor, metric="mae"):
    metric = metric.lower().strip()

    if metric == "mae":
        return torch.mean(torch.abs(raw_tensor - pub_tensor)).item()

    if metric == "mse":
        return torch.mean((raw_tensor - pub_tensor) ** 2).item()

    if metric == "mre":
        abs_diff = torch.abs(raw_tensor - pub_tensor)
        denom = torch.abs(raw_tensor)
        mask0 = denom == 0
        rel = torch.empty_like(abs_diff)
        rel[mask0] = torch.abs(pub_tensor)[mask0]
        rel[~mask0] = abs_diff[~mask0] / denom[~mask0]
        return torch.mean(rel).item()

    raise ValueError(f"Unknown metric: {metric}")


# ----------------------------
# DPI
# ----------------------------
def dpi_workflow(epsilon, raw_stream, metric="mae", device=None, verbose=False):

    if device is None:
        device = get_device()

    raw_tensor = torch.tensor(raw_stream, device=device, dtype=torch.float32)
    T, dim = raw_tensor.shape

    if verbose:
        print(f"  Processing {T} samples")

    # Budget split
    eps_per_step = float(epsilon) / float(T)

    sensitivity = 1.0

    published = torch.zeros_like(raw_tensor)

    alpha_ewma = 0.3  # 平滑系数

    ewma = raw_tensor[0].clone()

    for t in range(T):
        if verbose and t % 500 == 0 and t > 0:
            print(f"    Progress: {t}/{T} ({100*t/T:.1f}%)")

        current = raw_tensor[t]

        ewma = alpha_ewma * current + (1 - alpha_ewma) * ewma

        noisy_output = add_noise_torch(ewma, sensitivity, eps_per_step, device)

        noisy_output = torch.clamp(noisy_output, min=0)

        published[t] = noisy_output

    err = compute_error(raw_tensor, published, metric=metric)
    published_list = published.detach().cpu().numpy().tolist()
    return published_list, raw_tensor.detach(), published.detach(), err


# ----------------------------
# Runner
# ----------------------------
def run_dpi_gpu(epsilon_list, raw_stream, round_=5, metric="mae",
                device=None, verbose=False):

    if device is None:
        device = get_device()

    results = []

    for eps in epsilon_list:
        err_sum = 0.0
        for r in range(int(round_)):
            if verbose:
                print(f"  Round {r+1}/{round_}")
            _, _, _, err = dpi_workflow(
                eps, raw_stream,
                metric=metric, device=device, verbose=verbose
            )
            err_sum += err

        err_avg = err_sum / float(round_)
        print(f"epsilon: {eps} Done! metric={metric}, value={err_avg}")
        results.append(err_avg)

    print("DPI DONE!")
    return results


# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Using device:", device)

    datasets = ["covid19", "Dth", "Uem"]

    for ds in datasets:
        print(f"\n{ds}:")
        raw_stream = data_reader(ds)
        print(f"  Data size: {len(raw_stream)} samples")

        epsilon = [1]
        round_ = 5

        metric = "mae"

        error_ = run_dpi_gpu(
            epsilon, raw_stream,
            round_=round_, metric=metric, device=device, verbose=True
        )
        print(f"\n{ds} result ({metric}):", error_)
