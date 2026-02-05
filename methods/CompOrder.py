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

    if metric == "mre":
        abs_diff = torch.abs(raw_tensor - pub_tensor)
        denom = torch.abs(raw_tensor)
        mask0 = denom == 0
        rel = torch.empty_like(abs_diff)
        rel[mask0] = torch.abs(pub_tensor)[mask0]
        rel[~mask0] = abs_diff[~mask0] / denom[~mask0]
        return torch.mean(rel).item()

    if metric == "sum_mae":
        raw_cum = torch.cumsum(raw_tensor, dim=0)
        pub_cum = torch.cumsum(pub_tensor, dim=0)
        return torch.mean(torch.abs(raw_cum - pub_cum)).item()

    if metric == "sum_mre":
        raw_cum = torch.cumsum(raw_tensor, dim=0)
        pub_cum = torch.cumsum(pub_tensor, dim=0)
        abs_diff = torch.abs(raw_cum - pub_cum)
        denom = torch.abs(raw_cum)
        mask0 = denom == 0
        rel = torch.empty_like(abs_diff)
        rel[mask0] = torch.abs(pub_cum)[mask0]
        rel[~mask0] = abs_diff[~mask0] / denom[~mask0]
        return torch.mean(rel).item()

    raise ValueError(f"Unknown metric: {metric}")


# ----------------------------
# CompOrder 核心算法
# ----------------------------
def comporder_workflow(epsilon, sensitivity, raw_stream, delay_time,
                       metric="mae", device=None, verbose=False):

    if device is None:
        device = get_device()

    raw_tensor = torch.tensor(raw_stream, device=device, dtype=torch.float32)
    T, dim = raw_tensor.shape

    if verbose:
        print(f"  Processing {T} samples with delay_time={delay_time}")

    # Budget split
    eps_post = float(epsilon) / 4.0
    eps_pub = float(epsilon) - eps_post
    eps_1 = eps_post / 2.0
    eps_2 = eps_post / 2.0

    published = torch.zeros_like(raw_tensor)

    # rho
    rho = add_noise_torch(0.0, sensitivity, eps_1 / 2.0, device)

    flag_ = []

    for i in range(T):

        if verbose and i % 500 == 0 and i > 0:
            print(f"    Progress: {i}/{T} ({100*i/T:.1f}%)")

        if raw_tensor[i, 0] > sensitivity:
            noise_result = sensitivity + \
                add_noise_torch(0.0, sensitivity, eps_pub, device)
        else:
            noise_result = raw_tensor[i, 0] + \
                add_noise_torch(0.0, sensitivity, eps_pub, device)

        temp = []
        compare_end = min(i + delay_time, T)

        if compare_end > i + 1:
            eps_compare = eps_2 / (2.0 * (2.0 * float(delay_time) - 1.0))

            for j in range(i + 1, compare_end):

                compare_noise = add_noise_torch(
                    0.0, sensitivity, eps_compare, device)

                diff = raw_tensor[i, 0] - raw_tensor[j, 0] + compare_noise

                if diff > rho:
                    temp.append(0)  # data[i] > data[j]
                else:
                    temp.append(1)  # data[i] <= data[j]

        flag_.append(temp)

        low_bound = 0.0
        high_bound = 1e9

        start_idx = max(0, i - delay_time + 1)

        for j in range(start_idx, i):

            offset = i - j - 1
            if offset < len(flag_[j]):
                if flag_[j][offset] == 0:
                    # data[j] > data[i],
                    if published[j, 0] > low_bound:
                        low_bound = published[j, 0].item()
                else:
                    # data[j] <= data[i],
                    if published[j, 0] < high_bound:
                        high_bound = published[j, 0].item()

        if noise_result < low_bound or noise_result > high_bound:
            if low_bound > high_bound:

                noise_result = (low_bound + high_bound) / 2.0
            elif noise_result < low_bound:
                noise_result = low_bound
            elif noise_result > high_bound:
                noise_result = high_bound

        published[i, 0] = noise_result

    err = compute_error(raw_tensor, published, metric=metric)
    published_list = published.detach().cpu().numpy().tolist()
    return published_list, raw_tensor.detach(), published.detach(), err


# ----------------------------
# Runner
# ----------------------------
def run_comporder_gpu(epsilon_list, sensitivity, raw_stream, delay_time,
                      round_, metric="mae", device=None, verbose=False):

    if device is None:
        device = get_device()

    results = []

    for eps in epsilon_list:
        err_sum = 0.0
        for r in range(int(round_)):
            if verbose:
                print(f"  Round {r+1}/{round_}")
            _, _, _, err = comporder_workflow(
                eps, sensitivity, raw_stream, delay_time,
                metric=metric, device=device, verbose=verbose
            )
            err_sum += err

        err_avg = err_sum / float(round_)
        print(f"epsilon: {eps} Done! metric={metric}, value={err_avg}")
        results.append(err_avg)

    print("CompOrder DONE!")
    return results


# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Using device:", device)

    datasets = ["covid19"]

    for ds in datasets:
        print(f"\n{ds}:")
        raw_stream = data_reader(ds)
        print(f"  Data size: {len(raw_stream)} samples")

        epsilon = [1]
        sensitivity = 1
        delay_time = 100
        round_ = 3

        metric = "mae"

        error_ = run_comporder_gpu(
            epsilon, sensitivity,
            raw_stream, delay_time,
            round_, metric=metric, device=device, verbose=True
        )
        print(f"\n{ds} result ({metric}):", error_)
