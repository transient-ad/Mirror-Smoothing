#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BucOrder on GPU (PyTorch)

"""

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
    """
    add noise
    """
    if eps <= 0:
        return value

    scale = float(sensitivity) / max(float(eps), 1e-12)
    noise = laplace_sample(value.shape if isinstance(
        value, torch.Tensor) else (1,), scale, device)
    if isinstance(value, torch.Tensor):
        return value + noise
    else:
        return value + noise[0].item()


# ----------------------------
# Randomized Response
# ----------------------------
def randomized_response_torch(data, k, eps, device):
    """
    Randomized Response

    """
    n = data.shape[0]
    p = math.exp(eps) / (math.exp(eps) + k - 1)

    rand_vals = torch.rand(n, device=device)
    keep_original = rand_vals < p

    noisy_data = data.clone()
    need_random = ~keep_original

    if torch.sum(need_random) > 0:

        random_buckets = torch.randint(0, k, (n,), device=device)

        same_as_original = (random_buckets == data) & need_random
        while torch.sum(same_as_original) > 0:
            random_buckets[same_as_original] = torch.randint(
                0, k, (torch.sum(same_as_original).item(),), device=device)
            same_as_original = (random_buckets == data) & need_random

        noisy_data[need_random] = random_buckets[need_random]

    return noisy_data


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
# BucOrder core code
# ----------------------------
def bucorder_batch_process(data_batch, buc_size, eps_buc, eps_pub, sensitivity, device):

    batch_size, dim = data_batch.shape
    published = torch.zeros_like(data_batch)

    for d in range(dim):
        data_col = data_batch[:, d]

        max_val = torch.max(data_col).item()
        buc_num = int(np.ceil(max_val / buc_size)) + 1
        buc_num = max(buc_num, 2)  

        buc_alloc = (data_col / buc_size).long()
        buc_alloc = torch.clamp(buc_alloc, 0, buc_num - 1)

        # Randomized Response
        noise_bucalloc = randomized_response_torch(
            buc_alloc, buc_num, eps_buc, device)

        buc_sum = torch.zeros(buc_num, device=device, dtype=torch.float32)
        buc_innernum = torch.zeros(buc_num, device=device, dtype=torch.float32)

        for j in range(buc_num):
            mask = (noise_bucalloc == j)
            buc_innernum[j] = torch.sum(mask).float()
            buc_sum[j] = torch.sum(data_col[mask])

        buc_sum = buc_sum + \
            laplace_sample((buc_num,), sensitivity / eps_pub, device)

        # monotonicity constraint
        for j in range(1, buc_num):
            if buc_sum[j] < buc_sum[j - 1]:
                buc_sum[j] = buc_sum[j - 1]

        for j in range(buc_num):
            if buc_innernum[j] > 0:
                lower_bound = j * buc_size * buc_innernum[j]
                upper_bound = (j + 1) * buc_size * buc_innernum[j] - 1
                buc_sum[j] = torch.clamp(buc_sum[j], lower_bound, upper_bound)

        for i in range(batch_size):
            bucket_idx = noise_bucalloc[i]
            if buc_innernum[bucket_idx] > 0:
                published[i, d] = buc_sum[bucket_idx] / \
                    buc_innernum[bucket_idx]
            else:
                published[i, d] = data_col[i]  # fallback

    return published


# ----------------------------
# BucOrder
# ----------------------------
def bucorder_workflow(epsilon, sensitivity, raw_stream, delay_time, buc_size,
                      metric="mae", device=None):

    if device is None:
        device = get_device()

    raw_tensor = torch.tensor(raw_stream, device=device, dtype=torch.float32)
    T, dim = raw_tensor.shape

    # Budget split
    eps_buc = float(epsilon) / 4.0
    eps_pub = float(epsilon) - eps_buc

    published = torch.zeros_like(raw_tensor)

    num_batches = T // delay_time

    for i in range(num_batches):
        start_idx = i * delay_time
        end_idx = (i + 1) * delay_time

        data_batch = raw_tensor[start_idx:end_idx]

        pub_batch = bucorder_batch_process(
            data_batch, buc_size, eps_buc, eps_pub, sensitivity, device
        )

        published[start_idx:end_idx] = pub_batch

    if T % delay_time > 0:
        start_idx = num_batches * delay_time
        data_batch = raw_tensor[start_idx:]

        pub_batch = bucorder_batch_process(
            data_batch, buc_size, eps_buc, eps_pub, sensitivity, device
        )

        published[start_idx:] = pub_batch

    err = compute_error(raw_tensor, published, metric=metric)
    published_list = published.detach().cpu().numpy().tolist()
    return published_list, raw_tensor.detach(), published.detach(), err


# ----------------------------
# Runner
# ----------------------------
def run_bucorder_gpu(epsilon_list, sensitivity, raw_stream, delay_time, buc_size,
                     round_, metric="mae", device=None):
    """
    run BucOrder 

    """
    if device is None:
        device = get_device()

    results = []

    for eps in epsilon_list:
        err_sum = 0.0
        for _ in range(int(round_)):
            _, _, _, err = bucorder_workflow(
                eps, sensitivity, raw_stream, delay_time, buc_size,
                metric=metric, device=device
            )
            err_sum += err

        err_avg = err_sum / float(round_)
        print(f"epsilon: {eps} Done! metric={metric}, value={err_avg}")
        results.append(err_avg)

    print("BucOrder DONE!")
    return results


# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Using device:", device)
    print("\n" + "="*70)
    print("BucOrder ")
    print("="*70)

    datasets = ["covid19", ]

    for ds in datasets:
        print(f"\n{ds}:")
        raw_stream = data_reader(ds)

        epsilon = [1]
        sensitivity = 1
        delay_time = 120
        buc_size = 100
        round_ = 5

        metric = "mae"

        error_ = run_bucorder_gpu(
            epsilon, sensitivity,
            raw_stream, delay_time, buc_size,
            round_, metric=metric, device=device
        )
        print(f"{ds} result ({metric}):", error_)
