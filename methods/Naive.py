#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
import numpy as np
import torch
from methods.data_process import data_reader


# ----------------------------
# Device + seed
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
# GPU Laplace noise + Uniform workload
# ----------------------------
@torch.no_grad()
def uniform_workload_gpu(epsilon, sensitivity, raw_stream, window_size, device):
    """
    raw_stream: list[list] or np.ndarray, shape (T, dim)
    returns:
      raw: torch.Tensor (T, dim) on device
      pub: torch.Tensor (T, dim) on device
    """
    raw = torch.tensor(raw_stream, device=device, dtype=torch.float32)
    T, dim = raw.shape

    eps_per_release = float(epsilon) / float(window_size)
    scale = float(sensitivity) / max(eps_per_release, 1e-12)

    dist = torch.distributions.Laplace(
        loc=torch.tensor(0.0, device=device),
        scale=torch.tensor(scale, device=device),
    )
    noise = dist.sample((T, dim))
    pub = raw + noise
    return raw, pub


# ----------------------------
# Point metrics (GPU)
# ----------------------------
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


# ----------------------------
# Query metrics (GPU)
# ----------------------------
@torch.no_grad()
def sum_query_metric(raw, pub, query_num=100, metric="mae"):
    """
    随机区间和查询：对齐你原 sum_query 的逻辑，但全 GPU 计算
    使用 prefix-sum：
      sum[a:b] = ps[b] - ps[a]
    """
    T, dim = raw.shape
    if query_num <= 0:
        return 0.0

    # CPU 生成区间 (a,b)
    intervals = []
    while len(intervals) < query_num:
        a = random.randint(0, T - 1)
        b = random.randint(0, T - 1)
        if a == b:
            continue
        if a > b:
            a, b = b, a
        if b == a:
            continue
        intervals.append((a, b))

    idx_a = torch.tensor([x[0] for x in intervals],
                         device=raw.device, dtype=torch.long)
    idx_b = torch.tensor([x[1] for x in intervals],
                         device=raw.device, dtype=torch.long)

    # prefix sums: (T+1, dim)
    raw_ps = torch.zeros((T + 1, dim), device=raw.device, dtype=raw.dtype)
    pub_ps = torch.zeros((T + 1, dim), device=pub.device, dtype=pub.dtype)
    raw_ps[1:] = torch.cumsum(raw, dim=0)
    pub_ps[1:] = torch.cumsum(pub, dim=0)

    raw_sum = raw_ps[idx_b] - raw_ps[idx_a]  # (Q, dim)
    pub_sum = pub_ps[idx_b] - pub_ps[idx_a]  # (Q, dim)

    m = metric.lower()
    if m == "mae":
        return torch.mean(torch.abs(raw_sum - pub_sum)).item()
    if m == "mre":
        return point_mre(raw_sum, pub_sum)
    raise ValueError("metric must be 'mae' or 'mre' for sum_query_metric")


@torch.no_grad()
def count_query_metric(raw, pub, query_num=100, metric="mae"):

    T, dim = raw.shape
    if query_num <= 0:
        return 0.0

    max_vals = torch.max(raw, dim=0).values  # (dim,)

    raw_counts_all = []
    pub_counts_all = []

    for d in range(dim):
        maxv = float(max_vals[d].item())

        lows = []
        highs = []
        while len(lows) < query_num:
            a = random.uniform(0.0, maxv)
            b = random.uniform(0.0, maxv)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            lows.append(a)
            highs.append(b)

        low = torch.tensor(lows, device=raw.device, dtype=raw.dtype)    # (Q,)
        high = torch.tensor(highs, device=raw.device, dtype=raw.dtype)  # (Q,)

        x_raw = raw[:, d].unsqueeze(1)  # (T,1)
        x_pub = pub[:, d].unsqueeze(1)  # (T,1)

        raw_in = (x_raw >= low.unsqueeze(0)) & (
            x_raw < high.unsqueeze(0))  # (T,Q)
        pub_in = (x_pub >= low.unsqueeze(0)) & (
            x_pub < high.unsqueeze(0))  # (T,Q)

        raw_cnt = torch.sum(raw_in, dim=0).to(raw.dtype)  # (Q,)
        pub_cnt = torch.sum(pub_in, dim=0).to(pub.dtype)  # (Q,)

        raw_counts_all.append(raw_cnt.unsqueeze(1))  # (Q,1)
        pub_counts_all.append(pub_cnt.unsqueeze(1))  # (Q,1)

    raw_q = torch.cat(raw_counts_all, dim=1)  # (Q, dim)
    pub_q = torch.cat(pub_counts_all, dim=1)  # (Q, dim)

    m = metric.lower()
    if m == "mae":
        return torch.mean(torch.abs(raw_q - pub_q)).item()
    if m == "mre":
        return point_mre(raw_q, pub_q)
    raise ValueError("metric must be 'mae' or 'mre' for count_query_metric")


def compute_error(raw_tensor, pub_tensor, metric="mae", query_num=100):
    """
    metric:
      "mae", "mre",
      "sum_mae", "sum_mre",
      "count_mae", "count_mre"
    """
    m = metric.lower()
    if m == "mae":
        return point_mae(raw_tensor, pub_tensor)
    if m == "mre":
        return point_mre(raw_tensor, pub_tensor)
    if m == "sum_mae":
        return sum_query_metric(raw_tensor, pub_tensor, query_num=query_num, metric="mae")
    if m == "sum_mre":
        return sum_query_metric(raw_tensor, pub_tensor, query_num=query_num, metric="mre")
    if m == "count_mae":
        return count_query_metric(raw_tensor, pub_tensor, query_num=query_num, metric="mae")
    if m == "count_mre":
        return count_query_metric(raw_tensor, pub_tensor, query_num=query_num, metric="mre")
    raise ValueError(
        "metric must be one of: mae/mre/sum_mae/sum_mre/count_mae/count_mre")


# ----------------------------
# Runners (GPU)
# ----------------------------
def run_naive_gpu(
    epsilon_list,
    sensitivity,
    raw_stream,
    window_size,
    round_,
    metric="mae",
    query_num=100,
    Flag_=0,
    device=None,
):
    if device is None:
        device = get_device()

    results = []

    if Flag_ == 0:
        for eps in epsilon_list:
            err_sum = 0.0
            for _ in range(round_):
                raw_t, pub_t = uniform_workload_gpu(
                    eps, sensitivity, raw_stream, window_size, device)
                err_sum += compute_error(raw_t, pub_t,
                                         metric=metric, query_num=query_num)

            err_avg = err_sum / round_
            print(f"epsilon: {eps} Done! metric={metric}, value={err_avg}")
            results.append(err_avg)
        print("uniform DONE!")
    else:
        for w in window_size:
            err_sum = 0.0
            for _ in range(round_):
                raw_t, pub_t = uniform_workload_gpu(
                    epsilon_list, sensitivity, raw_stream, w, device)
                err_sum += compute_error(raw_t, pub_t,
                                         metric=metric, query_num=query_num)

            err_avg = err_sum / round_
            print(f"window size: {w} Done! metric={metric}, value={err_avg}")
            results.append(err_avg)
        print("uniform DONE!")

    return results


# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Using device:", device)

    sensitivity = 1
    epsilon_list = [1.0]
    window_size = 120
    round_ = 5

    # metric :"mae", "mre",
    metric = "mae"
    query_num = 100

    datasets = ["covid19", ]

    for ds in datasets:
        raw_stream = data_reader(ds)
        error_ = run_naive_gpu(
            epsilon_list=epsilon_list,
            sensitivity=sensitivity,
            raw_stream=raw_stream,
            window_size=window_size,
            round_=round_,
            metric=metric,
            query_num=query_num,
            device=device,
        )
        print(f"Uniform result ({metric}):", error_)
