#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AdaPub — Runner aligned with Naive/Uniform script style.

Supported metrics (same names as your Naive runner):
  "mae", "mre",
  "sum_mae", "sum_mre",
  "count_mae", "count_mre"

Notes:
- AdaPub core is implemented with NumPy (CPU).
- Metrics are computed on GPU (PyTorch) for consistency with other methods.
"""

import random
import sys
import math
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
# GPU metrics (copied/compatible with your Naive implementation)
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


@torch.no_grad()
def sum_query_metric(raw, pub, query_num=100, metric="mae"):
    T, dim = raw.shape
    if query_num <= 0:
        return 0.0

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

    raw_ps = torch.zeros((T + 1, dim), device=raw.device, dtype=raw.dtype)
    pub_ps = torch.zeros((T + 1, dim), device=pub.device, dtype=pub.dtype)
    raw_ps[1:] = torch.cumsum(raw, dim=0)
    pub_ps[1:] = torch.cumsum(pub, dim=0)

    raw_sum = raw_ps[idx_b] - raw_ps[idx_a]
    pub_sum = pub_ps[idx_b] - pub_ps[idx_a]

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

    max_vals = torch.max(raw, dim=0).values

    raw_counts_all = []
    pub_counts_all = []

    for d in range(dim):
        maxv = float(max_vals[d].item())
        lows, highs = [], []
        while len(lows) < query_num:
            a = random.uniform(0.0, maxv)
            b = random.uniform(0.0, maxv)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            lows.append(a)
            highs.append(b)

        low = torch.tensor(lows, device=raw.device, dtype=raw.dtype)
        high = torch.tensor(highs, device=raw.device, dtype=raw.dtype)

        x_raw = raw[:, d].unsqueeze(1)
        x_pub = pub[:, d].unsqueeze(1)

        raw_in = (x_raw >= low.unsqueeze(0)) & (x_raw < high.unsqueeze(0))
        pub_in = (x_pub >= low.unsqueeze(0)) & (x_pub < high.unsqueeze(0))

        raw_cnt = torch.sum(raw_in, dim=0).to(raw.dtype)
        pub_cnt = torch.sum(pub_in, dim=0).to(pub.dtype)

        raw_counts_all.append(raw_cnt.unsqueeze(1))
        pub_counts_all.append(pub_cnt.unsqueeze(1))

    raw_q = torch.cat(raw_counts_all, dim=1)
    pub_q = torch.cat(pub_counts_all, dim=1)

    m = metric.lower()
    if m == "mae":
        return torch.mean(torch.abs(raw_q - pub_q)).item()
    if m == "mre":
        return point_mre(raw_q, pub_q)
    raise ValueError("metric must be 'mae' or 'mre' for count_query_metric")


def compute_error(raw_tensor, pub_tensor, metric="mae", query_num=100):
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
# AdaPub core (NumPy CPU)
# ----------------------------
DEBUG_FORCE_NEW_GROUP_EACH_TIMESTAMP = False
DEBUG_ENABLE_DIMENSIONGROUPING = False
DEBUG_ENABLE_TIMEGROUPINGFILTER = True


def lap_arr(v, epsilon, sensitivity):
    """
    Add Laplace noise to a vector v (one timestamp).
    IMPORTANT: noise should be sampled independently per dimension.
    """
    v = np.asarray(v, dtype=float)
    noise = np.random.laplace(loc=0.0, scale=float(
        sensitivity) / float(epsilon), size=v.shape)
    return (v + noise).tolist()


class Adapub:
    seed = 123456
    g = 20  # number of hash functions, as in paper
    SHARE_EPS_P = 0.8
    SHARE_EPS_C = 0.2
    KP = 0.9
    KI = 0.1
    KD = 0.0
    GAMMA_FEEDBACK_ERROR = 1

    def __init__(self, epsilon, window_size, dim, sensitivity):
        self.epsilon = float(epsilon)
        self.window_size = int(window_size)
        self.dim = int(dim)
        self.sensitivity = float(sensitivity)

        # buffers
        self.partition_buffer = [0 for _ in range(self.g + 1)]
        self.new_group = []

    def run(self, org_stream):
        """
        org_stream: list[list[float]] length T, each element dim-d vector
        returns: sanitized_stream list[list[float]]
        """
        T = len(org_stream)
        self.new_group = [0 for _ in range(T)]

        eps_p = self.SHARE_EPS_P * self.epsilon
        eps_c = self.SHARE_EPS_C * self.epsilon
        if not DEBUG_ENABLE_TIMEGROUPINGFILTER:
            eps_p = self.epsilon
            eps_c = 0.0

        # Evenly distribute eps_p to each timestamp in the window (as in the original code)
        lambda_perturb = 1.0 / (eps_p / float(self.window_size))

        sanitized_stream = []

        # t=0: release uniformly (no prior for grouping)
        san_t0 = lap_arr(org_stream[0], epsilon=1.0 /
                         lambda_perturb, sensitivity=self.sensitivity)
        sanitized_stream.append(san_t0)

        # Initialize one cluster per dimension
        clusters = [self.Cluster(T, d, self) for d in range(self.dim)]

        for t in range(1, T):
            # 1) Partition dimensions based on last release (or dummy partition)
            last_release = sanitized_stream[t - 1]
            if not DEBUG_ENABLE_DIMENSIONGROUPING:
                groups = self.get_dummy_partition(last_release)
            else:
                groups = self.get_partition(last_release)

            # 2) Laplace perturbation on grouped sums (dimension grouping)
            san_t = self.laplace_perturbation(
                groups, org_stream[t], lambda_perturb)

            # 3) Time grouping + median smoothing (optional)
            if DEBUG_ENABLE_TIMEGROUPINGFILTER:
                for d in range(self.dim):
                    clusters[d].cluster(
                        t, eps_c, org_stream, sanitized_stream, san_t)
                    begin = clusters[d].current_group_begin_timestamp
                    san_t[d] = clusters[d].median_smoother(
                        sanitized_stream, begin, t - 1, san_t[d])

            sanitized_stream.append(san_t)

        return sanitized_stream

    def get_dummy_partition(self, last_release):
        # One dimension per group
        return {d: [d] for d in range(len(last_release))}

    def get_partition(self, last_release):
        d = len(last_release)
        groups = {}

        # set sentinel
        self.partition_buffer[0] = -sys.maxsize - 1

        max_count = max(last_release)
        min_count = min(last_release)

        # sample pivots
        for i in range(1, self.g + 1):
            if min_count == max_count:
                pivot = min_count
            else:
                pivot = int(np.random.randint(
                    int(min_count), max(1, int(max_count)), 1)[0])
            self.partition_buffer[i] = pivot

        self.partition_buffer.sort(reverse=True)

        # assign dims to groups
        for dim in range(d):
            old_release = last_release[dim]
            for p in range(len(self.partition_buffer)):
                group_key = self.partition_buffer[p]
                if old_release > group_key:
                    if group_key not in groups:
                        groups[group_key] = []
                    groups[group_key].append(dim)
                    break

        return groups

    def laplace_perturbation(self, groups, org_values, lambda_t):
        san_t = [0.0 for _ in range(len(org_values))]

        for key in groups.keys():
            group = groups[key]
            group_size = len(group)
            if group_size <= 0:
                continue

            s = 0.0
            for dim in group:
                s += float(org_values[dim])

            # noise on sum, then divide by |group| (as in your code comment)
            san_sum = s + np.random.laplace(0.0, self.sensitivity * lambda_t)
            san_avg = san_sum / float(group_size)

            for dim in group:
                san_t[dim] = float(san_avg)

        return san_t

    def dev(self, org_stream, t, dim, t_group_start):
        avg = 0.0
        for i in range(t_group_start, t + 1):
            avg += float(org_stream[i][dim])
        n = t - t_group_start + 1
        avg /= float(max(n, 1))

        dev = 0.0
        for i in range(t_group_start, t + 1):
            dev += abs(float(org_stream[i][dim]) - avg)
        return dev

    def feedback_error(self, last_release, current_perturbed_value):
        denom = max(float(current_perturbed_value),
                    float(self.GAMMA_FEEDBACK_ERROR))
        return abs(float(last_release) - float(current_perturbed_value)) / denom

    def pid_error(self, t, cluster, san_stream, intermediate_san_t):
        fb = self.feedback_error(
            san_stream[t - 1][cluster.my_dim], intermediate_san_t[cluster.my_dim])
        cluster.feedback_errors[t] = fb

        pid = fb * self.KP
        pid += self.KI * cluster.feedback_error_integral(t)
        pid += self.KD * (fb - cluster.feedback_errors[t - 1]) / 1.0
        return pid

    class Cluster:
        def __init__(self, num_timestamps, dim, obj):
            self.current_group_begin_timestamp = 0
            self.is_closed = False
            self.feedback_errors = [0.0 for _ in range(num_timestamps)]
            self.my_dim = int(dim)
            self.obj = obj

        def cluster(self, t, eps_c, org_stream, san_stream, san_stream_t):
            # Compute theta
            if t == 0:
                theta = 1.0 / max(self.obj.epsilon, 1e-12)
            else:
                delta_err = self.obj.pid_error(
                    t, self, san_stream, san_stream_t)
                theta = max(1.0, (delta_err * delta_err) /
                            max(self.obj.epsilon, 1e-12))

            if self.is_closed:
                self.current_group_begin_timestamp = t
                self.is_closed = False
                if DEBUG_FORCE_NEW_GROUP_EACH_TIMESTAMP:
                    self.is_closed = True
                self.obj.new_group[t] = 1
                return

            self.obj.new_group[t] = 0

            # Compute noisy deviation for deciding whether to close/open a new group
            dev = self.obj.dev(org_stream, t, self.my_dim,
                               self.current_group_begin_timestamp)
            if eps_c <= 0:
                noisy_dev = dev
            else:
                lam_dev = 2.0 * float(self.obj.window_size) / float(eps_c)
                noisy_dev = max(dev + np.random.laplace(0.0,
                                self.obj.sensitivity * lam_dev), 0.0)

            # Decision: if noisy_dev >= theta, close group and start a new one
            if noisy_dev >= theta:
                self.current_group_begin_timestamp = t
                self.is_closed = True

        def feedback_error_integral(self, t):
            n = t - self.current_group_begin_timestamp
            if n <= 0:
                return 0.0
            s = 0.0
            for i in range(self.current_group_begin_timestamp, t):
                s += float(self.feedback_errors[i])
            return s / float(n)

        def median_smoother(self, san_stream, begin, end, san_t_dim_value):
            vals = [float(san_t_dim_value)]
            for i in range(begin, end + 1):
                vals.append(float(san_stream[i][self.my_dim]))
            vals.sort()
            k = len(vals)
            if k % 2 == 0:
                return 0.5 * (vals[k // 2] + vals[k // 2 - 1])
            else:
                return vals[k // 2]


# ----------------------------
# Runner (aligned with Naive)
# ----------------------------
def adapub_workload_cpu(epsilon, sensitivity, raw_stream, window_size):
    """
    Run AdaPub on CPU and return raw/pub tensors (on CPU).
    raw_stream: list[list] shape (T, dim)
    """
    raw = np.asarray(raw_stream, dtype=np.float32)
    T, dim = raw.shape

    mech = Adapub(epsilon=epsilon, window_size=window_size,
                  dim=dim, sensitivity=sensitivity)
    pub_list = mech.run(raw_stream)
    pub = np.asarray(pub_list, dtype=np.float32)

    return raw, pub


def run_adapub_gpu(
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
    """
    Same signature style as run_naive_gpu, but AdaPub core runs on CPU (NumPy).
    Metrics are computed on GPU for consistency.
    """
    if device is None:
        device = get_device()

    results = []

    if Flag_ == 0:
        for eps in epsilon_list:
            err_sum = 0.0
            for _ in range(round_):
                raw_np, pub_np = adapub_workload_cpu(
                    eps, sensitivity, raw_stream, window_size)

                raw_t = torch.tensor(
                    raw_np, device=device, dtype=torch.float32)
                pub_t = torch.tensor(
                    pub_np, device=device, dtype=torch.float32)

                err_sum += compute_error(raw_t, pub_t,
                                         metric=metric, query_num=query_num)

            err_avg = err_sum / float(round_)
            print(f"epsilon: {eps} Done! metric={metric}, value={err_avg}")
            results.append(err_avg)

        print("AdaPub DONE!")
    else:
        for w in window_size:
            err_sum = 0.0
            for _ in range(round_):
                raw_np, pub_np = adapub_workload_cpu(
                    epsilon_list, sensitivity, raw_stream, w)

                raw_t = torch.tensor(
                    raw_np, device=device, dtype=torch.float32)
                pub_t = torch.tensor(
                    pub_np, device=device, dtype=torch.float32)

                err_sum += compute_error(raw_t, pub_t,
                                         metric=metric, query_num=query_num)

            err_avg = err_sum / float(round_)
            print(f"window size: {w} Done! metric={metric}, value={err_avg}")
            results.append(err_avg)

        print("AdaPub DONE!")

    return results


# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Using device:", device)

    sensitivity = 1
    epsilon_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    window_size = 120
    round_ = 1

    metric = "mae"      # "mae", "mre",
    query_num = 100

    raw_stream = data_reader("covid19")

    error_list = run_adapub_gpu(
        epsilon_list=epsilon_list,
        sensitivity=sensitivity,
        raw_stream=raw_stream,
        window_size=window_size,
        round_=round_,
        metric=metric,
        query_num=query_num,
        device=device,
    )
    print(f"AdaPub result ({metric}):", error_list)
