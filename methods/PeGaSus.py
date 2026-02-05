#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
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
# GPU Laplace helper
# ----------------------------
@torch.no_grad()
def laplace_scalar(scale, device):
    dist = torch.distributions.Laplace(
        loc=torch.tensor(0.0, device=device),
        scale=torch.tensor(float(scale), device=device),
    )
    return dist.sample(()).item()


# ----------------------------
# Metrics (GPU)
# ----------------------------
@torch.no_grad()
def point_mae(raw, pub):
    return torch.mean(torch.abs(raw - pub)).item()


@torch.no_grad()
def point_mre(raw, pub):
    abs_diff = torch.abs(raw - pub)
    denom = torch.abs(raw)
    mask0 = denom == 0
    rel = torch.empty_like(abs_diff)
    rel[mask0] = torch.abs(pub)[mask0]
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


def compute_error(raw, pub, metric="mae", query_num=100):
    """
    metric:
      "mae", "mre",
      "sum_mae", "sum_mre",
      "count_mae", "count_mre"
    """
    m = metric.lower()
    if m == "mae":
        return point_mae(raw, pub)
    if m == "mre":
        return point_mre(raw, pub)
    if m == "sum_mae":
        return sum_query_metric(raw, pub, query_num=query_num, metric="mae")
    if m == "sum_mre":
        return sum_query_metric(raw, pub, query_num=query_num, metric="mre")
    if m == "count_mae":
        return count_query_metric(raw, pub, query_num=query_num, metric="mae")
    if m == "count_mre":
        return count_query_metric(raw, pub, query_num=query_num, metric="mre")
    raise ValueError(
        "metric must be one of: mae/mre/sum_mae/sum_mre/count_mae/count_mre")


# ----------------------------
# Pegasus (GPU) core
# ----------------------------
class PegasusGPU:
    DEBUG_FORCE_NEW_GROUP_EACH_TIMESTAMP = False

    # smoother
    AVG_SMOOTHER = 0
    MEDIAN_SMOOTHER = 1
    JS_SMOOTHER = 2
    USE_SMOOTHER = MEDIAN_SMOOTHER

    class PegasusDataDim:

        def __init__(self, init_theta):
            self.idx_last_group = []
            self.trueStream_last_group = []
            self.perturbedStream_last_group_stat = []
            self.last_group_closed = True
            self.noisy_theta_prev = float(init_theta)

    def __init__(self, para_eps, para_w, para_d, para_sen, device=None):
        self.device = device if device is not None else get_device()

        self.eps_per_ts = float(para_eps) / float(para_w)
        self.eps_p = self.eps_per_ts * 0.8
        self.eps_g = self.eps_per_ts * 0.2
        self.sensitivity = float(para_sen)

        self.eps_g = max(self.eps_g, 1e-12)
        self.init_theta = 5.0 / self.eps_g

        self.no_dim = int(para_d)
        self.tempDataDim = [self.PegasusDataDim(
            self.init_theta) for _ in range(self.no_dim)]

    def perturber(self, c_t, eps_p):

        eps_p = max(float(eps_p), 1e-12)
        scale = self.sensitivity / eps_p
        return float(c_t) + laplace_scalar(scale, self.device)

    def dev(self, trueStream_last_group, c_t):

        avg = (sum(trueStream_last_group) + float(c_t)) / \
            (len(trueStream_last_group) + 1.0)
        dev_val = 0.0
        for c in trueStream_last_group:
            dev_val += abs(float(c) - avg)
        dev_val += abs(float(c_t) - avg)
        return dev_val

    def grouper(self, t, c_t, tempDataDim):

        if self.DEBUG_FORCE_NEW_GROUP_EACH_TIMESTAMP or tempDataDim.last_group_closed:
            # new group
            tempDataDim.idx_last_group.clear()
            tempDataDim.idx_last_group.append(int(t))
            tempDataDim.trueStream_last_group.clear()
            tempDataDim.trueStream_last_group.append(float(c_t))
            tempDataDim.perturbedStream_last_group_stat.clear()
            tempDataDim.last_group_closed = False

            lamdba_thres = 4.0 / self.eps_g
            scale_theta = self.sensitivity * lamdba_thres
            noisy_theta = float(self.init_theta) + \
                laplace_scalar(scale_theta, self.device)
        else:
            noisy_theta = float(tempDataDim.noisy_theta_prev)

            dev_val = self.dev(tempDataDim.trueStream_last_group, c_t)
            lamdba_dev = 8.0 / self.eps_g
            scale_dev = self.sensitivity * lamdba_dev
            noisy_dev = float(dev_val) + laplace_scalar(scale_dev, self.device)

            if abs(noisy_dev) < abs(noisy_theta):
                tempDataDim.idx_last_group.append(int(t))
                tempDataDim.trueStream_last_group.append(float(c_t))
                tempDataDim.last_group_closed = False
            else:
                # start new group
                tempDataDim.idx_last_group.clear()
                tempDataDim.idx_last_group.append(int(t))
                tempDataDim.trueStream_last_group.clear()
                tempDataDim.trueStream_last_group.append(float(c_t))
                tempDataDim.perturbedStream_last_group_stat.clear()
                tempDataDim.last_group_closed = True

        tempDataDim.noisy_theta_prev = float(noisy_theta)
        return tempDataDim.idx_last_group

    def averageSmoother(self, sanStream_last_group):
        return float(sum(sanStream_last_group) / max(len(sanStream_last_group), 1))

    def medianSmoother(self, sanStream_last_group):

        x = torch.tensor(sanStream_last_group,
                         device=self.device, dtype=torch.float32)
        return float(torch.mean(x).item())

    def jsSmoother(self, sanStream_last_group):
        avg = self.averageSmoother(sanStream_last_group)
        group_size = max(len(sanStream_last_group), 1)
        noisy_c_t = float(sanStream_last_group[-1])
        return float((noisy_c_t - avg) / group_size + avg)

    def smoother(self, sanStream_last_group):
        if self.USE_SMOOTHER == self.AVG_SMOOTHER:
            return self.averageSmoother(sanStream_last_group)
        elif self.USE_SMOOTHER == self.MEDIAN_SMOOTHER:
            return self.medianSmoother(sanStream_last_group)
        elif self.USE_SMOOTHER == self.JS_SMOOTHER:
            return self.jsSmoother(sanStream_last_group)
        return self.medianSmoother(sanStream_last_group)

    @torch.no_grad()
    def run(self, org_stream):
        """
        org_stream: list[list] or np.ndarray, shape (T, dim)
        return：pub_tensor (T, dim) on device
        """
        raw = torch.tensor(org_stream, device=self.device, dtype=torch.float32)
        T, dim = raw.shape
        assert dim == self.no_dim, f"dim mismatch: data dim={dim}, pegasus dim={self.no_dim}"

        pub = torch.zeros((T, dim), device=self.device, dtype=torch.float32)

        for t in range(T):
            for d in range(dim):
                tempData = self.tempDataDim[d]
                c_t = float(raw[t, d].item())

                # perturb
                san_c_t = self.perturber(c_t, self.eps_p)

                # group update
                self.grouper(t, c_t, tempData)

                # append perturbed value for smoother
                tempData.perturbedStream_last_group_stat.append(float(san_c_t))

                # smooth and output
                smoothed = self.smoother(
                    tempData.perturbedStream_last_group_stat)
                pub[t, d] = float(smoothed)

        return raw, pub


# ----------------------------
# Runners (GPU)
# ----------------------------
def run_pegasus_gpu(
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
    dim = len(raw_stream[0])

    if Flag_ == 0:
        for eps in epsilon_list:
            err_sum = 0.0
            for _ in range(int(round_)):
                mech = PegasusGPU(eps, window_size, dim,
                                  sensitivity, device=device)
                raw_t, pub_t = mech.run(raw_stream)
                err_sum += compute_error(raw_t, pub_t,
                                         metric=metric, query_num=query_num)

            err_avg = err_sum / float(round_)
            print(f"epsilon: {eps} Done! metric={metric}, value={err_avg}")
            results.append(err_avg)
        print("Pegasus (GPU) DONE!")
    else:
        for w in window_size:
            err_sum = 0.0
            for _ in range(int(round_)):
                mech = PegasusGPU(epsilon_list, w, dim,
                                  sensitivity, device=device)
                raw_t, pub_t = mech.run(raw_stream)
                err_sum += compute_error(raw_t, pub_t,
                                         metric=metric, query_num=query_num)

            err_avg = err_sum / float(round_)
            print(f"window size: {w} Done! metric={metric}, value={err_avg}")
            results.append(err_avg)
        print("Pegasus (GPU) DONE!")

    return results


# ----------------------------
# main (test)
# ----------------------------
if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Using device:", device)
    datasets = ["covid19", ]
    for ds in datasets:
        raw_stream = data_reader(ds)

        epsilon_list = [1.0]
        sensitivity = 1
        window_size = 100
        round_ = 1

        # metric ："mae","mre",
        metric = "mae"
        query_num = 200

        res = run_pegasus_gpu(
            epsilon_list=epsilon_list,
            sensitivity=sensitivity,
            raw_stream=raw_stream,
            window_size=window_size,
            round_=round_,
            metric=metric,
            query_num=query_num,
            device=device,
        )
        print("Pegasus result:", res)
