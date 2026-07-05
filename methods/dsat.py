import numpy as np
import math
import random

import torch
from methods.data_process import data_reader

DEBUG = False
# dim = 1
# window_size = 100
# para_eps = 0.1
para_ratio = 0.075 * 2
para_windowPID = 5
USE_PERTURBED_NUMBER_TUPLES = True
SHARE_EPS_NO_TUPLES = 0.1
NO_TUPELS_n = 500000
OWN_EPS_1_FRACTION = 0.5
OWN_EPS_1 = True
THRESHOLD_T = 0.025
TOLERANCE_DELTA = 0.05
PROP_GAIN_THETA = 0.5
USE_NON_PRIVATE_SAMPLING = False
skipping_points_M = 2

def lap_arr(v, epsilon, sensitivity):
    tmp = np.random.laplace(loc=0,scale=sensitivity/epsilon)
    new_arr = []
    for i in range(len(v)):
        new_arr.append(v[i] + tmp)
    
    return new_arr


def avg_dissimilarity(last_published, org_stream_t, lambda_t, sensitivity):
    avg = 0
    # calculate dissimilarity
    for location in range(len(last_published)):
        
        avg += abs(last_published[location] - org_stream_t[location])
    
    avg /= len(last_published) # normalize by dimensionality

    if (USE_NON_PRIVATE_SAMPLING == False):
        avg += np.random.laplace(loc = 0, scale = sensitivity * lambda_t) # add noise
    
    return avg


def computeK(cuttoff_point_c, dimensionality_U, delta, no_time_points_N, no_tuples_n):
    # Theorem 5.4
    root1 = no_tuples_n * no_tuples_n
    root2 = (8 * delta * delta + 32 * cuttoff_point_c * cuttoff_point_c * delta * delta) / (dimensionality_U * cuttoff_point_c * cuttoff_point_c)
    root = round(math.pow(root1 * root2, 1.0 / 3.0))
    return min(root, 1 - cuttoff_point_c / no_time_points_N)

 

def dsat_workflow(epsilon, sensitivity, raw_stream, window_size, dim):

    # res = [[0 for j in range(dim)] for i in range(len(raw_stream))]

    publish_num = 0
    length_N = len(raw_stream)
    dimensionality_U = dim        
    CUTOFFPOINT_RATION = para_ratio # each time stamp in window
    sanitized_stream = []
    # ArrayList<double[]> sanitized_stream = new ArrayList<double[]>(length_N)
    delta = sensitivity * 2 / dimensionality_U # sensitivity L1 distance
    cuttoff_point_c = math.ceil(window_size * CUTOFFPOINT_RATION)  # in paper for user-level: length_N * 0.01 in paper

    used_budgets_eps_2 = [0 for i in range(length_N)]
    used_budgets_eps_1 = [0 for i in range(length_N)]
    samplingpoint = [0 for i in range(length_N)]

    # repair function
    no_tuples_n = 0
    eps_for_determining_n = 0
    reducedEpsFirstWindow = 0
    if (USE_PERTURBED_NUMBER_TUPLES):
        eps_for_determining_n = epsilon * SHARE_EPS_NO_TUPLES
        reducedEpsFirstWindow =eps_for_determining_n
        lambda_t = 1 / eps_for_determining_n

        sum = 0
        for ii in range(dim):
            sum += raw_stream[0][ii]

        san_sum = sum + np.random.laplace(loc = 0, scale = sensitivity * lambda_t)
        no_tuples_n = san_sum
        if (no_tuples_n == 0):
            no_tuples_n = NO_TUPELS_n
        
    else:
        no_tuples_n = NO_TUPELS_n
    

    k = computeK(cuttoff_point_c, dimensionality_U, delta, window_size, no_tuples_n) # before: length N instead of k


    eps_1_per_window_paper  = epsilon * k
    eps_1_per_window_paper_first  = (epsilon - reducedEpsFirstWindow) * k

    eps_1_per_window_own = epsilon * OWN_EPS_1_FRACTION
    eps_1_per_window_own_frist = (epsilon - reducedEpsFirstWindow) * OWN_EPS_1_FRACTION

    eps_1_per_window=0
    noisy_threshold=0
    eps_1_per_window_first=0
    # @SuppressWarnings("unused")
    noisy_threshold_first=0
    if (OWN_EPS_1):
        eps_1_per_window  =eps_1_per_window_own
        eps_1_per_window_first=eps_1_per_window_own_frist
        noisy_threshold = THRESHOLD_T + np.random.laplace(loc = 0, scale = 2 * delta / (eps_1_per_window_first/cuttoff_point_c))
        used_budgets_eps_1[0] = eps_1_per_window_first/cuttoff_point_c - eps_for_determining_n
    else:
        eps_1_per_window = eps_1_per_window_paper
        eps_1_per_window_first=eps_1_per_window_paper_first
        noisy_threshold = THRESHOLD_T + np.random.laplace(loc = 0, scale = 2 * delta / (eps_1_per_window_first/cuttoff_point_c))
        used_budgets_eps_1[0] = eps_1_per_window_first/cuttoff_point_c - eps_for_determining_n  
    

    eps_2_per_window = epsilon - eps_1_per_window  # for sanitizing
    eps_2_per_window_first = epsilon - eps_1_per_window_first # for sanitizing
    total_budget_spent_eps_2 = 0 # eps_1

    no_sampling_points_window = 0
    last_realize = []
    for t in range(length_N):
        org_stream_t = raw_stream[t]

        if (t <= window_size):
        # Algorithm 2
            if (t == 0):
                scale = 1 / (eps_2_per_window_first / cuttoff_point_c)
                last_realize = lap_arr(org_stream_t, 1 / scale, sensitivity=sensitivity)
                publish_num += 1
                sanitized_stream.append(last_realize)
                total_budget_spent_eps_2 += eps_2_per_window_first / cuttoff_point_c
                used_budgets_eps_2[t] = eps_2_per_window_first / cuttoff_point_c
                samplingpoint[t] = 1
                no_sampling_points_window += 1
            else:
                if (t <= skipping_points_M): # Line 2
                    sanitized_stream.append(last_realize)
                    samplingpoint[t] = False

                else:
                    # Line 4
                    if (no_sampling_points_window >= cuttoff_point_c): # ">=" is important due to line 47
                        sanitized_stream.append(last_realize)
                        samplingpoint[t] = 0
                    else: # cf. Algo 1 ("continue")
                        # Line 5
                        noisy_dist = avg_dissimilarity(org_stream_t, last_realize,
                                2 * cuttoff_point_c * delta / eps_1_per_window_first, sensitivity) # L1 distance
                        used_budgets_eps_1[t] = eps_1_per_window_first /cuttoff_point_c

                        # Line 6
                        feedback_error_E = abs(no_sampling_points_window / t - cuttoff_point_c / window_size) # use here already w instead of N
                        prop_e = abs(feedback_error_E - TOLERANCE_DELTA) / TOLERANCE_DELTA
                        prop_part_u = prop_e * PROP_GAIN_THETA
                        # Line 7+8: adapt threshold
                        if ((no_sampling_points_window / t - cuttoff_point_c / window_size) <= 0): # use here already w instead of N
                            noisy_threshold = max(0, noisy_threshold - prop_part_u)
                        else:
                            noisy_threshold = min(2, noisy_threshold + prop_part_u)
                        
                        # Line 9-12: decide whether to sample
                        if (noisy_dist >= noisy_threshold): # wir verschenken hier budget im ersten window, wenn wir nicht samplen, wegen zeile 94. man sollte hier auch mit eps_rm arbeiten?
                            # sample
                            last_realize = lap_arr(org_stream_t, (eps_2_per_window_first / cuttoff_point_c), sensitivity)
                            publish_num += 1
                            sanitized_stream.append(last_realize)
                            total_budget_spent_eps_2 += eps_2_per_window_first / cuttoff_point_c
                            no_sampling_points_window += 1
                            used_budgets_eps_2[t] = eps_2_per_window_first / cuttoff_point_c
                            samplingpoint[t] = 1

                        else:
                            sanitized_stream.append(last_realize)
                            samplingpoint[t] = 0
                        
                        # Line 14+15 -- should not happen in w event variant
                        if (t == length_N and no_sampling_points_window < cuttoff_point_c): # end of the time series
                            remaining_budget = eps_2_per_window_first - total_budget_spent_eps_2
                            last_realize = lap_arr(org_stream_t, (remaining_budget), sensitivity)
                            publish_num += 1
                            sanitized_stream.append(last_realize)
                            no_sampling_points_window += 1
                            used_budgets_eps_2[t] = remaining_budget
                            total_budget_spent_eps_2 = epsilon
                            samplingpoint[t] = 1

        else: # second window begins
            if (DEBUG):
                1
                # System.out.println(t-w-1 + " budget " + used_budgets_eps_2[t-w-1])
                # System.out.println(Arrays.toString(Arrays.copyOfRange(used_budgets_eps_2, t-w, t-1)))
            
            eps_spent = 0
            for ii in range(max(0, t - window_size), t):
                eps_spent += used_budgets_eps_2[ii]

            # eps_spent = Arrays.stream(Arrays.copyOfRange(used_budgets_eps_2, Math.max(0, t-w), t-1)).sum() # total_budget_spent_eps_2 - used_budgets_eps_2[t-w-1]
            eps_rm = eps_2_per_window - eps_spent
            # System.err.println("Use wrong eps_rm check!")
            if (eps_rm <= 0.00001): # rounding issues
                sanitized_stream.append(last_realize)
                samplingpoint[t] = False
            else:
                # line 7

                no_sampling_points_window = 0
                for ii in range(max(0, t - window_size), t):
                    if samplingpoint[ii] == 1:
                        no_sampling_points_window += 1
                # no_sampling_points_window = (int) Arrays.stream(Arrays.copyOfRange(samplingpoint, Math.max(0, t-w), t-1)).filter(x -> x.booleanValue()).count() # paper: (int) (total_budget_spent_eps_2/(eps_2_per_window / cuttoff_point_c)) //number publications current window
                # Line 8-11: -- analogous to above
                noisy_dist = avg_dissimilarity(org_stream_t, last_realize,
                        2 * cuttoff_point_c * delta / eps_1_per_window, sensitivity) # L1 distance
                # Line 6
                feedback_error_E = abs(no_sampling_points_window / t - cuttoff_point_c / window_size) # between target sampling rate and actual one
                prop_e = abs(feedback_error_E - TOLERANCE_DELTA) / TOLERANCE_DELTA
                prop_part_u = prop_e * PROP_GAIN_THETA
                # Line 7+8: adapt threshold
                if ((no_sampling_points_window / t - cuttoff_point_c / window_size) <= 0):
                    noisy_threshold = max(0, noisy_threshold - prop_part_u)
                else:
                    noisy_threshold = min(2, noisy_threshold + prop_part_u)
                
                # Line 9-12: decide whether to sample
                if (noisy_dist >= noisy_threshold):
                    # sample
                    last_realize = lap_arr(org_stream_t, (eps_2_per_window / cuttoff_point_c), sensitivity)
                    publish_num += 1
                    sanitized_stream.append(last_realize)
                    total_budget_spent_eps_2 += eps_2_per_window / cuttoff_point_c
                    used_budgets_eps_2[t] = eps_2_per_window / cuttoff_point_c
                    samplingpoint[t] = 1

                else:
                    sanitized_stream.append(last_realize)
                    samplingpoint[t] = 0

    return sanitized_stream


# ----------------------------
# Device + seed (GPU)
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
# GPU metric functions (aligned with Naive.py)
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

    idx_a = torch.tensor([x[0] for x in intervals], device=raw.device, dtype=torch.long)
    idx_b = torch.tensor([x[1] for x in intervals], device=raw.device, dtype=torch.long)

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
    raise ValueError("metric must be one of: mae/mre/sum_mae/sum_mre/count_mae/count_mre")


# ----------------------------
# GPU runner (main entry point for run_code.py)
# ----------------------------
def run_dsat_gpu(
    epsilon_list,
    sensitivity,
    raw_stream,
    window_size,
    round_,
    metric="mre",
    query_num=100,
    Flag_=0,
    device=None,
):
    """
    DSAT GPU wrapper — algorithm runs on CPU (NumPy), metrics computed on GPU (PyTorch).

    Parameters (aligned with run_naive_gpu / run_adapub_gpu):
        epsilon_list : list[float]
        sensitivity  : float
        raw_stream   : list[list] or np.ndarray, shape (T, dim)
        window_size  : int
        round_       : int
        metric       : str — "mae", "mre", "sum_mae", "sum_mre", "count_mae", "count_mre"
        query_num    : int — used only for sum_/count_ metrics
        Flag_        : int — 0 = iterate epsilon_list, 1 = iterate window_size (not used here)
        device       : torch.device or None
    """
    if device is None:
        device = get_device()

    dim = len(raw_stream[0])
    results = []

    if Flag_ == 0:
        for eps in epsilon_list:
            err_sum = 0.0
            for _ in range(int(round_)):
                published_result = dsat_workflow(eps, sensitivity, raw_stream, window_size, dim)

                raw_t = torch.tensor(raw_stream, device=device, dtype=torch.float32)
                pub_t = torch.tensor(published_result, device=device, dtype=torch.float32)

                err_sum += compute_error(raw_t, pub_t, metric=metric, query_num=query_num)

            err_avg = err_sum / float(round_)
            print(f"epsilon: {eps} Done! metric={metric}, value={err_avg}")
            results.append(err_avg)

        print("DSAT (GPU) DONE!")
    else:
        for w in window_size:
            err_sum = 0.0
            for _ in range(int(round_)):
                published_result = dsat_workflow(epsilon_list, sensitivity, raw_stream, w, dim)

                raw_t = torch.tensor(raw_stream, device=device, dtype=torch.float32)
                pub_t = torch.tensor(published_result, device=device, dtype=torch.float32)

                err_sum += compute_error(raw_t, pub_t, metric=metric, query_num=query_num)

            err_avg = err_sum / float(round_)
            print(f"window size: {w} Done! metric={metric}, value={err_avg}")
            results.append(err_avg)

        print("DSAT (GPU) DONE!")

    return results


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Using device:", device)

    raw_stream = data_reader('Uem')
    epsilon_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    sensitivity = 1
    window_size = 100
    round_ = 1
    metric = "mre"

    error_list = run_dsat_gpu(
        epsilon_list=epsilon_list,
        sensitivity=sensitivity,
        raw_stream=raw_stream,
        window_size=window_size,
        round_=round_,
        metric=metric,
        device=device,
    )
    print("DSAT result:", error_list)