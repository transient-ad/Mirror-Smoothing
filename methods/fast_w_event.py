import numpy as np
import math
import random

import torch
from methods.data_process import data_reader

para_ratio = 0.075
para_windowPID = 5

def publish(epsilon, sensitivity, raw_stream):


    P = 100.0
    Q = 100000.0
    R = 1000000.0
    K = 0.0
    Cp = 0.9
    Ci = 0.1
    Cd = 0.0
    theta = 5.0
    xi = 0.1
    minIntvl = 1
    publish_num = 0

    query = [0 for i in range(len(raw_stream))]
    predict = [0 for i in range(len(raw_stream))]
    publish = [0 for i in range(len(raw_stream))]
    m = para_ratio * len(raw_stream)
    if m <= 0:
        m = 1

    eps_per_sample = epsilon / m

    interval = 1

    nextquery = max(1, para_windowPID) + interval - 1

    for i in range(len(raw_stream)):
            # begin: Algorithm 3
            if (i == 0):
                publish[i] = raw_stream[i] + np.random.laplace(loc = 0,scale = sensitivity / eps_per_sample)
                publish_num += 1
                query[i] = 1
            else:
                # Line 2 'obtain prior estimate from prediction'
                predct = 0
                for ii in range(len(query)):
                    if query[len(query) - 1 - ii] == 1:
                        predct = publish[len(query) - 1 - ii]
                        break

                P += Q
                predict[i] = int(predct)
                card = 0
                for ii in range(len(query)):
                    if query[ii] == 1:
                        card += 1

                # if (Mechansim.TRUNCATE)
                #     this.predict[i] = Mechansim.truncate(predct);
                # // (not covered in algo just in text) exception handling; integral value can only be calculated when there have been >= windowPID samples
                if ((card < para_windowPID) and (card < m)):

                    publish[i] = raw_stream[i] + np.random.laplace(loc = 0,scale = sensitivity / eps_per_sample)
                    publish_num += 1
                    query[i] = 1
                    # correctKF(i, predct)
                    K = P / (P + R)
                    correct = predct + K * (publish[i] - predct)
                    publish[i] = correct
                    P = 1.0 - K

                    # Line 3 'if sampling point and numSamples < M'
                elif ((i == nextquery) and (card < m)):
                    # Line 4 'perturb'
                    publish[i] = raw_stream[i] + np.random.laplace(loc = 0,scale = sensitivity / eps_per_sample)
                    publish_num += 1
                    query[i] = 1
                    # line 6: obtain posterior estimate from correction
                    # correctKF(i, predct)
                    K = P / (P + R)
                    correct = predct + K * (publish[i] - predct)
                    publish[i] = correct
                    P = 1.0 - K

                    # line 8: adjust sampling rate by adaptive sampling =>
                    # begin: Algo. 9
                    # ratio = PID(i); // PID error


                    sum = 0.0
                    lastValue = 0.0
                    change = 0.0
                    timeDiff = 0
                    next = i
                    for j in range(para_windowPID -1, -1, -1):
                        index = 0
                        for index in range(next, -1, -1):

                            if (query[index] == 1):
                                next = index - 1
                                break
                        
                        
                        if (j == para_windowPID - 1):
                            # Feedback error (cf. Def. 4)
                            lastValue = abs(publish[index] - predict[index]) / (1.0 * max(publish[index], 1))
                            change = abs(publish[index] - predict[index]) / (1.0 * max(publish[index], 1))
                            timeDiff = index
                        
                        if (j == para_windowPID - 2):
                            change -= abs(publish[index] - predict[index]) / (1.0 * max(publish[index], 1))
                            timeDiff -= index
                        
                        sum += abs(publish[index] - predict[index]) / (1.0 * max(publish[index], 1))
                    
                    # Eq. (29) . last value = e_k_n = feedback error
                    ratio = Cp * lastValue + Ci * sum + Cd * change / timeDiff
                    
                    try:
                        deltaI = (int) (theta * (1.0 - math.exp((ratio - xi) / xi))); # (32) - max fehlt
                    
                    except Exception as e:
                        # print(Cp, lastValue, Ci, sum, Cd, change, timeDiff, theta, ratio, xi)
                        deltaI = 0

                    interval += deltaI
                    if (interval < minIntvl):
                        interval = minIntvl
                    
                    nextquery += interval
                    # end: Algo 9
                else:
                    # line 10
                    publish[i] = int(predct)
                
            
            # end: Algorithm 3

        
    return publish, publish_num

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
def run_fast_gpu(
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
    Fast (w/ event) GPU wrapper — algorithm runs on CPU (NumPy), metrics computed on GPU (PyTorch).

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
                # Build result matrix: same logic as original run_fast
                res = [[0.0 for _ in range(dim)] for __ in range(len(raw_stream))]
                for t in range(0, len(raw_stream), window_size + 1):
                    end_index = min(t + window_size, len(raw_stream) - 1) + 1
                    for d in range(dim):
                        substream_of_dimension = []
                        for i in range(t, end_index - 1):
                            substream_of_dimension.append(raw_stream[i][d])
                        published, _ = publish(eps, sensitivity, substream_of_dimension)
                        for i, val in enumerate(published):
                            if t + i < len(res):
                                res[t + i][d] = val

                raw_t = torch.tensor(raw_stream, device=device, dtype=torch.float32)
                pub_t = torch.tensor(res, device=device, dtype=torch.float32)

                err_sum += compute_error(raw_t, pub_t, metric=metric, query_num=query_num)

            err_avg = err_sum / float(round_)
            print(f"epsilon: {eps} Done! metric={metric}, value={err_avg}")
            results.append(err_avg)

        print("FAST (GPU) DONE!")
    else:
        for w in window_size:
            err_sum = 0.0
            for _ in range(int(round_)):
                res = [[0.0 for _ in range(dim)] for __ in range(len(raw_stream))]
                for t in range(0, len(raw_stream), w + 1):
                    end_index = min(t + w, len(raw_stream) - 1) + 1
                    for d in range(dim):
                        substream_of_dimension = []
                        for i in range(t, end_index - 1):
                            substream_of_dimension.append(raw_stream[i][d])
                        published, _ = publish(epsilon_list, sensitivity, substream_of_dimension)
                        for i, val in enumerate(published):
                            if t + i < len(res):
                                res[t + i][d] = val

                raw_t = torch.tensor(raw_stream, device=device, dtype=torch.float32)
                pub_t = torch.tensor(res, device=device, dtype=torch.float32)

                err_sum += compute_error(raw_t, pub_t, metric=metric, query_num=query_num)

            err_avg = err_sum / float(round_)
            print(f"window size: {w} Done! metric={metric}, value={err_avg}")
            results.append(err_avg)

        print("FAST (GPU) DONE!")

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

    error_list = run_fast_gpu(
        epsilon_list=epsilon_list,
        sensitivity=sensitivity,
        raw_stream=raw_stream,
        window_size=window_size,
        round_=round_,
        metric=metric,
        device=device,
    )
    print("Fast result:", error_list)
