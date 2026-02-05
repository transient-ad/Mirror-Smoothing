import numpy as np
import random
import pandas as pd

# ---------- Tunable parameters ----------
LEN = 10_000            # unified length for all synthetic streams
DIM = 10                # unified dimension for all streams
SEED = 0               # random seed (reproducible)

# Sparse Spike (Burst) params
BURST_PROB = 0.01       # probability of a burst at each time step
BURST_K_RANGE = (1, 3)  # number of affected dimensions in a burst
BURST_MAG_RANGE = (500, 3000)  # spike magnitude range (additive)

# Correlated Latent-Factor params
LATENT_AR = 0.98        # AR(1) coefficient for the latent factor
LATENT_SIGMA = 30.0     # innovation scale for latent factor
LATENT_LOAD_RANGE = (0.8, 1.2) # loading magnitude range for each dimension
# ---------------------------------------

np.random.seed(SEED)
random.seed(SEED)


def rand_gauss(mean_range, var_range):
    mu = random.uniform(*mean_range)
    var = random.uniform(*var_range)
    return np.random.normal(mu, var)


def rand_laplace(mean_range, var_range):
    mu = random.uniform(*mean_range)
    var = random.uniform(*var_range)
    return np.random.laplace(mu, var)


def sample_point(mean_range, var_range):
    """Sample one value from Gaussian/Laplace with 50/50 probability."""
    if random.random() < 0.5:
        return rand_gauss(mean_range, var_range)
    else:
        return rand_laplace(mean_range, var_range)


def gen_high_volatility(length, dim):
    """
    High Volatility Stream:
    - mean shifts randomly within [0, 3000]
    - variance/scale ranges within [50, 300]
    - each value drawn from either Gaussian or Laplace (50/50)
    """
    mean_range = (0, 3000)
    var_range = (50, 300)
    stream = np.zeros((length, dim), dtype=int)
    for t in range(length):
        for d in range(dim):
            stream[t, d] = int(sample_point(mean_range, var_range))
    return stream


def gen_low_volatility(length, dim):
    """
    Low Volatility Stream:
    - per-dim mean within [500, 1500]
    - variance/scale within [5, 20]
    - each value drawn from either Gaussian or Laplace (50/50)
    """
    mean_range = (500, 1500)
    var_range = (5, 20)
    stream = np.zeros((length, dim), dtype=int)
    for t in range(length):
        for d in range(dim):
            stream[t, d] = int(sample_point(mean_range, var_range))
    return stream


def gen_distribution_drift_stream(length, dim, var_fixed=50, mean_start=100, mean_peak=5000):
    """
    Distribution Drift Stream:
    - fixed variance/scale = var_fixed
    - mean drifts linearly: mean_start -> mean_peak over first half,
      then mean_peak -> mean_start over second half
    - each value drawn from either Gaussian or Laplace (50/50)
    """
    half = length // 2
    mu_up = np.linspace(mean_start, mean_peak, half, endpoint=False)
    mu_down = np.linspace(mean_peak, mean_start, length - half, endpoint=True)
    mu = np.concatenate([mu_up, mu_down])

    stream = np.zeros((length, dim), dtype=int)
    for t in range(length):
        mean_range = (float(mu[t]), float(mu[t]))
        var_range = (var_fixed, var_fixed)
        for d in range(dim):
            stream[t, d] = int(sample_point(mean_range, var_range))
    return stream


def gen_periodic_switch_stream(length, dim, period=1000,
                               regimes=((500, 5), (1500, 20), (3000, 50))):
    """
    Periodic Switch Stream:
    - cycles through regimes every 'period' steps
    - regimes are (mean, var/scale)
    - each value drawn from either Gaussian or Laplace (50/50)
    """
    stream = np.zeros((length, dim), dtype=int)
    n_reg = len(regimes)

    for t in range(length):
        r = (t // period) % n_reg
        mu, var = regimes[r]
        mean_range = (mu, mu)
        var_range = (var, var)
        for d in range(dim):
            stream[t, d] = int(sample_point(mean_range, var_range))
    return stream


def gen_sparse_spike_stream(length, dim,
                            burst_prob=0.01,
                            base_mean_range=(500, 1500),
                            base_var_range=(5, 20),
                            burst_k_range=(1, 3),
                            burst_mag_range=(500, 3000)):
    """
    Sparse Spike (Burst) Stream:
    - mostly low-volatility baseline
    - with probability burst_prob at each step, add a large spike to
      a small subset of dimensions (k in burst_k_range)
    """
    stream = np.zeros((length, dim), dtype=float)

    for t in range(length):
        # baseline
        for d in range(dim):
            stream[t, d] = sample_point(base_mean_range, base_var_range)

        # burst event
        if random.random() < float(burst_prob):
            k = random.randint(int(burst_k_range[0]), int(burst_k_range[1]))
            dims = random.sample(range(dim), k)
            for d in dims:
                spike = random.uniform(*burst_mag_range)
                stream[t, d] += spike

    return stream.astype(int)


def gen_correlated_latent_factor_stream(length, dim,
                                       base_mean_range=(500, 1500),
                                       base_var_range=(5, 20),
                                       ar=0.98, latent_sigma=30.0,
                                       load_range=(0.8, 1.2)):
    """
    Correlated Latent-Factor Stream:
    - all dimensions share a time-varying latent factor z_t (AR(1))
    - x_{t,d} = baseline_{t,d} + a_d * z_t
    - baseline_{t,d} is low-volatility (Gaussian/Laplace 50/50)
    """
    # per-dimension loadings
    A = np.array([random.uniform(*load_range) for _ in range(dim)], dtype=float)

    # latent factor z_t (AR(1))
    z = np.zeros(length, dtype=float)
    for t in range(1, length):
        z[t] = float(ar) * z[t - 1] + np.random.normal(0.0, float(latent_sigma))

    stream = np.zeros((length, dim), dtype=float)
    for t in range(length):
        # baseline part (keeps your 50/50 Gaussian/Laplace style)
        for d in range(dim):
            stream[t, d] = sample_point(base_mean_range, base_var_range) + A[d] * z[t]

    return stream.astype(int)


def save_csv(arr, filename):
    pd.DataFrame(arr).to_csv(filename, index=False)
    print(f"Saved {filename}")


if __name__ == "__main__":
    # 1) High Volatility
    vol_stream = gen_high_volatility(LEN, DIM)
    save_csv(vol_stream, f"high_volatility_len{LEN}dim{DIM}.csv")

    # 2) Low Volatility
    low_stream = gen_low_volatility(LEN, DIM)
    save_csv(low_stream, f"low_volatility_len{LEN}dim{DIM}.csv")

    # 3) Distribution Drift
    drift_stream = gen_distribution_drift_stream(LEN, DIM, var_fixed=50, mean_start=100, mean_peak=5000)
    save_csv(drift_stream, f"distribution_drift_len{LEN}dim{DIM}.csv")

    # 4) Periodic Switch
    periodic_stream = gen_periodic_switch_stream(LEN, DIM, period=1000,
                                                 regimes=((500, 5), (1500, 20), (3000, 50)))
    save_csv(periodic_stream, f"periodic_switch_len{LEN}dim{DIM}.csv")

    # 5) Sparse Spike (Burst)
    spike_stream = gen_sparse_spike_stream(
        LEN, DIM,
        burst_prob=BURST_PROB,
        base_mean_range=(500, 1500),
        base_var_range=(5, 20),
        burst_k_range=BURST_K_RANGE,
        burst_mag_range=BURST_MAG_RANGE,
    )
    save_csv(spike_stream, f"sparse_spike_len{LEN}dim{DIM}.csv")

    # 6) Correlated Latent-Factor
    corr_stream = gen_correlated_latent_factor_stream(
        LEN, DIM,
        base_mean_range=(500, 1500),
        base_var_range=(5, 20),
        ar=LATENT_AR,
        latent_sigma=LATENT_SIGMA,
        load_range=LATENT_LOAD_RANGE,
    )
    save_csv(corr_stream, f"correlated_latent_factor_len{LEN}dim{DIM}.csv")
