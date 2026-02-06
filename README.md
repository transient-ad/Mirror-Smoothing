# Mirror-Smoothing

This repository contains the implementation and experimental code for the paper **"Mirror-Smoothing: Leveraging historical data for enhanced utility in differentially private data stream release"**.

## Overview

Mirror-Smoothing is a comprehensive differential privacy framework for continuous data stream release, integrating three key mechanisms:

- **Sample-Mirror**: Guides when to sample by looking at historical sampling data, focusing on timestamps with clear data changes to address sampling myopia.
- **Budget-Mirror**: Dynamically allocates the privacy budget by learning from historical consumption, ensuring it is neither excessive nor insufficient, thus mitigating misallocation.
- **Noise-Mirror**: Smooths the current output by combining fresh noise with historically generated noise, leveraging their temporal correlation to effectively reduce noise volatility.

## Project Structure

```

├── dataset/
│   ├── real_world_datasets/      # 6 real-world datasets
│   │   ├── COVID_19_Deaths.csv
│   │   ├── Flu_Deaths.csv
│   │   ├── Unemployment.csv
│   │   ├── Cab_Industry.csv
│   │   ├── td_output.csv         # T-Drive trajectories
│   │   └── energydata.csv
│   └── synthetic_datasets/       # 6 synthetic datasets
│       ├── high_volatility_len10000dim10.csv
│       ├── low_volatility_len10000dim10.csv
│       ├── distribution_drift_len10000dim10.csv
│       ├── periodic_switch_len10000dim10.csv
│       ├── sparse_spike_len10000dim10.csv
│       ├── correlated_latent_factor_len10000dim10.csv
│       └── generate_synthetic_data.py     # Generate synthetic data

│
├── methods/
│   ├── mirror_smoothing.py       # Our proposed method
│   ├── SPAS.py                   # Baseline: SPAS
│   ├── Naive.py                  # Baseline: Naive 
│   ├── BucOrder.py               # Baseline: BucOrder
│   ├── CompOrder.py              # Baseline: CompOrder
│   ├── DPI.py                    # Baseline: DPI
│   ├── PeGaSus.py                # Baseline: PeGaSus
│   ├── AdaPub.py                 # Baseline: AdaPub
│   └── data_process.py           # Data loading utilities
│
├── output_mae/                   # MAE experimental results
├── output_mre/                   # MRE experimental results
│
└── run_code.py                   # Main experimental script
```

## Requirements

```bash
pip install numpy pandas torch matplotlib seaborn
```

**Hardware**: GPU recommended (CUDA support) for faster computation.

## Generate Synthetic Data

```bash
python generate_synthetic_data.py 
```

This will get :

- high_volatility_len10000dim10.csv
- low_volatility_len10000dim10.csv
- distribution_drift_len10000dim10.csv
- periodic_switch_len10000dim10.csv
- sparse_spike_len10000dim10.csv
- correlated_latent_factor_len10000dim10.csv

## Quick Start

### 1. Run All Experiments

```bash
python run_code.py
```

This will:

- Test 8 methods on 12 datasets
- Vary epsilon from 0.01 to 1.0
- Save results to `./output_mae/<date>/` or `./output_mre/<date>/`

### 2. Configuration

Edit `run_code.py` to customize:

```python
# Choose metric (line 141-142)
metric = "mae"  # or "mre"

# Datasets (line 113)
datasets = [
    "covid19", "flu_deaths", "unemployment", 
    "cab", "tdrive", "energy",
    "high_volatility", "low_volatility", ...
]

# Privacy budgets (line 130)
epsilon_list = [0.01, 0.1, 0.2, ..., 1.0]

# Window size (line 134)
window_size = 120

# Number of rounds (line 138)
round_ = 3
```

## Datasets

### Real-World Datasets (6)

1. **COVID-19 Deaths**
2. **Flu Deaths**
3. **Unemployment**
4. **Cab Trips**
5. **T-Drive Trajectories**
6. **Energy**

### Synthetic Datasets (6)

1. **High Volatility**
2. **Low Volatility**
3. **Distribution Drift**
4. **Periodic Switch**
5. **Sparse Spike**
6. **Correlated Latent Factor**

## License

This project is licensed under the MIT License.

---

**Note**: This code is provided for research purposes. Ensure compliance with privacy regulations when deploying in production environments.
