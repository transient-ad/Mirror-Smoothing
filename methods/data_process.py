def data_reader(name: str):
    """
    Read and return a stream as a list of row vectors: list[list[int]].

    This loader is organized into two groups:

    1) real_world_datasets (mostly 1D):
       - covid19, unemp_new, ilinet_new, footmart, Dth, Tdv, Ret, nation, taxi

    2) synthetic_datasets (10D):
       - high_volatility, low_volatility, distribution_drift, periodic_switch,
         sparse_spike, correlated_latent_factor

    Notes:
    - All outputs are returned as list[list[int]].
    - Use _read_csv_lastcol for 1D datasets.
    - Use _read_csv_allcols for multi-dimensional datasets.
    - Set `skip` to 1 if the CSV has a header row, otherwise set it to 0.
    """

    def _read_csv_lastcol(path, skip, col_idx=-1, as_float=False):
        """Generic CSV reader: skip first `skip` lines, read a single column (default last) -> 1D."""
        buf, cnt = [], 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                cnt += 1
                if cnt <= skip:
                    continue
                cols = line.strip().split(",")
                if not cols:
                    continue
                val = cols[col_idx]
                buf.append([int(float(val))] if as_float else [int(val)])
        return buf

    def _read_csv_allcols(path, skip, as_float=False):
        """Generic CSV reader: skip first `skip` lines, read all columns -> multi-D."""
        buf, cnt = [], 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                cnt += 1
                if cnt <= skip:
                    continue
                cols = [c.strip() for c in line.strip().split(",")]
                # Skip empty lines
                if len(cols) == 0 or (len(cols) == 1 and cols[0] == ""):
                    continue
                if as_float:
                    buf.append([int(float(x)) for x in cols])
                else:
                    buf.append([int(x) for x in cols])
        return buf

    key = name.strip().lower()

    # ------------------------------------------------------------------
    # Real-world datasets (mostly 1D)
    # ------------------------------------------------------------------
    # IMPORTANT: Adjust `skip` if your CSV files have/omit headers.
    real_world_datasets = {

        "covid19": {
            "path": "./dataset/real_world_datasets/COVID_19_Deaths.csv",
            "skip": 2, "col_idx": -1, "as_float": True
        },
        "unemployment": {
            "path": "./dataset/real_world_datasets/Unemployment.csv",
            "skip": 2, "col_idx": -1, "as_float": False
        },
        "cab": {
            "path": "./dataset/real_world_datasets/Cab_Industry.csv",
            "skip": 0, "col_idx": -1, "as_float": True
        },
        "flu_deaths": {
            "path": "./dataset/real_world_datasets/Flu_Deaths.csv",
            "skip": 2, "col_idx": -6, "as_float": True
        },
        "tdrive": {
            "path": "./dataset/real_world_datasets/td_output.csv",
            "skip": 2, "as_float": False
        },
        "energy": {
            "path": "./dataset/real_world_datasets/energydata.csv",
            "skip": 1, "as_float": False
        },
    }

    # Multi-dimensional real-world datasets that should read ALL columns
    real_world_multidim = {
        "tdrive": real_world_datasets["tdrive"],
    }

    # If requested dataset is a multi-D real dataset
    if key in real_world_multidim:
        cfg = real_world_multidim[key]
        return _read_csv_allcols(cfg["path"], skip=cfg["skip"], as_float=cfg.get("as_float", False))

    # If requested dataset is a 1D real dataset
    if key in real_world_datasets and key not in real_world_multidim:
        cfg = real_world_datasets[key]
        return _read_csv_lastcol(cfg["path"], skip=cfg["skip"], col_idx=cfg.get("col_idx", -1), as_float=cfg.get("as_float", False))

    # ------------------------------------------------------------------
    # Synthetic datasets (10D)
    # ------------------------------------------------------------------
    synthetic_datasets = {
        "high_volatility": {
            "path": "./dataset/synthetic_datasets/high_volatility_len10000dim10.csv",
            "skip": 1, "as_float": True
        },
        "low_volatility": {
            "path": "./dataset/synthetic_datasets/low_volatility_len10000dim10.csv",
            "skip": 1, "as_float": True
        },
        "distribution_drift": {
            "path": "./dataset/synthetic_datasets/distribution_drift_len10000dim10.csv",
            "skip": 1, "as_float": True
        },
        "periodic_switch": {
            "path": "./dataset/synthetic_datasets/periodic_switch_len10000dim10.csv",
            "skip": 1, "as_float": True
        },
        "sparse_spike": {
            "path": "./dataset/synthetic_datasets/sparse_spike_len10000dim10.csv",
            "skip": 1, "as_float": True
        },
        "latent_factor": {
            "path": "./dataset/synthetic_datasets/correlated_latent_factor_len10000dim10.csv",
            "skip": 1, "as_float": True
        },
    }

    if key in synthetic_datasets:
        cfg = synthetic_datasets[key]
        return _read_csv_allcols(cfg["path"], skip=cfg["skip"], as_float=cfg.get("as_float", False))

    raise ValueError(
        f"data_reader: unknown dataset name '{name}'. "
        f"Valid real_world_datasets={sorted([k for k in real_world_datasets.keys() if k not in real_world_multidim]) + ['tdv', 'ret']}, "
        f"synthetic_datasets={sorted(synthetic_datasets.keys())}."
    )
