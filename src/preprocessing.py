"""
Preprocessing pipeline for CSE-CIC-IDS2018 (parquet format).

Loads parquet files, removes duplicates, segments flows into sequential
windows, normalizes features, and saves train/val/test splits as PyTorch
tensors.

Usage:
    python src/preprocessing.py
    python src/preprocessing.py --config configs/default.yaml
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import RobustScaler


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_data(raw_dir: str) -> pd.DataFrame:
    """Load all parquet files from the directory and concatenate into a single DataFrame."""
    parquet_files = sorted(Path(raw_dir).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")

    dfs = []
    for f in parquet_files:
        print(f"  Loading {f.name}...", end=" ")
        df = pd.read_parquet(f)
        print(f"{len(df):,} rows")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(df_all):,} rows, {len(df_all.columns)} columns")
    return df_all


def clean_data(
    df: pd.DataFrame,
    metadata_columns: list[str],
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Clean the dataset: remove duplicates, extract labels, drop metadata."""
    n_initial = len(df)

    # 1. Remove fully duplicated rows
    if drop_duplicates:
        n_before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"  Removed {n_dropped:,} duplicate rows")

    # 2. Extract and separate labels
    labels = df["Label"].copy()

    # 3. Drop metadata columns (Label is the only non-numeric column in parquet)
    cols_to_drop = [c for c in metadata_columns if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} metadata columns")

    # 4. Replace Inf with NaN, then impute with median
    df = df.replace([np.inf, -np.inf], np.nan)
    n_nan = df.isna().sum().sum()
    if n_nan > 0:
        df = df.fillna(df.median())
        print(f"  Imputed {n_nan:,} NaN/Inf values with column median")

    print(f"  Final shape: {df.shape} (from {n_initial:,} initial rows)")
    return df, labels


def create_sequential_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int = 50,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Segment data into fixed-size sequential windows.

    Since the parquet files lack timestamps, we use sequential windowing:
    consecutive groups of `window_size` flows form each window. The file
    order (sorted by date) preserves the original temporal ordering.
    """
    n_samples = len(features)
    n_windows = n_samples // window_size

    windows = []
    window_labels = []

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        windows.append(features[start:end])
        window_labels.append(labels[start:end])

    print(f"  Windows created: {n_windows:,} (window_size={window_size} flows)")
    return windows, window_labels


def sequential_split(
    windows: list[np.ndarray],
    window_labels: list[np.ndarray],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict:
    """Sequential (non-random) split into train/val/test.

    Preserves the file ordering so that earlier capture days go to train,
    middle days to validation, and later days to test. This prevents data
    leakage from future observations into training.
    """
    n = len(windows)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": {
            "windows": windows[:train_end],
            "labels": window_labels[:train_end],
        },
        "val": {
            "windows": windows[train_end:val_end],
            "labels": window_labels[train_end:val_end],
        },
        "test": {
            "windows": windows[val_end:],
            "labels": window_labels[val_end:],
        },
    }

    for name, data in splits.items():
        print(f"  {name}: {len(data['windows']):,} windows")

    return splits


def aggregate_window_labels(window_labels: list[np.ndarray]) -> np.ndarray:
    """Assign a label per window: 1 if ANY flow is an attack, 0 if all are benign."""
    result = np.zeros(len(window_labels), dtype=np.int64)
    for i, labels in enumerate(window_labels):
        if any(l != "Benign" for l in labels):
            result[i] = 1
    return result


def save_splits(splits: dict, scaler: RobustScaler, output_dir: str, window_size: int):
    """Save splits as .pt tensors and the scaler as .pkl."""
    os.makedirs(output_dir, exist_ok=True)

    for name, data in splits.items():
        if len(data["windows"]) == 0:
            print(f"  Skipping {name}: no windows")
            continue

        # Stack windows — all have the same size (no padding needed)
        X = np.stack(data["windows"]).astype(np.float32)
        X_tensor = torch.from_numpy(X)

        # Per-window labels
        y = aggregate_window_labels(data["labels"])
        y_tensor = torch.from_numpy(y)

        torch.save(X_tensor, os.path.join(output_dir, f"{name}_X.pt"))
        torch.save(y_tensor, os.path.join(output_dir, f"{name}_y.pt"))
        print(f"  {name}: X={X_tensor.shape}, y={y_tensor.shape}")

    # Save scaler
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved to {output_dir}/scaler.pkl")


def main():
    parser = argparse.ArgumentParser(description="CSE-CIC-IDS2018 Preprocessing Pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = config["data"]["raw_dir"]
    output_dir = config["data"]["processed_dir"]
    prep_cfg = config["preprocessing"]

    # 1. Load raw data
    print("\n=== 1. Loading raw data ===")
    df_raw = load_raw_data(raw_dir)

    # 2. Clean data
    print("\n=== 2. Cleaning data ===")
    df_clean, labels = clean_data(
        df_raw,
        prep_cfg["metadata_columns"],
        drop_duplicates=prep_cfg.get("drop_duplicates", True),
    )

    # 3. Extract features
    print("\n=== 3. Extracting features ===")
    feature_names = df_clean.columns.tolist()
    features = df_clean.values.astype(np.float32)

    # 4. Create sequential windows
    print("\n=== 4. Creating sequential windows ===")
    windows, window_labels = create_sequential_windows(
        features,
        labels.values,
        window_size=prep_cfg["window_size"],
    )

    # 5. Sequential split
    print("\n=== 5. Sequential split ===")
    splits = sequential_split(
        windows,
        window_labels,
        train_ratio=prep_cfg["split"]["train"],
        val_ratio=prep_cfg["split"]["val"],
    )

    # 6. Fit scaler on training data only (prevents data leakage)
    print("\n=== 6. Fitting scaler on training data ===")
    train_flat = np.concatenate(splits["train"]["windows"], axis=0)
    scaler = RobustScaler()
    scaler.fit(train_flat)
    print(f"  Scaler fitted on {len(train_flat):,} training flows")

    # Apply scaler to all splits
    for split_name in splits:
        splits[split_name]["windows"] = [
            scaler.transform(w).astype(np.float32) for w in splits[split_name]["windows"]
        ]

    # 7. Save
    print("\n=== 7. Saving splits ===")
    save_splits(splits, scaler, output_dir, window_size=prep_cfg["window_size"])

    # Save feature names
    with open(os.path.join(output_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)
    print(f"  Feature names saved ({len(feature_names)} features)")

    print("\n=== Preprocessing complete ===")


if __name__ == "__main__":
    main()
