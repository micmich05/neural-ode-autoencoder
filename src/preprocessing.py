"""
Preprocessing pipeline for CSE-CIC-IDS2018 (parquet format).

Loads parquet files, removes duplicates, selects features, performs a
stratified split, normalizes with RobustScaler (fit on train-benign only),
creates fixed-size windows, and saves train/val/test tensors.

Design decisions are documented in notebooks/02_feature_engineering.ipynb.

Usage:
    python src/preprocessing.py
    python src/preprocessing.py --config configs/default.yaml
"""

import argparse
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_data(raw_dir: str) -> pd.DataFrame:
    """Load all parquet files in chronological order and concatenate."""
    parquet_files = sorted(Path(raw_dir).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")

    def extract_date(path):
        m = re.search(r"(\d{2})-(\d{2})-(\d{4})", path.name)
        if m:
            day, month, year = m.groups()
            return f"{year}-{month}-{day}"
        return "9999"

    parquet_files.sort(key=extract_date)

    dfs = []
    for f in parquet_files:
        print(f"  Loading {f.name}...", end=" ")
        df = pd.read_parquet(f)
        print(f"{len(df):,} rows")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(df_all):,} rows, {len(df_all.columns)} columns")
    return df_all


def clean_and_select_features(
    df: pd.DataFrame,
    features_to_drop: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Remove duplicates, extract labels, drop unwanted features."""
    n_initial = len(df)

    # 1. Remove fully duplicated rows
    df = df.drop_duplicates().reset_index(drop=True)
    n_deduped = len(df)
    print(f"  Removed {n_initial - n_deduped:,} duplicate rows")

    # 2. Extract labels
    labels = df["Label"].copy()

    # 3. Keep only numeric columns, drop Label
    df = df.select_dtypes(include=[np.number])

    # 4. Drop features identified in feature engineering analysis
    cols_to_drop = [c for c in features_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} features → {len(df.columns)} features kept")

    # 5. Replace any remaining Inf with NaN, impute with median
    df = df.replace([np.inf, -np.inf], np.nan)
    n_nan = df.isna().sum().sum()
    if n_nan > 0:
        df = df.fillna(df.median())
        print(f"  Imputed {n_nan:,} NaN/Inf values with column median")

    print(f"  Final shape: {df.shape}")
    return df, labels


def stratified_split(
    features: pd.DataFrame,
    labels: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> dict:
    """Stratified split ensuring all attack types in every split."""
    test_ratio = 1.0 - train_ratio - val_ratio

    # First: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=labels,
    )

    # Second: split temp into val/test (50/50 of the 30%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
    )

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    for name, (X, y) in splits.items():
        n_b = (y == "Benign").sum()
        n_a = len(y) - n_b
        print(f"  {name:5s}: {len(y):>10,} flows  (benign={n_b:,}, attack={n_a:,})")

    return splits


def create_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed-size windows from benign and attack flows separately.

    The stratified split shuffles rows, so benign and attack flows are
    interleaved. Creating windows from the mixed sequence would make
    nearly every window contain at least one attack flow, leaving zero
    clean benign windows for evaluation.

    Instead, we group flows by class, window each group independently,
    then concatenate and shuffle the resulting windows.
    """
    benign_mask = y == 0
    attack_mask = y == 1

    windows_list = []
    labels_list = []

    for mask, label in [(benign_mask, 0), (attack_mask, 1)]:
        X_group = X[mask]
        n_windows = len(X_group) // window_size
        if n_windows == 0:
            continue
        n_used = n_windows * window_size
        X_w = X_group[:n_used].reshape(n_windows, window_size, -1)
        y_w = np.full(n_windows, label, dtype=np.int64)
        windows_list.append(X_w)
        labels_list.append(y_w)

    X_windows = np.concatenate(windows_list, axis=0)
    y_windows = np.concatenate(labels_list, axis=0)

    # Shuffle windows so benign/attack are interleaved (not grouped)
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X_windows))
    return X_windows[idx], y_windows[idx]


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

    # 2. Clean and select features
    print("\n=== 2. Cleaning data & selecting features ===")
    df_clean, labels = clean_and_select_features(df_raw, prep_cfg["features_to_drop"])
    feature_names = df_clean.columns.tolist()

    # 3. Stratified split
    print("\n=== 3. Stratified split ===")
    splits = stratified_split(
        df_clean, labels,
        train_ratio=prep_cfg["split"]["train"],
        val_ratio=prep_cfg["split"]["val"],
        random_state=prep_cfg.get("random_state", 42),
    )

    # 4. Filter training to benign-only
    print("\n=== 4. Filtering training set to benign only ===")
    X_train, y_train = splits["train"]
    benign_mask = y_train == "Benign"
    X_train_benign = X_train[benign_mask]
    print(f"  Train: {len(X_train):,} → {len(X_train_benign):,} benign-only flows")

    # 5. Fit scaler on training benign data only
    print("\n=== 5. Fitting scaler on training benign data ===")
    scaler = RobustScaler()
    scaler.fit(X_train_benign)
    print(f"  Scaler fitted on {len(X_train_benign):,} benign training flows")

    # Scale all splits
    X_train_scaled = scaler.transform(X_train_benign).astype(np.float32)

    X_val, y_val = splits["val"]
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    X_test, y_test = splits["test"]
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # Create binary labels for val/test (0=benign, 1=attack)
    y_train_binary = np.zeros(len(X_train_benign), dtype=np.int64)
    y_val_binary = (y_val.values != "Benign").astype(np.int64)
    y_test_binary = (y_test.values != "Benign").astype(np.int64)

    # 6. Create windows
    print("\n=== 6. Creating windows ===")
    window_size = prep_cfg["window_size"]

    X_train_w, y_train_w = create_windows(X_train_scaled, y_train_binary, window_size)
    X_val_w, y_val_w = create_windows(X_val_scaled, y_val_binary, window_size)
    X_test_w, y_test_w = create_windows(X_test_scaled, y_test_binary, window_size)

    for name, X, y in [("train", X_train_w, y_train_w), ("val", X_val_w, y_val_w), ("test", X_test_w, y_test_w)]:
        n_a = int(y.sum())
        print(f"  {name:5s}: X={str(X.shape):25s}  benign={len(y)-n_a:,}  attack={n_a:,}")

    # 7. Save
    print("\n=== 7. Saving tensors ===")
    os.makedirs(output_dir, exist_ok=True)

    for name, X, y in [("train", X_train_w, y_train_w), ("val", X_val_w, y_val_w), ("test", X_test_w, y_test_w)]:
        torch.save(torch.from_numpy(X), os.path.join(output_dir, f"{name}_X.pt"))
        torch.save(torch.from_numpy(y), os.path.join(output_dir, f"{name}_y.pt"))

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(output_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)

    print(f"  Saved to {output_dir}/")
    print(f"  Features: {len(feature_names)}")
    print(f"  Scaler: RobustScaler (fitted on train benign)")

    print("\n=== Preprocessing complete ===")


if __name__ == "__main__":
    main()
