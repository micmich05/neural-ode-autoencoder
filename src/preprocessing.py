"""
Pipeline de preprocesamiento para CSE-CIC-IDS2018.

Carga los CSVs crudos, limpia, segmenta en ventanas temporales,
normaliza y guarda splits train/val/test como tensores PyTorch.

Uso:
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
    """Carga todos los CSVs del directorio y concatena."""
    csv_files = sorted(Path(raw_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSVs en {raw_dir}")

    dfs = []
    for f in csv_files:
        print(f"  Cargando {f.name}...", end=" ")
        df = pd.read_csv(f, encoding="utf-8", low_memory=False)
        df.columns = df.columns.str.strip()
        print(f"{len(df):,} filas")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(df_all):,} filas, {len(df_all.columns)} columnas")
    return df_all


def clean_data(df: pd.DataFrame, metadata_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Limpia el dataset según los problemas conocidos de CIC-IDS2018."""
    n_initial = len(df)

    # 1. Eliminar filas con headers duplicados
    if "Label" in df.columns:
        mask = df["Label"] == "Label"
        n_header_rows = mask.sum()
        if n_header_rows > 0:
            df = df[~mask].reset_index(drop=True)
            print(f"  Eliminadas {n_header_rows:,} filas con headers duplicados")

    # 2. Extraer y separar labels
    labels = df["Label"].copy()

    # 3. Eliminar columna duplicada Fwd Header Length
    fwd_header_cols = [c for c in df.columns if "Fwd Header Length" in c]
    if len(fwd_header_cols) > 1:
        # Mantener la primera, eliminar duplicadas
        to_drop = fwd_header_cols[1:]
        df = df.drop(columns=to_drop)
        print(f"  Eliminadas columnas duplicadas: {to_drop}")

    # 4. Eliminar columnas de metadata
    cols_to_drop = [c for c in metadata_columns if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  Eliminadas {len(cols_to_drop)} columnas de metadata")

    # 5. Convertir todo a numérico
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6. Reemplazar Inf con NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # 7. Imputar NaN con mediana
    n_nan = df.isna().sum().sum()
    if n_nan > 0:
        medians = df.median()
        df = df.fillna(medians)
        print(f"  Imputados {n_nan:,} valores NaN/Inf con mediana")

    print(f"  Shape final: {df.shape} (de {n_initial:,} filas iniciales)")
    return df, labels


def parse_timestamps(df_original: pd.DataFrame) -> pd.Series:
    """Parsea la columna Timestamp del DataFrame original."""
    if "Timestamp" not in df_original.columns:
        raise ValueError("Columna 'Timestamp' no encontrada")
    return pd.to_datetime(df_original["Timestamp"], errors="coerce", dayfirst=True)


def create_temporal_windows(
    features: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    window_size_seconds: int = 30,
    min_flows_per_window: int = 5,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Segmenta los datos en ventanas temporales."""
    # Ordenar por timestamp
    valid_mask = ~pd.isna(timestamps)
    sort_idx = np.argsort(timestamps[valid_mask])

    features_sorted = features[valid_mask][sort_idx]
    labels_sorted = labels[valid_mask][sort_idx]
    timestamps_sorted = timestamps[valid_mask][sort_idx]

    # Crear ventanas
    windows = []
    window_labels = []

    ts_series = pd.Series(timestamps_sorted)
    window_td = pd.Timedelta(seconds=window_size_seconds)

    current_start = ts_series.iloc[0]
    end_time = ts_series.iloc[-1]

    while current_start < end_time:
        window_end = current_start + window_td
        mask = (ts_series >= current_start) & (ts_series < window_end)
        indices = np.where(mask.values)[0]

        if len(indices) >= min_flows_per_window:
            windows.append(features_sorted[indices])
            window_labels.append(labels_sorted[indices])

        current_start = window_end

    print(f"  Ventanas creadas: {len(windows)} (window={window_size_seconds}s, min_flows={min_flows_per_window})")
    return windows, window_labels


def temporal_split(
    windows: list[np.ndarray],
    window_labels: list[np.ndarray],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict:
    """Split temporal (NO aleatorio) en train/val/test."""
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
        print(f"  {name}: {len(data['windows'])} ventanas")

    return splits


def pad_windows(windows: list[np.ndarray], max_len: int | None = None) -> np.ndarray:
    """Pad variable-length windows to uniform size."""
    if max_len is None:
        max_len = max(w.shape[0] for w in windows)

    n_features = windows[0].shape[1]
    padded = np.zeros((len(windows), max_len, n_features), dtype=np.float32)

    for i, w in enumerate(windows):
        length = min(w.shape[0], max_len)
        padded[i, :length, :] = w[:length]

    return padded


def aggregate_window_labels(window_labels: list[np.ndarray]) -> np.ndarray:
    """Asigna label por ventana: 1 si ANY flow es ataque, 0 si todos son benignos."""
    result = np.zeros(len(window_labels), dtype=np.int64)
    for i, labels in enumerate(window_labels):
        if any(l != "Benign" for l in labels):
            result[i] = 1
    return result


def save_splits(splits: dict, scaler: RobustScaler, output_dir: str, max_window_len: int | None = None):
    """Guarda splits como tensores .pt y el scaler como .pkl."""
    os.makedirs(output_dir, exist_ok=True)

    # Determinar max_window_len global
    if max_window_len is None:
        all_windows = []
        for split_data in splits.values():
            all_windows.extend(split_data["windows"])
        max_window_len = int(np.percentile([w.shape[0] for w in all_windows], 95))
        print(f"  Max window length (p95): {max_window_len}")

    for name, data in splits.items():
        if len(data["windows"]) == 0:
            print(f"  Saltando {name}: sin ventanas")
            continue

        # Pad windows
        X = pad_windows(data["windows"], max_len=max_window_len)
        X_tensor = torch.from_numpy(X)

        # Labels por ventana
        y = aggregate_window_labels(data["labels"])
        y_tensor = torch.from_numpy(y)

        torch.save(X_tensor, os.path.join(output_dir, f"{name}_X.pt"))
        torch.save(y_tensor, os.path.join(output_dir, f"{name}_y.pt"))
        print(f"  {name}: X={X_tensor.shape}, y={y_tensor.shape}")

    # Guardar scaler
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler guardado en {output_dir}/scaler.pkl")


def main():
    parser = argparse.ArgumentParser(description="Preprocesamiento CSE-CIC-IDS2018")
    parser.add_argument("--config", default="configs/default.yaml", help="Path al archivo de configuración")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = config["data"]["raw_dir"]
    output_dir = config["data"]["processed_dir"]
    prep_cfg = config["preprocessing"]

    # 1. Cargar datos crudos
    print("\n=== 1. Cargando datos crudos ===")
    df_raw = load_raw_data(raw_dir)

    # 2. Parsear timestamps (antes de eliminar la columna)
    print("\n=== 2. Parseando timestamps ===")
    timestamps = parse_timestamps(df_raw)
    n_valid = timestamps.notna().sum()
    print(f"  Timestamps válidos: {n_valid:,} / {len(timestamps):,}")

    # 3. Limpiar datos
    print("\n=== 3. Limpiando datos ===")
    df_clean, labels = clean_data(df_raw, prep_cfg["metadata_columns"])

    # 4. Normalizar con RobustScaler (fit en todo el dataset, luego re-fit solo en train)
    print("\n=== 4. Normalizando ===")
    feature_names = df_clean.columns.tolist()
    features = df_clean.values.astype(np.float32)

    # 5. Crear ventanas temporales
    print("\n=== 5. Creando ventanas temporales ===")
    windows, window_labels = create_temporal_windows(
        features,
        labels.values,
        timestamps.values,
        window_size_seconds=prep_cfg["window_size_seconds"],
        min_flows_per_window=prep_cfg["min_flows_per_window"],
    )

    # 6. Split temporal
    print("\n=== 6. Split temporal ===")
    splits = temporal_split(
        windows,
        window_labels,
        train_ratio=prep_cfg["split"]["train"],
        val_ratio=prep_cfg["split"]["val"],
    )

    # 7. Fit scaler solo en train
    print("\n=== 7. Ajustando scaler en train ===")
    train_flat = np.concatenate(splits["train"]["windows"], axis=0)
    scaler = RobustScaler()
    scaler.fit(train_flat)
    print(f"  Scaler ajustado en {len(train_flat):,} flujos de train")

    # Aplicar scaler a todas las ventanas
    for split_name in splits:
        splits[split_name]["windows"] = [
            scaler.transform(w).astype(np.float32) for w in splits[split_name]["windows"]
        ]

    # 8. Guardar
    print("\n=== 8. Guardando splits ===")
    save_splits(splits, scaler, output_dir)

    # Guardar feature names
    with open(os.path.join(output_dir, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)
    print(f"  Feature names guardados ({len(feature_names)} features)")

    print("\n=== Preprocesamiento completado ===")


if __name__ == "__main__":
    main()
