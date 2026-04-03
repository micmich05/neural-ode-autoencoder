# Neural ODE Autoencoder for DDoS Detection

Unsupervised anomaly detection system for network traffic using a Neural ODE Autoencoder. The model learns the temporal dynamics of normal traffic and flags deviations as potential DDoS attacks — no labeled data is needed during training.

## Architecture

```
Network Flows → [BiGRU Encoder] → z₀ → [Neural ODE: dz/dt = fθ(z,t)] → z₁ → [MLP Decoder] → Reconstruction
                                                                                                      ↓
                                                                                              Reconstruction Error
                                                                                                      ↓
                                                                                              Anomaly Score → Alert
```

| Component | Details |
|-----------|---------|
| **Encoder** | Bidirectional GRU (2 layers, 128 hidden) |
| **Latent space** | 32-dimensional |
| **Neural ODE** | MLP dynamics with SiLU activation, Dormand-Prince (dopri5) solver |
| **Decoder** | MLP with SiLU activation |

**Key idea:** The Neural ODE learns a continuous-time dynamical system `dz/dt = fθ(z,t)` that captures how normal traffic evolves in latent space. DDoS traffic follows fundamentally different dynamics, producing high reconstruction error.

## Dataset

[CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) — a large-scale intrusion detection dataset with ~16M network flows and ~80 features, including multiple DDoS/DoS attack types.

The preprocessing pipeline:
1. Loads raw CSVs and fixes known issues (duplicate headers, Inf/NaN values, duplicate columns)
2. Segments flows into **temporal windows** (30s default, min 5 flows per window)
3. Applies **temporal split** (70/15/15) — no data leakage from future to past
4. Normalizes with **RobustScaler** fitted only on training data
5. Pads windows and exports as PyTorch tensors

## Project Structure

```
├── configs/
│   └── default.yaml          # Model, training, and preprocessing hyperparameters
├── data/
│   ├── raw/                  # Raw CSE-CIC-IDS2018 CSVs (not tracked)
│   └── processed/            # Preprocessed tensors (not tracked)
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory data analysis
├── src/
│   └── preprocessing.py      # Data loading, cleaning, windowing, and splitting
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- ~20 GB disk space for the raw dataset

### Installation

```bash
git clone https://github.com/<your-username>/neural-ode-autoencoder.git
cd neural-ode-autoencoder
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Download the Dataset

Download CSE-CIC-IDS2018 from [Kaggle](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv) or the [UNB CIC website](https://www.unb.ca/cic/datasets/ids-2018.html) and place the CSV files in `data/raw/`.

### Preprocessing

```bash
python src/preprocessing.py --config configs/default.yaml
```

This generates train/val/test tensors in `data/processed/`.

## Training Approach

The model is trained **entirely unsupervised** using reconstruction loss on benign traffic. Labels are used **only** for post-hoc evaluation, never during training. At inference time, windows with reconstruction error above a learned threshold are flagged as anomalies.

### Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| **AUROC** | Primary — threshold-independent discrimination |
| **AUPRC** | Performance under class imbalance |
| **F1-Score** | Balanced precision/recall at optimal threshold |
| **Detection Latency** | Time from attack onset to first alert |

### Baselines

- Isolation Forest
- MLP Autoencoder
- LSTM Autoencoder

## Tech Stack

- **PyTorch** + **torchdiffeq** — Neural ODE implementation
- **scikit-learn** — Preprocessing and baselines
- **pandas** / **NumPy** — Data manipulation
- **MLflow** — Experiment tracking
- **Matplotlib** / **Plotly** — Visualization

## License

This project is for academic/research purposes.
