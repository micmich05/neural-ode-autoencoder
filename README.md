# Neural ODE Autoencoder for DDoS Detection

Unsupervised anomaly detection system for network traffic using a Neural ODE Autoencoder. The model learns the continuous-time dynamics of normal traffic in a latent space and flags deviations as potential DDoS attacks — no labeled data is needed during training.

## Problem Formulation

### Task

**Unsupervised anomaly detection** on network flow data. The model is trained exclusively on benign traffic and must identify attack traffic (DDoS, DoS, brute-force, etc.) at inference time without ever having seen labeled attack examples.

### Input Space

Each sample is a **window** of $T = 50$ consecutive network flows, where each flow is described by $D = 49$ numeric features:

$$X = \{x^{(i)}\}_{i=1}^{N}, \quad x^{(i)} \in \mathbb{R}^{T \times D}$$

The features capture five categories of network behavior: flow-level statistics, directional packet sizes, inter-arrival times, TCP flags, and session behavior indicators. All features are normalized with a RobustScaler fitted on training benign data only.

### Target Variable

The target is a binary label per window:

$$y^{(i)} = \begin{cases} 0 & \text{if all flows in the window are benign} \\ 1 & \text{if all flows in the window are attacks} \end{cases}$$

> **Important:** Labels are never used during training. They are used only for evaluation (AUROC, AUPRC, F1) and threshold selection on the validation set.

### Learning Objective

The model learns a reconstruction function $g_\theta: \mathbb{R}^{T \times D} \to \mathbb{R}^{T \times D}$ by minimizing the reconstruction error on benign windows:

$$\mathcal{L}(\theta) = \frac{1}{N_{\text{benign}}} \sum_{i: y^{(i)}=0} \frac{1}{T \cdot D} \left\| x^{(i)} - g_\theta(x^{(i)}) \right\|_F^2$$

### Anomaly Score

At inference, the **anomaly score** for a new window $x$ is its reconstruction error:

$$s(x) = \frac{1}{T \cdot D} \left\| x - g_\theta(x) \right\|_F^2$$

A window is flagged as anomalous if $s(x) > \tau$, where the threshold $\tau$ is chosen on the validation set to maximize the F1-score.

### Why This Works

The model learns to reconstruct only benign traffic patterns. When it encounters attack traffic:
1. The encoder maps it to an unfamiliar region of the latent space
2. The Neural ODE evolves the latent code using dynamics learned from benign data
3. The decoder produces a "benign-like" reconstruction
4. The mismatch between the attack input and benign-like output produces a high anomaly score

## Architecture

```
┌─────────────┐      ┌──────────────────┐       ┌──────────────────┐       ┌─────────── ──┐       ┌─────────────┐
│   Input x   │      │   BiGRU Encoder  │       │    Neural ODE    │       │ MLP Decoder  │       │  Output x̂   │
│ (B, 50, 49) │─────>│  2-layer BiGRU   │──────>│  dz/dt = fθ(z,t) │──────>│  MLP + SiLU  │──────>│ (B, 50, 49) │
│             │      │  128 hidden/dir  │       │  dopri5, t∈[0,1] │       │  no output   │       │             │
└─────────────┘      └──────────────────┘       └──────────────────┘       │  activation  │       └──────┬──────┘
                              │                          │                 └────────── ───┘              │
                         z₀ (B, 32)                 z₁ (B, 32)                                           │
                                                                                                         ▼
                                                                                              ┌─────────────────┐
                                                                                              │  Anomaly Score  │
                                                                                              │ MSE(x, x̂) > τ   │
                                                                                              │    → Alert      │
                                                                                              └─────────────────┘
```

### Component Summary

| Component | Architecture | Parameters |
|-----------|-------------|------------|
| **Encoder** | BiGRU (2 layers, 128 hidden, bidirectional) → LayerNorm → Dropout(0.2) → Linear(256, 32) | ~530K |
| **Latent space** | 32-dimensional | — |
| **Neural ODE** | MLP dynamics: $[z \| t]$ (33) → LN+128 → LN+128 → 32, SiLU, dopri5 | ~22K |
| **Decoder** | MLP: 32 → LN+256 → LN+256 → 2450, SiLU, Dropout(0.2) | ~620K |
| **Total** | | **~1.17M** |

### BiGRU Encoder

Maps a window of $T$ network flows into a latent vector $z_0$:

$$x \in \mathbb{R}^{B \times 50 \times 49} \xrightarrow{\text{BiGRU}} [\overrightarrow{h_T} \| \overleftarrow{h_1}] \in \mathbb{R}^{B \times 256} \xrightarrow{\text{LN + Linear}} z_0 \in \mathbb{R}^{B \times 32}$$

The bidirectional GRU processes the sequence in both directions, capturing temporal patterns in flow ordering. The final hidden states from forward and backward passes are concatenated, normalized (LayerNorm), and linearly projected to the latent dimension. Dropout (0.2) is applied between GRU layers and before projection.

### Neural ODE

The core innovation: a continuous-time transformation in latent space defined as an ODE initial value problem:

$$\frac{dz}{dt} = f_\theta(z, t), \quad z(0) = z_0, \quad z_1 = z_0 + \int_0^1 f_\theta(z(t), t) \, dt$$

The dynamics function $f_\theta$ is an MLP with LayerNorm and SiLU activations that takes the concatenation $[z \| t] \in \mathbb{R}^{33}$ as input. The Dormand-Prince solver (`dopri5`) adaptively integrates from $t=0$ to $t=1$ with tolerances `atol=rtol=1e-5`. No dropout in $f_\theta$ — the adaptive solver calls it multiple times per step, so stochastic dropout would break error estimation.

### MLP Decoder

Reconstructs the full $50 \times 49$ window from the evolved latent vector:

$$z_1 \in \mathbb{R}^{32} \xrightarrow{\text{LN+SiLU+Dropout}} \mathbb{R}^{256} \xrightarrow{\text{LN+SiLU+Dropout}} \mathbb{R}^{256} \xrightarrow{\text{Linear}} \mathbb{R}^{2450} \xrightarrow{\text{reshape}} \hat{x} \in \mathbb{R}^{50 \times 49}$$

No output activation — targets are RobustScaler-transformed and unbounded.

### Regularization

In addition to the reconstruction loss, the model uses a **kinetic energy regularization** on the ODE dynamics:

$$\mathcal{R}_{\text{KE}}(\theta) = \int_0^1 \left\| f_\theta(z(t), t) \right\|^2 dt$$

This penalizes the total "energy" of the latent trajectory, encouraging straight-line paths and smooth dynamics without constraining the latent space to a specific distribution (unlike the KL divergence in VAEs). This is more natural for Neural ODEs and preserves the geometric flexibility of the latent space. See [Finlay et al., 2020](https://arxiv.org/abs/2002.02798).

The total loss is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda \cdot \mathcal{R}_{\text{KE}}$$

where $\lambda$ controls the regularization strength.

## Dataset

[CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) — a large-scale intrusion detection dataset with **~6.7M network flows** and **77 numeric features** across 10 parquet files (one per capture day), including 15 attack types.

### Raw Data Overview

```
10 parquet files (~692 MB total)          Label distribution (after dedup)
──────────────────────────────────        ─────────────────────────────────
Bruteforce   (2018-02-14)  619K rows      Benign ████████████████████ 79.0%
DoS1         (2018-02-15)  795K rows      DDoS   ████               9.1%
DoS2         (2018-02-16)  592K rows      DoS    ██                 3.1%
DDoS1        (2018-02-20)  955K rows      Bot    █                  2.3%
DDoS2        (2018-02-21)  561K rows      Infil  █                  1.8%
Web1         (2018-02-22)  830K rows      Brute  █                  1.5%
Web2         (2018-02-23)  829K rows      Other  ░                  <1%
Infil1       (2018-02-28)  457K rows      ─────────────────────────────────
Infil2       (2018-03-01)  249K rows      6.32M flows after dedup
Botnet       (2018-03-02)  772K rows      15 distinct attack types
```

### Preprocessing Pipeline

Every decision is documented with rationale in [`notebooks/02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb).

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          PREPROCESSING PIPELINE                              │
│                                                                              │
│  ┌─────────────┐    ┌───────────────┐    ┌──────────────────┐                │
│  │  10 Parquet │───>│  Dedup Rows   │───>│ Feature Selection│                │
│  │  Files      │    │  -339K (5.1%) │    │ 77 → 49 features │                │
│  │  6.66M rows │    │  6.32M rows   │    │                  │                │
│  └─────────────┘    └───────────────┘    └────────┬─────────┘                │
│                                                   │                          │
│                                                   ▼                          │
│                                        ┌──────────────────┐                  │
│                                        │ Stratified Split │                  │
│                                        │ (by Label, 42)   │                  │
│                                        └─┬──────┬───────┬─┘                  │
│                                          │      │       │                    │
│                              ┌───────────┘      │       └───────────┐        │
│                              ▼                  ▼                   ▼        │
│                     ┌────────────────┐ ┌──────────────┐  ┌──────────────┐    │
│                     │  Train (70%)   │ │  Val (15%)   │  │  Test (15%)  │    │
│                     │  4.42M flows   │ │  948K flows  │  │  948K flows  │    │
│                     └───────┬────────┘ └──────┬───────┘  └───── ─┬──────┘    │
│                             │                 │                  │           │
│                             ▼                 │                  │           │
│                     ┌────────────────┐        │                  │           │
│                     │ Filter Benign  │        │                  │           │
│                     │ Only (remove   │        │                  │           │
│                     │ 928K attacks)  │        │                  │           │
│                     └───────┬────────┘        │                  │           │
│                             │                 │                  │           │
│                             ▼                 │                  │           │
│                     ┌────────────────┐        │                  │           │
│                     │ Fit Scaler     │        │                  │           │
│                     │ (RobustScaler  │        │                  │           │
│                     │  on 3.5M       │        │                  │           │
│                     │  benign flows) │        │                  │           │
│                     └───────┬────────┘        │                  │           │
│                             │                 │                  │           │
│                             ▼                 ▼                  ▼           │
│                     ┌─────────────────────────────────────────────────┐      │
│                     │           Transform All Splits                  │      │
│                     └────────────────────┬────────────────────────────┘      │
│                                          │                                   │
│                              ┌───────────┼───────────┐                       │
│                              ▼           ▼           ▼                       │
│                     ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│                     │ Window   │  │ Window   │  │ Window   │                 │
│                     │ (50 flows│  │ by class │  │ by class │                 │
│                     │  each)   │  │ + shuffle│  │ + shuffle│                 │
│                     └────┬─────┘  └────┬─────┘  └────┬─────┘                 │
│                          ▼             ▼             ▼                       │
│                     ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│                     │ train_X  │  │  val_X   │  │  test_X  │                 │
│                     │ train_y  │  │  val_y   │  │  test_y  │                 │
│                     │ (69918,  │  │ (18959,  │  │ (18959,  │                 │
│                     │  50, 49) │  │  50, 49) │  │  50, 49) │                 │
│                     │ 100%     │  │ 79%B/21%A│  │ 79%B/21%A│                 │
│                     │ benign   │  │          │  │          │                 │
│                     └──────────┘  └──────────┘  └──────────┘                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Details

#### 1. Duplicate Removal

339,529 fully duplicated rows (5.1%) are removed before any splitting. This prevents the same flow from appearing in both train and test sets, which would be a form of data leakage.

#### 2. Feature Selection (77 → 49 features)

| Category | Features Dropped | Count | Reason |
|----------|-----------------|-------|--------|
| **Zero-variance** | `Bwd PSH Flags`, `Bwd URG Flags`, 6× `Avg Bulk Rate` features | 8 | Constant 0 across all rows — zero information |
| **Perfect duplicates** | `Subflow Fwd/Bwd Packets/Bytes`, `Avg Fwd/Bwd Segment Size`, `Fwd PSH Flags`, `CWE/ECE Flag Count`, `Fwd IAT Total/Std/Max/Min`, `Idle Std/Max/Min` | 16 | Exact copies of other features (r=1.0), produced by CICFlowMeter computing the same quantity under different names |
| **Low-information** | `Fwd Seg Size Min`, `Fwd URG Flags`, `Fwd Act Data Packets`, `Packet Length Variance` | 4 | Near-constant, direct mapping to Protocol, or monotonic transform of a kept feature |

The 49 kept features span five categories:

- **Flow-level** (4): Protocol, Flow Duration, Flow Bytes/s, Flow Packets/s
- **Directional packet stats** (16): Total/Max/Min/Mean/Std of Fwd and Bwd packet lengths, header lengths, Fwd/Bwd Packets/s
- **Inter-arrival times** (10): Flow IAT Mean/Std/Max/Min, Fwd IAT Mean, Bwd IAT Total/Mean/Std/Max/Min
- **TCP flags** (7): FIN, SYN, RST, PSH, ACK, URG Flag Count + Down/Up Ratio
- **Session behavior** (12): Packet Length Min/Max/Mean/Std, Avg Packet Size, Init Fwd/Bwd Win Bytes, Active Mean/Std/Max/Min, Idle Mean

#### 3. Stratified Split (not sequential)

A sequential split is **not viable** for this dataset: each parquet file contains a different attack type, so any contiguous split leaves entire attack categories missing from val or test. For example, a naive sequential split puts only 524 web attacks (0.05%) in the test set — useless for evaluation.

Instead, we use a **stratified random split** (seed=42) that guarantees proportional representation of all 15 attack types in every split:

```
Split     Flows       Benign       Attack      Attack types
─────     ─────────   ──────────   ─────────   ────────────
Train     4,424,002   3,495,912    928,090     all 15
Val         948,000     749,124    198,876     all 15
Test        948,001     749,124    198,877     all 15
```

#### 4. Benign-Only Training

Since this is an **unsupervised anomaly detector**, the training set is filtered to benign flows only (3,495,912 flows). The 928,090 attack flows in the train split are discarded — the model never sees any attack during training. Val and test retain their full mix for threshold tuning and evaluation.

#### 5. Normalization (RobustScaler)

A RobustScaler (median + IQR) is fitted **exclusively on training benign data**. This prevents two forms of leakage:
- **Test leakage**: scaler parameters would encode information about unseen data
- **Attack leakage**: attack flows would shift the "normal" baseline, degrading anomaly sensitivity

The same fitted scaler is applied to transform val and test sets.

#### 6. Windowing

Flows are grouped into fixed-size windows of **50 flows** for the BiGRU sequence encoder. Key design choices:

- **Windowing happens after splitting** — no window spans a split boundary
- **Benign and attack flows are windowed separately** within val/test, then shuffled together. This is necessary because the stratified split interleaves benign and attack flows randomly; naively windowing the mixed sequence would contaminate every window with at least one attack flow, leaving zero clean benign windows for evaluation
- **Window label**: 0 if all flows are benign, 1 if all flows are attacks

#### Final Output

```
data/processed/
├── train_X.pt          # (69918, 50, 49)  float32  — benign-only windows
├── train_y.pt          # (69918,)         int64    — all zeros
├── val_X.pt            # (18959, 50, 49)  float32  — 79% benign, 21% attack
├── val_y.pt            # (18959,)         int64    — 0=benign, 1=attack
├── test_X.pt           # (18959, 50, 49)  float32  — 79% benign, 21% attack
├── test_y.pt           # (18959,)         int64    — 0=benign, 1=attack
├── scaler.pkl          # RobustScaler fitted on train benign
└── feature_names.pkl   # list of 49 feature names
```

### No-Leakage Guarantees

| Leakage Vector | Prevention |
|----------------|------------|
| Same flow in train and test | Duplicates removed before splitting |
| Scaler sees test data | Fitted on training benign only |
| Labels in training | Training tensor is 100% benign (verified by assertion) |
| Windows span split boundary | Windowing applied per-split after scaling |
| Future data in training | Stratified split preserves proportions, not temporal order (acceptable since model is not autoregressive) |

## Project Structure

```
├── configs/
│   └── default.yaml                    # All hyperparameters and feature drop list
├── data/
│   ├── raw/archive/                    # Raw CSE-CIC-IDS2018 parquets (not tracked)
│   └── processed/                      # Output tensors (not tracked)
├── checkpoints/                        # Model checkpoints (not tracked)
├── results/                            # Evaluation plots (not tracked)
├── figures/                            # Architecture diagrams from notebooks
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb    # Feature selection & dataset decisions
│   ├── 03_model_architecture.ipynb     # Architecture docs, math, diagrams
│   └── 04_training.ipynb              # Training with progress bars and curves
├── src/
│   ├── __init__.py                     # Exports model classes
│   ├── preprocessing.py                # Full preprocessing pipeline
│   ├── dataset.py                      # FlowWindowDataset + DataLoader factory
│   ├── model.py                        # BiGRUEncoder, ODEFunc, MLPDecoder, NeuralODEAutoencoder
│   ├── train.py                        # Training loop with early stopping
│   └── evaluate.py                     # Anomaly scoring, metrics, plots
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- ~700 MB disk space for the raw parquet dataset
- ~1.1 GB for the processed tensors

### Installation

```bash
git clone https://github.com/<your-username>/neural-ode-autoencoder.git
cd neural-ode-autoencoder
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Download the Dataset

Download CSE-CIC-IDS2018 from [Kaggle](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv) or the [UNB CIC website](https://www.unb.ca/cic/datasets/ids-2018.html) and place the parquet files in `data/raw/archive/`.

### Preprocessing

```bash
python src/preprocessing.py --config configs/default.yaml
```

This generates train/val/test tensors in `data/processed/`.

### Training

```bash
python src/train.py --config configs/default.yaml
```

Trains on benign-only windows with MSE loss, cosine annealing LR, and early stopping. Best checkpoint saved to `checkpoints/best.pt`.

### Evaluation

```bash
python src/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt --split test
```

Computes AUROC, AUPRC, F1-score and generates score distribution, ROC, and precision-recall plots in `results/`.

## Training Approach

The model is trained **entirely unsupervised** using reconstruction loss on benign traffic. Labels are used **only** for post-hoc evaluation, never during training. At inference time, windows with reconstruction error above a learned threshold are flagged as anomalies.

### Class Balance

The natural distribution (~79% benign, ~21% attack at window level) is preserved without resampling:

- **Training is unsupervised** — class balance is irrelevant (model sees only benign)
- **Evaluation uses threshold-independent metrics** (AUROC, AUPRC) that are robust to imbalance
- **Resampling would distort evaluation** — undersampling benign reduces the false-positive test surface; oversampling attacks inflates recall

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
- **pandas** / **NumPy** / **PyArrow** — Data manipulation
- **MLflow** — Experiment tracking
- **Matplotlib** / **Plotly** — Visualization

## License

This project is for academic/research purposes.
