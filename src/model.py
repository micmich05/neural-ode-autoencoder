"""
Neural ODE Autoencoder for unsupervised anomaly detection.

Architecture:
    BiGRU Encoder → z₀ → Neural ODE (dz/dt = fθ(z,t)) → z₁ → MLP Decoder → x̂

The model learns continuous-time dynamics of normal network traffic in
latent space. DDoS and other attacks follow different dynamics, producing
high reconstruction error that serves as an anomaly score.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint


class BiGRUEncoder(nn.Module):
    """Bidirectional GRU encoder that maps flow windows to a latent vector z₀.

    Input:  (B, seq_len, input_dim)   e.g. (B, 50, 49)
    Output: (B, latent_dim)           e.g. (B, 32)

    Uses the final hidden states from both directions, concatenated and
    projected to the latent dimension.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Concat forward + backward final hidden → 2 * hidden_size
        self.projection = nn.Linear(2 * hidden_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h_n: (num_layers * 2, B, hidden_size) for bidirectional
        _, h_n = self.gru(x)
        # Take last layer: forward = h_n[-2], backward = h_n[-1]
        h_fwd = h_n[-2]  # (B, hidden_size)
        h_bwd = h_n[-1]  # (B, hidden_size)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, 2*hidden_size)
        z0 = self.projection(h_cat)  # (B, latent_dim)
        return z0


class ODEFunc(nn.Module):
    """Dynamics function f_θ(z, t) for the Neural ODE.

    Defines dz/dt = f_θ(z, t) where f_θ is a small MLP that takes the
    current state z and time t as input. The ODE solver (dopri5) calls
    this function adaptively to integrate z from t=0 to t=1.

    Input:  t (scalar), z (B, latent_dim)
    Output: dz/dt (B, latent_dim)
    """

    def __init__(self, latent_dim: int = 32, hidden_size: int = 128):
        super().__init__()
        # +1 for time concatenation
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, latent_dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Expand t to match batch dimension: scalar → (B, 1)
        t_expanded = t.expand(z.shape[0], 1)
        z_t = torch.cat([z, t_expanded], dim=-1)  # (B, latent_dim + 1)
        return self.net(z_t)


class MLPDecoder(nn.Module):
    """MLP decoder that reconstructs flow windows from latent vectors.

    Input:  (B, latent_dim)                    e.g. (B, 32)
    Output: (B, seq_len, input_dim)            e.g. (B, 50, 49)

    No output activation — targets are RobustScaler-transformed and unbounded.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_size: int = 128,
        seq_len: int = 50,
        input_dim: int = 49,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, seq_len * input_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)  # (B, seq_len * input_dim)
        return out.view(-1, self.seq_len, self.input_dim)  # (B, seq_len, input_dim)


class NeuralODEAutoencoder(nn.Module):
    """Full Neural ODE Autoencoder for anomaly detection.

    Pipeline:
        x → BiGRUEncoder → z₀ → ODE(t=0→1) → z₁ → MLPDecoder → x̂

    The anomaly score for a window is the mean squared reconstruction error:
        score = (1 / (seq_len × input_dim)) × ||x - x̂||²_F
    """

    def __init__(self, config: dict):
        super().__init__()
        enc_cfg = config["model"]["encoder"]
        ode_cfg = config["model"]["neural_ode"]
        dec_cfg = config["model"]["decoder"]
        latent_dim = config["model"]["latent_dim"]
        window_size = config["preprocessing"]["window_size"]

        # Number of input features (49 after feature selection)
        # Inferred at first forward pass or set explicitly
        self.input_dim = None

        self.encoder = BiGRUEncoder(
            input_dim=0,  # placeholder, set in _init_input_dim
            hidden_size=enc_cfg["hidden_size"],
            num_layers=enc_cfg["num_layers"],
            latent_dim=latent_dim,
        )

        self.ode_func = ODEFunc(
            latent_dim=latent_dim,
            hidden_size=ode_cfg["hidden_size"],
        )

        self.decoder = MLPDecoder(
            latent_dim=latent_dim,
            hidden_size=dec_cfg["hidden_size"],
            seq_len=window_size,
            input_dim=0,  # placeholder, set in _init_input_dim
        )

        # ODE solver settings
        self.solver = ode_cfg["solver"]
        self.atol = ode_cfg["atol"]
        self.rtol = ode_cfg["rtol"]
        self.integration_time = torch.tensor(
            [0.0, ode_cfg["integration_time"]], dtype=torch.float32
        )

        self._initialized = False

    def _init_input_dim(self, input_dim: int):
        """Lazily initialize layers that depend on the input feature dimension."""
        if self._initialized:
            return
        device = self.integration_time.device

        self.input_dim = input_dim
        self.encoder = BiGRUEncoder(
            input_dim=input_dim,
            hidden_size=self.encoder.gru.hidden_size,
            num_layers=self.encoder.gru.num_layers,
            latent_dim=self.encoder.projection.out_features,
        ).to(device)

        self.decoder = MLPDecoder(
            latent_dim=self.decoder.net[0].in_features,
            hidden_size=self.decoder.net[0].out_features,
            seq_len=self.decoder.seq_len,
            input_dim=input_dim,
        ).to(device)

        self._initialized = True

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the full autoencoder.

        Args:
            x: Input windows (B, seq_len, input_dim)

        Returns:
            x_hat: Reconstructed windows (B, seq_len, input_dim)
            z0: Latent initial state (B, latent_dim) — useful for visualization
        """
        self._init_input_dim(x.shape[-1])

        # Encode
        z0 = self.encoder(x)  # (B, latent_dim)

        # Integrate ODE from t=0 to t=T
        t_span = self.integration_time.to(x.device)
        z_traj = odeint(
            self.ode_func,
            z0,
            t_span,
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol,
        )  # (2, B, latent_dim) — states at t=0 and t=T
        z1 = z_traj[-1]  # (B, latent_dim) — state at t=T

        # Decode
        x_hat = self.decoder(z1)  # (B, seq_len, input_dim)

        return x_hat, z0

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-window anomaly score (mean squared error).

        Args:
            x: Input windows (B, seq_len, input_dim)

        Returns:
            scores: Per-window MSE (B,)
        """
        x_hat, _ = self.forward(x)
        # MSE per window: mean over seq_len and input_dim
        scores = ((x - x_hat) ** 2).mean(dim=(1, 2))
        return scores
