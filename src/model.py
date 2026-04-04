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
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        latent_dim: int = 32,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        concat_size = 2 * hidden_size
        self.norm = nn.LayerNorm(concat_size) if layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(concat_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        # Last layer: forward = h_n[-2], backward = h_n[-1]
        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, 2*hidden_size)
        h_cat = self.norm(h_cat)
        h_cat = self.dropout(h_cat)
        return self.projection(h_cat)  # (B, latent_dim)


class ODEFunc(nn.Module):
    """Dynamics function f_θ(z, t) for the Neural ODE.

    Defines dz/dt = f_θ(z, t). No dropout — the adaptive solver calls this
    multiple times per step, so stochastic dropout would break error estimation.
    LayerNorm is safe since it's deterministic and per-sample.

    Input:  t (scalar), z (B, latent_dim)
    Output: dz/dt (B, latent_dim)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_size: int = 128,
        num_layers: int = 3,
        layer_norm: bool = False,
    ):
        super().__init__()
        layers = []
        in_dim = latent_dim + 1  # +1 for time concatenation
        for i in range(num_layers - 1):
            out_dim = hidden_size
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.SiLU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        t_expanded = t.expand(z.shape[0], 1)
        z_t = torch.cat([z, t_expanded], dim=-1)
        return self.net(z_t)


class MLPDecoder(nn.Module):
    """MLP decoder that reconstructs flow windows from latent vectors.

    Input:  (B, latent_dim)           e.g. (B, 32)
    Output: (B, seq_len, input_dim)   e.g. (B, 50, 49)

    No output activation — targets are RobustScaler-transformed and unbounded.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_size: int = 256,
        num_layers: int = 3,
        seq_len: int = 50,
        input_dim: int = 49,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

        layers = []
        in_dim = latent_dim
        for i in range(num_layers - 1):
            out_dim = hidden_size
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, seq_len * input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(-1, self.seq_len, self.input_dim)


class NeuralODEAutoencoder(nn.Module):
    """Full Neural ODE Autoencoder for anomaly detection.

    Pipeline:
        x → BiGRUEncoder → z₀ → ODE(t=0→1) → z₁ → MLPDecoder → x̂

    Anomaly score = per-window MSE: (1/(T·D)) × ||x - x̂||²_F
    """

    def __init__(self, config: dict):
        super().__init__()
        enc_cfg = config["model"]["encoder"]
        ode_cfg = config["model"]["neural_ode"]
        dec_cfg = config["model"]["decoder"]
        latent_dim = config["model"]["latent_dim"]
        window_size = config["preprocessing"]["window_size"]
        input_dim = config["model"].get("input_dim", 49)

        self.encoder = BiGRUEncoder(
            input_dim=input_dim,
            hidden_size=enc_cfg["hidden_size"],
            num_layers=enc_cfg["num_layers"],
            latent_dim=latent_dim,
            dropout=enc_cfg.get("dropout", 0.0),
            layer_norm=enc_cfg.get("layer_norm", False),
        )

        self.ode_func = ODEFunc(
            latent_dim=latent_dim,
            hidden_size=ode_cfg["hidden_size"],
            num_layers=ode_cfg.get("num_layers", 3),
            layer_norm=ode_cfg.get("layer_norm", False),
        )

        self.decoder = MLPDecoder(
            latent_dim=latent_dim,
            hidden_size=dec_cfg["hidden_size"],
            num_layers=dec_cfg.get("num_layers", 3),
            seq_len=window_size,
            input_dim=input_dim,
            dropout=dec_cfg.get("dropout", 0.0),
            layer_norm=dec_cfg.get("layer_norm", False),
        )

        self.solver = ode_cfg["solver"]
        self.atol = ode_cfg["atol"]
        self.rtol = ode_cfg["rtol"]
        self.integration_time = torch.tensor(
            [0.0, ode_cfg["integration_time"]], dtype=torch.float32
        )

        # Kinetic energy regularization: number of time points for quadrature
        self.ke_n_steps = ode_cfg.get("ke_n_steps", 10)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            x_hat: Reconstructed windows (B, T, D)
            z0: Initial latent state (B, latent_dim)
            ke_reg: Kinetic energy regularization scalar
        """
        z0 = self.encoder(x)

        # Integrate with multiple time points for kinetic energy estimation
        t_span = torch.linspace(0, self.integration_time[-1].item(),
                                self.ke_n_steps, device=x.device)
        z_traj = odeint(
            self.ode_func, z0, t_span,
            method=self.solver, atol=self.atol, rtol=self.rtol,
        )  # (ke_n_steps, B, latent_dim)
        z1 = z_traj[-1]  # (B, latent_dim)

        # Kinetic energy: ∫₀¹ ||f_θ(z(t), t)||² dt ≈ (1/K) Σ ||f_θ(z_k, t_k)||²
        ke_reg = self._kinetic_energy(z_traj, t_span)

        x_hat = self.decoder(z1)
        return x_hat, z0, ke_reg

    def _kinetic_energy(
        self, z_traj: torch.Tensor, t_span: torch.Tensor
    ) -> torch.Tensor:
        """Compute kinetic energy along the ODE trajectory.

        Uses the trapezoidal rule to approximate:
            R_KE = (1/B) ∫₀¹ ||f_θ(z(t), t)||² dt

        Args:
            z_traj: (K, B, latent_dim) — latent states at K time points
            t_span: (K,) — time points

        Returns:
            Scalar kinetic energy (averaged over batch).
        """
        K = t_span.shape[0]
        velocities_sq = []
        for k in range(K):
            v = self.ode_func(t_span[k], z_traj[k])  # (B, latent_dim)
            velocities_sq.append((v ** 2).sum(dim=-1))  # (B,)
        # Stack: (K, B), apply trapezoidal rule over time, then average over batch
        v_sq = torch.stack(velocities_sq, dim=0)  # (K, B)
        dt = t_span[1:] - t_span[:-1]  # (K-1,)
        # Trapezoidal: ∫ ≈ Σ (f(t_k) + f(t_{k+1})) / 2 * dt_k
        integral = (0.5 * (v_sq[:-1] + v_sq[1:]) * dt.unsqueeze(1)).sum(dim=0)  # (B,)
        return integral.mean()

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-window MSE anomaly score (B,)."""
        x_hat, _, _ = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=(1, 2))
