import torch
import torch.nn as nn


class SpeakerPredictor(nn.Module):
    """Speaker embedding projection module."""

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 256,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        use_mean_std: bool = True,
    ):
        super().__init__()
        proj_in = input_dim * (2 if use_mean_std else 1)

        self.use_mean_std = use_mean_std
        self.proj = nn.Sequential(
            nn.Linear(proj_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape [B, D, T]
        Returns:
            proj_feat (torch.Tensor): shape [B, output_dim]
        """
        sim_mean = x.mean(dim=2)  # [B, D]
        if self.use_mean_std:
            sim_std = x.std(dim=2)  # [B, D]
            feat = torch.cat([sim_mean, sim_std], dim=-1)
        else:
            feat = sim_mean
        return self.proj(feat)
