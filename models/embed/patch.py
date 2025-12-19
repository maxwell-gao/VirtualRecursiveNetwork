"""
Patch Embedding modules for Vision-based models.

- StandardPatchEmbed: Standard ViT patch embedding for discrete tokens
- MetricPatchEmbed: Finsler-Randers geometry-based adaptive patch embedding
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metric import (
    _finsler_randers,
    _sample_tangent_ball,
    HAS_TORCHVISION,
)


class StandardPatchEmbed(nn.Module):
    """Standard ViT patch embedding for discrete tokens."""

    def __init__(
        self, image_size: int, patch_size: int, vocab_size: int, hidden_size: int
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.hidden_size = hidden_size

        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Conv2d(hidden_size, hidden_size, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len) integer tokens
        Returns:
            (B, num_patches, hidden_size)
        """
        B = x.shape[0]
        if x.dim() == 2:
            x = x.view(B, self.image_size, self.image_size)
        x = self.token_embed(x.long()).permute(0, 3, 1, 2)
        return self.proj(x).flatten(2).transpose(1, 2)


class MetricPatchEmbed(nn.Module):
    """Patch embedding with learnable Finsler-Randers metric for adaptive boundaries."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        vocab_size: int,
        hidden_size: int,
        eps_w: float = 0.5,
    ):
        super().__init__()
        assert HAS_TORCHVISION, "torchvision required for MetricPatchEmbed"

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.eps_w = eps_w

        # Predict metric: 4 (M) + 2 (w) + 1 (scale) = 7 channels
        self.metric_net = nn.Conv2d(vocab_size, 7, patch_size, stride=patch_size)
        self.proj_weight = nn.Parameter(
            torch.randn(hidden_size, vocab_size, patch_size, patch_size)
            / (patch_size * math.sqrt(vocab_size))
        )
        self.proj_bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len) integer tokens
        Returns:
            (B, num_patches, hidden_size)
        """
        B = x.shape[0]
        if x.dim() == 2:
            x = x.view(B, self.image_size, self.image_size)

        # Convert to one-hot for convolution
        x = F.one_hot(x.long(), self.vocab_size).permute(0, 3, 1, 2).float()

        P = self.patch_size
        params = self.metric_net(x)

        # Build positive-definite M
        v = F.normalize(params[:, :2].permute(0, 2, 3, 1), dim=-1)
        eigvals = 2 * torch.sigmoid(params[:, 2:4].permute(0, 2, 3, 1))
        scale = 0.5 + 1.5 * torch.sigmoid(params[:, 4:5].permute(0, 2, 3, 1))
        eigvals = eigvals * scale

        R = torch.stack([v, torch.stack([-v[..., 1], v[..., 0]], -1)], -1)
        M = R @ torch.diag_embed(eigvals) @ R.transpose(-1, -2)
        M = M.permute(0, 3, 4, 1, 2).reshape(B, 4, *params.shape[-2:])

        # Build drift w
        w = params[:, 5:7]
        if self.eps_w < 1.0:
            w = w * (1 - self.eps_w) * torch.sigmoid(w.norm(dim=1, keepdim=True))
        else:
            w = torch.zeros_like(w)

        # Sample tangent ball and compute offsets
        y = _sample_tangent_ball(B, M, w, P, P, x.device)

        grid = torch.stack(
            torch.meshgrid(
                torch.arange(-(P // 2), P // 2 + 1, device=x.device),
                torch.arange(-(P // 2), P // 2 + 1, device=x.device),
                indexing="ij",
            ),
            -1,
        ).float()

        offsets = (y.permute(0, 4, 5, 1, 2, 3).flip(-1) - grid).permute(
            0, 3, 4, 5, 1, 2
        )
        offsets = offsets.reshape(B, -1, *params.shape[-2:])

        out = torchvision.ops.deform_conv2d(
            x, offsets, self.proj_weight, self.proj_bias, stride=P
        )
        return out.flatten(2).transpose(1, 2)


def create_patch_embed(config) -> nn.Module:
    """Factory function to create patch embedding based on config."""
    image_size = config.image_size or int(math.sqrt(config.seq_len))

    if getattr(config, "patch_embed_type", "none") == "none":
        return None
    elif config.patch_embed_type == "standard":
        return StandardPatchEmbed(
            image_size=image_size,
            patch_size=config.patch_size,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )
    elif config.patch_embed_type == "metric":
        return MetricPatchEmbed(
            image_size=image_size,
            patch_size=config.patch_size,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            eps_w=getattr(config, "metric_eps_w", 0.5),
        )
    else:
        raise ValueError(f"Unknown patch_embed_type: {config.patch_embed_type}")
