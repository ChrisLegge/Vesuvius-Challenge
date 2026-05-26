"""
3D Residual U-Net for Vesuvius surface detection.

Architecture overview
---------------------
Encoder–decoder with skip connections. Each encoder stage applies two
residual convolution blocks then halves spatial dimensions with strided
convolution. The decoder mirrors this with transposed convolution upsampling
and skip-connection concatenation.

The design choices reflect the constraints of the competition:

  - 3D convolutions throughout: scroll CT is genuinely volumetric; 2D
    slice-by-slice models lose inter-slice continuity information that
    matters for topology preservation.

  - Residual blocks: allow gradients to propagate cleanly through the
    ~20-layer depth needed to have a receptive field spanning a full
    192-voxel patch. Without residuals, the late-phase topology losses
    (which provide sparse, low-magnitude gradients) fail to back-propagate
    to the encoder.

  - Instance normalisation instead of batch norm: patch-based training
    with batch size 1–2 makes batch statistics unstable. Instance norm
    normalises per-channel per-sample, which is robust to small batches.

  - Configurable base features: the three specialists use different memory
    budgets. The anti-merge specialist uses base_features=32 to fit a
    160³ patch at the same GPU memory as base_features=48 at 192³.

  - Single-channel sigmoid output: binary segmentation. The composite
    loss (BCE + Dice + topology terms) operates on logits, not
    probabilities — the final sigmoid is applied only at inference.

Specialist configurations used in training
------------------------------------------
  Model A (generalist):         patch=192, base_features=48, depth=4
  Model B (anti-merge):         patch=160, base_features=32, depth=4
  Model C (surface specialist): patch=192, base_features=48, depth=4,
                                 dropout=0.1

Usage
-----
  from vesuvius.model import build_model, ResidualUNet3D

  # Instantiate (no weights — architecture only)
  model = build_model(patch_size=192, base_features=48)

  # Forward pass
  x = torch.randn(1, 1, 192, 192, 192)
  logits = model(x)   # shape: (1, 1, 192, 192, 192)

  # Count parameters
  n_params = sum(p.numel() for p in model.parameters())
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock3D(nn.Module):
    """
    Two-layer 3×3×3 residual block with instance normalisation.

    Structure:
        x → Conv3d → IN → ReLU → Dropout → Conv3d → IN → (+x) → ReLU

    The identity shortcut uses a 1×1×1 convolution when in_channels ≠
    out_channels (projection shortcut, following He et al. 2016).
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        dropout:      float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels,  out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.in1   = nn.InstanceNorm3d(out_channels, affine=True)
        self.in2   = nn.InstanceNorm3d(out_channels, affine=True)
        self.drop  = nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity()

        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.in1(self.conv1(x)),  inplace=True)
        out = self.drop(out)
        out = self.in2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class EncoderBlock(nn.Module):
    """
    One encoder stage: ResidualBlock3D then strided-conv downsampling.

    Returns (skip, downsampled) so the skip connection can be stored
    separately from the feature map passed to the next stage.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        dropout:      float = 0.0,
    ) -> None:
        super().__init__()
        self.res   = ResidualBlock3D(in_channels,  out_channels, dropout)
        self.down  = nn.Conv3d(out_channels, out_channels, 2, stride=2, bias=False)
        self.in_dn = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.res(x)
        down = F.relu(self.in_dn(self.down(skip)), inplace=True)
        return skip, down


class DecoderBlock(nn.Module):
    """
    One decoder stage: transposed-conv upsampling then ResidualBlock3D.

    The skip connection from the corresponding encoder stage is
    concatenated channel-wise before the residual block, following the
    standard U-Net design.
    """

    def __init__(
        self,
        in_channels:  int,
        skip_channels: int,
        out_channels: int,
        dropout:      float = 0.0,
    ) -> None:
        super().__init__()
        self.up  = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2,
                                       bias=False)
        self.res = ResidualBlock3D(in_channels + skip_channels, out_channels,
                                   dropout)

    def forward(
        self,
        x:    torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)
        # Crop skip if spatial dims differ (padding artefacts from odd sizes)
        if x.shape != skip.shape:
            skip = skip[
                :, :,
                : x.shape[2],
                : x.shape[3],
                : x.shape[4],
            ]
        x = torch.cat([x, skip], dim=1)
        return self.res(x)


# ---------------------------------------------------------------------------
# Full architecture
# ---------------------------------------------------------------------------

class ResidualUNet3D(nn.Module):
    """
    3D Residual U-Net for binary surface segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Typically 1 (CT intensity).
    base_features : int
        Number of feature maps in the first encoder stage. Doubled at
        each subsequent stage (standard U-Net feature pyramid).
    depth : int
        Number of encoder/decoder stages (excluding bottleneck).
        Controls the receptive field:
          depth=4 → receptive field ~ 2^4 = 16× the kernel size.
    dropout : float
        Dropout probability in ResidualBlock3D. Applied in mid/late
        training phases; typically 0.0 (generalist) or 0.1 (surface
        specialist) in this project.

    Input / output
    --------------
    Input  : (B, in_channels, D, H, W)  — float32, normalised CT patch
    Output : (B, 1, D, H, W)            — raw logits (apply sigmoid for
                                           inference probabilities)

    Example parameter counts
    ------------------------
    base_features=32, depth=4 : ~3.1M parameters  (Model B, 160³ patch)
    base_features=48, depth=4 : ~6.9M parameters  (Models A and C, 192³)
    """

    def __init__(
        self,
        in_channels:   int   = 1,
        base_features: int   = 48,
        depth:         int   = 4,
        dropout:       float = 0.0,
    ) -> None:
        super().__init__()
        self.depth = depth

        # Feature map sizes at each depth level
        features = [base_features * (2 ** i) for i in range(depth + 1)]
        # features = [48, 96, 192, 384, 768] for base=48, depth=4

        # Stem: bring input channels to base_features
        self.stem = ResidualBlock3D(in_channels, features[0], dropout)

        # Encoder
        self.encoders = nn.ModuleList([
            EncoderBlock(features[i], features[i + 1], dropout)
            for i in range(depth)
        ])

        # Bottleneck
        self.bottleneck = ResidualBlock3D(features[depth], features[depth],
                                          dropout)

        # Decoder (reversed).
        # Each encoder stage i produces a skip with features[i+1] channels
        # (the output of its ResidualBlock3D). The decoder at position i
        # receives the skip from encoder[depth-1-i], which has
        # features[depth-i] channels — equal to the upsampled input channels.
        self.decoders = nn.ModuleList([
            DecoderBlock(
                in_channels   = features[depth - i],
                skip_channels = features[depth - i],   # skip == in channels
                out_channels  = features[depth - i - 1],
                dropout       = dropout,
            )
            for i in range(depth)
        ])

        # Output head: 1×1×1 conv → single-channel logit
        self.head = nn.Conv3d(features[0], 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming (He) initialisation for conv layers; zero-bias where applicable."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d) and m.weight is not None:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        # Encode — store skips
        skips: list[torch.Tensor] = []
        for enc in self.encoders:
            skip, x = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        # Decode — consume skips in reverse order
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.head(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_model(
    patch_size:    int   = 192,
    base_features: int   = 48,
    depth:         int   = 4,
    in_channels:   int   = 1,
    dropout:       float = 0.0,
) -> ResidualUNet3D:
    """
    Construct a ResidualUNet3D matching the competition training configuration.

    Patch size is used only as documentation; the model is fully
    convolutional and accepts any spatial dimensions divisible by 2^depth.
    The recommended patch sizes are:
      - 192 for Model A (generalist) and Model C (surface specialist)
      - 160 for Model B (anti-merge)

    Parameters
    ----------
    patch_size : int
        Training patch size in voxels (isotropic). Not used by the model
        itself; included to make the configuration explicit.
    base_features : int
        Number of feature maps in the first encoder stage.
    depth : int
        Encoder/decoder depth (number of downsampling stages).
    in_channels : int
        Input channels (1 for single-channel CT intensity).
    dropout : float
        Per-block dropout rate (0.0 = disabled).

    Returns
    -------
    ResidualUNet3D
        Randomly initialised model. Load trained weights via
        ``model.load_state_dict(torch.load(path))``.

    Examples
    --------
    >>> model_a = build_model(patch_size=192, base_features=48)
    >>> model_b = build_model(patch_size=160, base_features=32)
    >>> model_c = build_model(patch_size=192, base_features=48, dropout=0.1)
    """
    _ = patch_size   # documented but unused — model is fully convolutional
    return ResidualUNet3D(
        in_channels   = in_channels,
        base_features = base_features,
        depth         = depth,
        dropout       = dropout,
    )
