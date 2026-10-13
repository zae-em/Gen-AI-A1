"""Convolutional denoising autoencoder.

Used twice: as the single universal model in Task 1 and as the architecture of
the three corruption specialists in Task 2.

Bottleneck
----------
At 128x128 input the encoder halves the resolution four times, so the latent
tensor is (latent_dim, 8, 8).  With latent_dim = 128 that is 8192 values
against 49152 input values, a 6x compression - a genuine bottleneck rather
than a wide identity path.  `compression_ratio()` reports the exact figure for
whatever configuration Optuna selects, which is the number to quote in the
report.

Skip connections are off by default.  A single coarse skip (the deepest
encoder stage only) can be enabled with `use_skip=True`; it is deliberately
*not* a U-Net, because full-resolution skips would let the network copy the
corrupted input straight to the output and defeat the exercise.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    # Upsample + conv instead of ConvTranspose2d: same receptive growth without
    # the checkerboard artefacts transposed convolutions produce on smooth fur.
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class DenoisingAutoencoder(nn.Module):
    def __init__(self, base_channels: int = 32, latent_dim: int = 128,
                 dropout: float = 0.1, depth: int = 4, image_size: int = 128,
                 use_skip: bool = False):
        super().__init__()
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.depth = depth
        self.image_size = image_size
        self.use_skip = use_skip

        chans = [base_channels * (2 ** i) for i in range(depth)]

        enc = []
        in_ch = 3
        for ch in chans:
            enc.append(nn.Sequential(conv_block(in_ch, ch), conv_block(ch, ch, stride=2)))
            in_ch = ch
        self.encoder_stages = nn.ModuleList(enc)

        self.to_latent = nn.Sequential(
            nn.Conv2d(in_ch, latent_dim, 1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
        )
        self.from_latent = conv_block(latent_dim, in_ch)

        dec = []
        rev = list(reversed(chans))
        for i, ch in enumerate(rev):
            out_ch = rev[i + 1] if i + 1 < len(rev) else base_channels
            first_in = ch * 2 if (i == 0 and use_skip) else ch
            dec.append(nn.Sequential(up_block(first_in, out_ch), conv_block(out_ch, out_ch)))
        self.decoder_stages = nn.ModuleList(dec)

        self.head = nn.Sequential(nn.Conv2d(base_channels, 3, 3, padding=1), nn.Sigmoid())

    # -- introspection ------------------------------------------------------
    def latent_spatial(self) -> int:
        return self.image_size // (2 ** self.depth)

    def compression_ratio(self) -> float:
        s = self.latent_spatial()
        return (3 * self.image_size ** 2) / (self.latent_dim * s * s)

    # -- forward ------------------------------------------------------------
    def encode(self, x):
        skip = None
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if self.use_skip and i == len(self.encoder_stages) - 1:
                skip = x
        return self.to_latent(x), skip

    def decode(self, z, skip=None):
        x = self.from_latent(z)
        for i, stage in enumerate(self.decoder_stages):
            if i == 0 and self.use_skip and skip is not None:
                x = torch.cat([x, skip], dim=1)
            x = stage(x)
        return self.head(x)

    def forward(self, x):
        z, skip = self.encode(x)
        return self.decode(z, skip)


def build_autoencoder(cfg: dict) -> DenoisingAutoencoder:
    return DenoisingAutoencoder(
        base_channels=cfg.get("base_channels", 32),
        latent_dim=cfg.get("latent_dim", 128),
        dropout=cfg.get("dropout", 0.1),
        depth=cfg.get("depth", 4),
        image_size=cfg.get("image_size", 128),
        use_skip=cfg.get("use_skip", False),
    )
