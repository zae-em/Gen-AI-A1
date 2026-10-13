"""Style-conditioned conditional GAN for face-to-sketch generation (Task 4).

Generator: U-Net encoder-decoder, pix2pix style.
Discriminator: PatchGAN over (photo, sketch) pairs.

Style conditioning
------------------
The three FS2K styles are a learned `nn.Embedding`, injected in two places:

  1. broadcast over the spatial grid and concatenated to the input image, so
     early filters can specialise per style;
  2. projected and added to the bottleneck activation, so the style also
     reaches the global structure of the sketch.

The discriminator receives the same embedding, broadcast and concatenated with
the photo/sketch pair.  Without that, the discriminator cannot tell a Style 1
sketch paired with a Style 3 condition from a correct pair, and the generator
is free to ignore the condition entirely.

Both images live in [-1, 1]; the generator output is tanh.
"""

from __future__ import annotations

import torch
import torch.nn as nn

N_STYLES = 3


def down(in_ch: int, out_ch: int, norm: bool = True) -> nn.Sequential:
    layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up(in_ch: int, out_ch: int, dropout: float = 0.0) -> nn.Sequential:
    layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
              nn.BatchNorm2d(out_ch),
              nn.ReLU(inplace=True)]
    if dropout > 0:
        layers.insert(2, nn.Dropout(dropout))
    return nn.Sequential(*layers)


class StyleUNetGenerator(nn.Module):
    """128x128 U-Net with 6 down / 6 up stages (bottleneck is 2x2)."""

    def __init__(self, base_channels: int = 64, style_dim: int = 32,
                 dropout: float = 0.5, in_ch: int = 3, out_ch: int = 3):
        super().__init__()
        self.style_dim = style_dim
        self.style_embedding = nn.Embedding(N_STYLES, style_dim)

        b = base_channels
        self.d1 = down(in_ch + style_dim, b, norm=False)   # 64
        self.d2 = down(b, b * 2)                           # 32
        self.d3 = down(b * 2, b * 4)                       # 16
        self.d4 = down(b * 4, b * 8)                       # 8
        self.d5 = down(b * 8, b * 8)                       # 4
        self.d6 = down(b * 8, b * 8, norm=False)           # 2

        self.style_to_bottleneck = nn.Linear(style_dim, b * 8)

        self.u1 = up(b * 8, b * 8, dropout)                # 4
        self.u2 = up(b * 16, b * 8, dropout)               # 8
        self.u3 = up(b * 16, b * 4, dropout)               # 16
        self.u4 = up(b * 8, b * 2)                         # 32
        self.u5 = up(b * 4, b)                             # 64
        self.u6 = nn.Sequential(                           # 128
            nn.ConvTranspose2d(b * 2, out_ch, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, photo: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        emb = self.style_embedding(style.long().view(-1))              # (N, S)
        emb_map = emb[:, :, None, None].expand(-1, -1, photo.shape[2], photo.shape[3])
        x = torch.cat([photo, emb_map], dim=1)

        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)

        d6 = d6 + self.style_to_bottleneck(emb)[:, :, None, None]

        u1 = self.u1(d6)
        u2 = self.u2(torch.cat([u1, d5], 1))
        u3 = self.u3(torch.cat([u2, d4], 1))
        u4 = self.u4(torch.cat([u3, d3], 1))
        u5 = self.u5(torch.cat([u4, d2], 1))
        return self.u6(torch.cat([u5, d1], 1))


class StylePatchDiscriminator(nn.Module):
    """70x70-receptive-field PatchGAN over (photo, style, sketch)."""

    def __init__(self, base_channels: int = 64, style_dim: int = 32,
                 in_ch: int = 6):
        super().__init__()
        self.style_embedding = nn.Embedding(N_STYLES, style_dim)
        b = base_channels
        self.net = nn.Sequential(
            down(in_ch + style_dim, b, norm=False),
            down(b, b * 2),
            down(b * 2, b * 4),
            nn.Conv2d(b * 4, b * 8, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(b * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(b * 8, 1, 4, stride=1, padding=1),   # logits, no sigmoid
        )

    def forward(self, photo, style, sketch):
        emb = self.style_embedding(style.long().view(-1))
        emb_map = emb[:, :, None, None].expand(-1, -1, photo.shape[2], photo.shape[3])
        return self.net(torch.cat([photo, sketch, emb_map], dim=1))


def init_weights(module: nn.Module, gain: float = 0.02) -> None:
    """DCGAN-style N(0, 0.02) initialisation, as used in the pix2pix paper."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, gain)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.1)


def build_generator(cfg: dict) -> StyleUNetGenerator:
    return StyleUNetGenerator(
        base_channels=cfg.get("base_channels", 64),
        style_dim=cfg.get("style_dim", 32),
        dropout=cfg.get("dropout", 0.5),
    )


def build_discriminator(cfg: dict) -> StylePatchDiscriminator:
    return StylePatchDiscriminator(
        base_channels=cfg.get("base_channels", 64),
        style_dim=cfg.get("style_dim", 32),
    )
