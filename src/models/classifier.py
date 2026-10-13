"""Four-class corruption classifier (Task 2), reused as the gate in Task 3.

The same class is used for both so that the Task 3 gate can be initialised
directly from the trained Task 2 checkpoint with a plain `load_state_dict` -
the assignment requires that the joint training does not start from random
components.

Design note: the first block deliberately keeps stride 1 so that isolated
salt-and-pepper pixels survive into the feature maps.  An aggressive stride-2
stem averages them away and the clean / salt-pepper confusion goes up sharply
at p = 0.02.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.corruptions import CORRUPTION_NAMES


class CorruptionClassifier(nn.Module):
    def __init__(self, base_channels: int = 32, dropout: float = 0.3,
                 n_classes: int = len(CORRUPTION_NAMES), depth: int = 4):
        super().__init__()
        self.base_channels = base_channels
        self.depth = depth
        chans = [base_channels * (2 ** i) for i in range(depth)]

        layers = [
            nn.Conv2d(3, chans[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(chans[0]),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_ch = chans[0]
        for ch in chans:
            layers += [
                nn.Conv2d(in_ch, ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = ch
        self.features = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, in_ch // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_ch // 2, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class ClassifierWithProbs(nn.Module):
    """Export wrapper: returns (logits, probabilities).

    The application shows the four probabilities, so the softmax is baked into
    the ONNX graph rather than being reimplemented in the backend.
    """

    def __init__(self, classifier: CorruptionClassifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, x):
        logits = self.classifier(x)
        return logits, torch.softmax(logits, dim=1)


def build_classifier(cfg: dict) -> CorruptionClassifier:
    return CorruptionClassifier(
        base_channels=cfg.get("base_channels", 32),
        dropout=cfg.get("dropout", 0.3),
        depth=cfg.get("depth", 4),
    )
