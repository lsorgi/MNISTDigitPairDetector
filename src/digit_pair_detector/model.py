import torch
import torch.nn as nn
from typing import Tuple, List
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ModelConfig:
    input_size: Tuple[int, int, int] = (28, 28, 1)
    filter_count: int = 32
    dropout: float = 0.5
    kernel_size: int = 5
    split_features: bool = False
    class_count: int = -1  # automatically set

    def __post_init__(self):
        self.class_count = 10
        if len(self.input_size) != 3:
            raise ValueError(f'Invalid input shape len ({len(self.input_size)})')
        if any([v < 0 for v in self.input_size]):
            raise ValueError(f'Invalid input shape ({self.input_size})')
        if self.filter_count <= 0:
            raise ValueError(f'Invalid feature count {self.filter_count}')
        if self.filter_count <= 0:
            raise ValueError(f'Invalid class count {self.class_count}')
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError(f'Invalid dropout probability ({self.cls_dropout})')


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        m.reset_parameters()


class ResBlock(nn.Module):

    def __init__(self, channels: int, kernel_size: int = 3):

        super().__init__()

        self._channels = channels

        self._res_module = nn.Sequential(
            nn.Conv2d(in_channels=self._channels,
                      out_channels=self._channels,
                      kernel_size=(kernel_size, kernel_size),
                      padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(num_features=self._channels),
        )

        self._res_module.apply(_init_weights)

    def forward(self, x: torch.Tensor):
        return self._res_module(x) + x


class MyDigitPairDetector(nn.Module):

    def __init__(self, cfg: ModelConfig):

        super().__init__()

        self._cfg = cfg
        self._enable_probs = True

        self._encoder = nn.Sequential(
            nn.Conv2d(in_channels=self._cfg.input_size[-1],
                      out_channels=self._cfg.filter_count,
                      kernel_size=(self._cfg.kernel_size, self._cfg.kernel_size),
                      padding='same'),
            ResBlock(channels=self._cfg.filter_count,
                     kernel_size=self._cfg.kernel_size),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=self._cfg.filter_count,
                      out_channels=2 * self._cfg.filter_count,
                      kernel_size=(1, 1),
                      padding='same'),
            ResBlock(channels=2 * self._cfg.filter_count,
                     kernel_size=self._cfg.kernel_size),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=2 * self._cfg.filter_count,
                      out_channels=4 * self._cfg.filter_count,
                      kernel_size=(1, 1),
                      padding='same'),
            ResBlock(channels=4 * self._cfg.filter_count,
                     kernel_size=self._cfg.kernel_size),
            # encoder head
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Dropout(p=cfg.dropout, inplace=True) if cfg.dropout > 0 else nn.Identity()
        )

        feature_count_multiplier = 2 if self._cfg.split_features else 4
        self._head0 = nn.Sequential(nn.Linear(in_features=feature_count_multiplier * self._cfg.filter_count,
                                              out_features=self._cfg.class_count,
                                              bias=True),
                                    nn.LeakyReLU())
        self._head1 = nn.Sequential(nn.Linear(in_features=feature_count_multiplier * self._cfg.filter_count,
                                              out_features=self._cfg.class_count,
                                              bias=True),
                                    nn.LeakyReLU())

        self._probs = nn.Softmax()

        self._encoder.apply(_init_weights)
        self._head0.apply(_init_weights)
        self._head1.apply(_init_weights)

    def enable_probabilities(self, flag: bool):
        self._enable_probs = flag

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self._encoder(x)

        if self._cfg.split_features:
            # split the feature vector and use the same head
            split_index = features.size()[1] // 2
            logits0 = self._head0(features[:, split_index:])
            logits1 = self._head0(features[:, :split_index])
        else:
            # full feature vector through different heads
            logits0 = self._head0(features)
            logits1 = self._head1(features)

        if self._enable_probs:
            probs0 = self._probs(logits0)
            probs1 = self._probs(logits1)
            return [probs0, probs1]
        else:
            return [logits0, logits1]
