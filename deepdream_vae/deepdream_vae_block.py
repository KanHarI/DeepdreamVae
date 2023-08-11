import dataclasses
from typing import Callable

import torch


@dataclasses.dataclass
class DeepdreamVAEBlockConfig:
    n_layers: int
    n_channels_in: int
    n_channels_out: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    dropout: float
    init_std: float


class DeepdreamVAEBlock(torch.nn.Module):
    def __init__(self, config: DeepdreamVAEBlockConfig):
        super().__init__()
        self.config = config
        self.first_conv = torch.nn.Conv2d(
            in_channels=self.config.n_channels_in,
            out_channels=self.config.n_channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.conv_layers = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    in_channels=self.config.n_channels_out,
                    out_channels=self.config.n_channels_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
                for _ in range(self.config.n_layers - 1)
            ]
        )

    def init_weights(self) -> None:
        torch.nn.init.normal_(self.first_conv.weight, std=self.config.init_std)
        for conv_layer in self.conv_layers:
            torch.nn.init.normal_(conv_layer.weight, std=self.config.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(x)
        for conv_layer in self.conv_layers:
            x = x + conv_layer(x)
        return x
