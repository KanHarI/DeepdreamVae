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
    device: str
    dtype: torch.dtype
    ln_eps: float


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
            device=self.config.device,
            dtype=self.config.dtype,
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
                    device=self.config.device,
                    dtype=self.config.dtype,
                )
                for _ in range(self.config.n_layers - 1)
            ]
        )
        self.first_layer_norm = torch.nn.LayerNorm(
            self.config.n_channels_in,
            eps=self.config.ln_eps,
            device=self.config.device,
            dtype=self.config.dtype,
        )
        self.layer_norms = torch.nn.ModuleList(
            [
                torch.nn.LayerNorm(
                    self.config.n_channels_out,
                    eps=self.config.ln_eps,
                    device=self.config.device,
                    dtype=self.config.dtype,
                )
                for _ in range(self.config.n_layers - 1)
            ]
        )

    def init_weights(self) -> None:
        torch.nn.init.normal_(self.first_conv.weight, std=self.config.init_std)
        for conv_layer in self.conv_layers:
            torch.nn.init.normal_(conv_layer.weight, std=self.config.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_conv(self.layer_norms[0](x))
        for j, conv_layer in enumerate(self.conv_layers):
            x = x + conv_layer(self.layer_norms[j](x))
        return x
