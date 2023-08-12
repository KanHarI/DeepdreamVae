import dataclasses
from typing import Callable

import torch


@dataclasses.dataclass
class BlockConfig:
    n_layers: int
    n_channels_in: int
    n_channels_out: int
    activation: Callable[[torch.Tensor], torch.Tensor]
    dropout: float
    init_std: float
    device: str
    dtype: torch.dtype
    ln_eps: float
    image_size: int


class Block(torch.nn.Module):
    def __init__(self, config: BlockConfig):
        super().__init__()
        self.config = config
        self.conv_layers = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    in_channels=self.config.n_channels_out,
                    out_channels=self.config.n_channels_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                    device=self.config.device,
                    dtype=self.config.dtype,
                )
                for _ in range(self.config.n_layers)
            ]
        )
        self.layer_norms = torch.nn.ModuleList(
            [
                torch.nn.LayerNorm(
                    [
                        self.config.n_channels_out,
                        self.config.image_size,
                        self.config.image_size,
                    ],
                    eps=self.config.ln_eps,
                    device=self.config.device,
                    dtype=self.config.dtype,
                )
                for _ in range(self.config.n_layers)
            ]
        )
        self.activation = config.activation

    def init_weights(self) -> None:
        for conv_layer in self.conv_layers:
            torch.nn.init.normal_(conv_layer.weight, std=self.config.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tmp = torch.zeros(
            (x.shape[0], self.config.n_channels_out, x.shape[2], x.shape[3]),
            device=self.config.device,
            dtype=self.config.dtype,
        )
        # Add the min of the input and output channels as skip connections
        tmp[:, : self.config.n_channels_in, :, :] = x[
            :, : self.config.n_channels_out, :, :
        ]
        x = tmp
        for j, conv_layer in enumerate(self.conv_layers):
            x = x + self.activation(conv_layer(self.layer_norms[j](x)))
        return x
