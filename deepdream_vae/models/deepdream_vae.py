import dataclasses
from typing import Callable

import torch

from deepdream_vae.models.deepdream_vae_block import (
    DeepdreamVAEBlock,
    DeepdreamVAEBlockConfig,
)


@dataclasses.dataclass
class DeepdreamVAEConfig:
    n_layers_per_block: int
    n_blocks: int
    n_first_block_channels: int
    init_std: float
    activation: Callable[[torch.Tensor], torch.Tensor]
    device: str
    dtype: torch.dtype
    ln_eps: float


class DeepdreamVAE(torch.nn.Module):
    def __init__(self, config: DeepdreamVAEConfig):
        super().__init__()
        self.config = config
        self.encoder_blocks_config: list[DeepdreamVAEBlockConfig] = []
        block_channels = self.config.n_first_block_channels
        for i in range(self.config.n_blocks):
            self.encoder_blocks_config.append(
                DeepdreamVAEBlockConfig(
                    n_layers=self.config.n_layers_per_block,
                    n_channels_in=block_channels,
                    n_channels_out=block_channels * 2,
                    activation=self.config.activation,
                    dropout=0.0,
                    init_std=0.02,
                    device=self.config.device,
                    dtype=self.config.dtype,
                    ln_eps=self.config.ln_eps,
                )
            )
            block_channels *= 2
        self.decoders_blocks_config: list[DeepdreamVAEBlockConfig] = []
        for i in range(self.config.n_blocks):
            self.decoders_blocks_config.append(
                DeepdreamVAEBlockConfig(
                    n_layers=self.config.n_layers_per_block,
                    n_channels_in=block_channels,
                    n_channels_out=block_channels // 2,
                    activation=self.config.activation,
                    dropout=0.0,
                    init_std=0.02,
                    device=self.config.device,
                    dtype=self.config.dtype,
                    ln_eps=self.config.ln_eps,
                )
            )
            block_channels //= 2
        self.encoder_blocks = torch.nn.ModuleList(
            [DeepdreamVAEBlock(config) for config in self.encoder_blocks_config]
        )
        self.decoders_blocks = torch.nn.ModuleList(
            [DeepdreamVAEBlock(config) for config in self.decoders_blocks_config]
        )

    def init_weights(self) -> None:
        for encoder_block in self.encoder_blocks:
            encoder_block.init_weights()
        for decoder_block in self.decoders_blocks:
            decoder_block.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skipped = [x]
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skipped.append(x)
            x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        for decoder_block in self.decoders_blocks:
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            x += skipped.pop()
            x = decoder_block(x)
        return x
