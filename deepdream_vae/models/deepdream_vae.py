import dataclasses
import math
from typing import Callable

import torch

from deepdream_vae.models.block import Block, BlockConfig


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
    image_size: int
    n_layers_mini_block: int
    mixing_factor_scale: float


class DeepdreamVAE(torch.nn.Module):
    def __init__(self, config: DeepdreamVAEConfig):
        super().__init__()
        self.config = config
        self.encoder_blocks_config: list[BlockConfig] = []
        block_channels = self.config.n_first_block_channels
        image_size = self.config.image_size
        for i in range(self.config.n_blocks):
            self.encoder_blocks_config.append(
                BlockConfig(
                    n_layers=self.config.n_layers_per_block,
                    n_layers_mini_block=self.config.n_layers_mini_block,
                    n_channels_in=block_channels * 2,
                    n_channels_out=block_channels * 2,
                    activation=self.config.activation,
                    dropout=0.0,
                    init_std=self.config.init_std,
                    device=self.config.device,
                    dtype=self.config.dtype,
                    ln_eps=self.config.ln_eps,
                    image_size=image_size,
                )
            )
            block_channels *= 2
            image_size //= 2
        image_size *= 2
        self.decoders_blocks_config: list[BlockConfig] = []
        for i in range(self.config.n_blocks):
            self.decoders_blocks_config.append(
                BlockConfig(
                    n_layers=self.config.n_layers_per_block,
                    n_layers_mini_block=self.config.n_layers_mini_block,
                    n_channels_in=block_channels,
                    n_channels_out=block_channels // 2,
                    activation=self.config.activation,
                    dropout=0.0,
                    init_std=self.config.init_std,
                    device=self.config.device,
                    dtype=self.config.dtype,
                    ln_eps=self.config.ln_eps,
                    image_size=image_size,
                )
            )
            block_channels //= 2
            image_size *= 2
        self.encoder_blocks = torch.nn.ModuleList(
            [Block(config) for config in self.encoder_blocks_config]
        )
        self.decoders_blocks = torch.nn.ModuleList(
            [Block(config) for config in self.decoders_blocks_config]
        )
        self.channels_expander = torch.nn.Parameter(
            torch.zeros(
                (3, self.config.n_first_block_channels),
                device=self.config.device,
                dtype=self.config.dtype,
                requires_grad=True,
            )
        )
        self.final_block_config = BlockConfig(
            n_layers=self.config.n_layers_per_block,
            n_layers_mini_block=self.config.n_layers_mini_block,
            n_channels_in=self.config.n_first_block_channels,
            n_channels_out=self.config.n_first_block_channels,
            activation=self.config.activation,
            dropout=0.0,
            init_std=self.config.init_std,
            device=self.config.device,
            dtype=self.config.dtype,
            ln_eps=self.config.ln_eps,
            image_size=self.config.image_size,
        )
        self.mixing_factors = torch.nn.Parameter(
            torch.zeros(
                (self.config.n_blocks + 1),
                device=self.config.device,
                dtype=self.config.dtype,
                requires_grad=True,
            )
        )
        self.final_block = Block(self.final_block_config)

    def init_weights(self) -> None:
        torch.nn.init.normal_(
            self.channels_expander,
            mean=0.0,
            std=1.0 / math.sqrt(self.config.n_first_block_channels),
        )
        for encoder_block in self.encoder_blocks:
            encoder_block.init_weights()
        for decoder_block in self.decoders_blocks:
            decoder_block.init_weights()
        self.final_block.init_weights()

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("lc,...lhw->...chw", self.channels_expander, x)
        skipped = [x]
        for i, encoder_block in enumerate(self.encoder_blocks):
            x = torch.cat([x, torch.randn_like(x)], dim=1)
            x = encoder_block(x)
            skipped.append(x)
            x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        for j, decoder_block in enumerate(self.decoders_blocks):
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            mixing_factor = torch.sigmoid(
                self.mixing_factors[j] * self.config.mixing_factor_scale
            )
            x = decoder_block(mixing_factor * x + (1 - mixing_factor) * skipped.pop())
        mixing_factor = torch.sigmoid(
            self.mixing_factors[-1] * self.config.mixing_factor_scale
        )
        x = self.final_block(mixing_factor * x + (1 - mixing_factor) * skipped.pop())
        # Using the same layer for color-> channels and channels -> color
        # results in much faster initial convergence.
        x = torch.einsum("lc,...chw->...lhw", self.channels_expander, x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)
