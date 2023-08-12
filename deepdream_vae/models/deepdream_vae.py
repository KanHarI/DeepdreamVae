import dataclasses
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
                    n_channels_in=block_channels,
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
        self.noise_volume = torch.nn.Parameter(
            torch.zeros(
                (block_channels,),
                device=self.config.device,
                dtype=self.config.dtype,
                requires_grad=True,
            )
        )
        self.decoders_blocks_config: list[BlockConfig] = []
        for i in range(self.config.n_blocks):
            self.decoders_blocks_config.append(
                BlockConfig(
                    n_layers=self.config.n_layers_per_block,
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
        self.channels_expander = torch.zeros(
            (3, self.config.n_first_block_channels),
            device=self.config.device,
            dtype=self.config.dtype,
            requires_grad=True,
        )
        self.channels_contractor = torch.zeros(
            (3, self.config.n_first_block_channels),
            device=self.config.device,
            dtype=self.config.dtype,
            requires_grad=True,
        )

    def init_weights(self) -> None:
        for encoder_block in self.encoder_blocks:
            encoder_block.init_weights()
        for decoder_block in self.decoders_blocks:
            decoder_block.init_weights()
        torch.nn.init.normal_(
            self.channels_expander, mean=0.0, std=self.config.init_std
        )
        torch.nn.init.normal_(
            self.channels_contractor, mean=0.0, std=self.config.init_std
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("lc,...lhw->...chw", self.channels_expander, x)
        skipped = [x]
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skipped.append(x)
            x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x + torch.einsum(
            "bchw,c->bchw", torch.randn_like(x), torch.exp(self.noise_volume)
        )
        for decoder_block in self.decoders_blocks:
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            x += skipped.pop()
            x = decoder_block(x)
        x = torch.einsum("cl,...lhw->...chw", self.channels_contractor, x)
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.transform(x)
