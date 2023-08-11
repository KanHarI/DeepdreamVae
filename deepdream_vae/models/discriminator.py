import dataclasses
from typing import Callable

import torch

from deepdream_vae.models.block import Block, BlockConfig


@dataclasses.dataclass
class DiscriminatorConfig:
    n_layers_per_block: int
    n_blocks: int
    n_first_block_channels: int
    init_std: float
    activation: Callable[[torch.Tensor], torch.Tensor]
    device: str
    dtype: torch.dtype
    ln_eps: float
    image_size: int
    loss_eps: float


class Discriminator(torch.nn.Module):
    def __init__(self, config: DiscriminatorConfig):
        super().__init__()
        self.config = config
        self.color_to_channels = torch.zeros(
            (3, config.n_first_block_channels),
            device=config.device,
            dtype=config.dtype,
            requires_grad=True,
        )
        num_channels = config.n_first_block_channels
        image_size = config.image_size
        self.encoder_blocks_config: list[BlockConfig] = []
        for i in range(config.n_blocks):
            self.encoder_blocks_config.append(
                BlockConfig(
                    n_layers=config.n_layers_per_block,
                    n_channels_in=num_channels,
                    n_channels_out=num_channels * 2,
                    activation=config.activation,
                    dropout=0.0,
                    init_std=config.init_std,
                    device=config.device,
                    dtype=config.dtype,
                    ln_eps=config.ln_eps,
                    image_size=image_size,
                )
            )
            num_channels *= 2
            image_size //= 2
        self.encoder_blocks = torch.nn.ModuleList(
            [Block(config) for config in self.encoder_blocks_config]
        )
        self.estimator = torch.zeros(
            (num_channels, image_size, image_size),
            device=config.device,
            dtype=config.dtype,
            requires_grad=True,
        )

    def init_weights(self) -> None:
        torch.nn.init.normal_(self.color_to_channels, std=self.config.init_std)
        for block in self.encoder_blocks:
            block.init_weights()
        torch.nn.init.normal_(self.estimator, std=self.config.init_std)

    def estimate_image(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("lc,...lhw->...chw", self.color_to_channels, x)
        for block in self.encoder_blocks:
            x = block(x)
        x = torch.einsum("chw,bchw->b", self.estimator, x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = self.estimate_image(x)
        loss = -torch.log(torch.abs(x - targets) + self.config.loss_eps)
        return loss.mean()
