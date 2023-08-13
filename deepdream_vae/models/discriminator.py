import dataclasses
import typing
from typing import Callable

import torch

from deepdream_vae.models.block import Block, BlockConfig
from deepdream_vae.utils.inv_sigmoid import inv_sigmoid


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
    discriminator_cheat_loss: float
    cheat_loss_eps: float
    n_stacked_images_in: int


class Discriminator(torch.nn.Module):
    def __init__(self, config: DiscriminatorConfig):
        super().__init__()
        self.config = config
        self.color_to_channels = torch.zeros(
            (config.n_stacked_images_in * 3, config.n_first_block_channels),
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

    def estimate_is_image_deepdream(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.einsum("lc,...lhw->...chw", self.color_to_channels, x)
        for block in self.encoder_blocks:
            x = block(x)
            x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.einsum("chw,bchw->b", self.estimator, x)
        logits = torch.sigmoid(x)
        return logits, x

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, x = self.estimate_is_image_deepdream(x)
        loss_if_positive = -torch.log(logits + self.config.loss_eps)
        loss_if_negative = -torch.log(1 - logits + self.config.loss_eps)
        cheat_loss = (
            inv_sigmoid(
                self.config.cheat_loss_eps
                + targets * (1 - 2 * self.config.cheat_loss_eps)
            )
            - x
        ) ** 2 * self.config.discriminator_cheat_loss
        loss = (
            loss_if_positive * targets + loss_if_negative * (1 - targets) + cheat_loss
        ).mean()
        # loss = (
        #     -targets * torch.log(logits + self.config.loss_eps)
        #     - (1 - targets) * torch.log(1 - logits + self.config.loss_eps)
        #     - (targets * 2 - 1)
        #     * x
        #     * self.config.discriminator_cheat_loss  # Rescue us when we're stuck
        # )
        # loss = ((logits - targets) ** 2).mean()
        return typing.cast(torch.Tensor, loss.mean())
