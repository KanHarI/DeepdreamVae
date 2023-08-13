import dataclasses
from typing import Callable

import torch

from deepdream_vae.utils.new_gelu import new_gelu
from deepdream_vae.utils.softplus import softplus


@dataclasses.dataclass
class DiscriminatorConf:
    n_blocks: int
    n_layers_per_block: int
    n_layers_mini_block: int
    n_first_block_channels: int
    _activation: str
    device: str
    _dtype: str
    ln_eps: float
    loss_eps: float
    discriminator_cheat_loss: float
    cheat_loss_eps: float
    bilinear_form_dimension: int

    @property
    def activation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        match self._activation:
            case "relu":
                return torch.nn.functional.relu
            case "gelu":
                return torch.nn.functional.gelu
            case "new_gelu":
                return new_gelu
            case "softplus":
                return softplus
            case _:
                raise ValueError(f"Unknown activation: {self._activation}")

    @property
    def dtype(self) -> torch.dtype:
        match self._dtype:
            case "bfloat16":
                return torch.bfloat16
            case "float16":
                return torch.float16
            case "float32":
                return torch.float32
            case "float64":
                return torch.float64
            case _:
                raise ValueError(f"Unknown dtype: {self._dtype}")
