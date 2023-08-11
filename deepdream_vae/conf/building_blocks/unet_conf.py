import dataclasses
from typing import Callable

import torch

from deepdream_vae.utils.new_gelu import new_gelu


@dataclasses.dataclass
class UNetConf:
    n_blocks: int
    n_layers_per_block: int
    n_first_block_channels: int
    _activation: str

    @property
    def activation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        match self._activation:
            case "relu":
                return torch.nn.functional.relu
            case "gelu":
                return torch.nn.functional.gelu
            case "new_gelu":
                return new_gelu
            case _:
                raise ValueError(f"Unknown activation: {self._activation}")
