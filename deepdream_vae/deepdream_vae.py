import dataclasses

import torch


@dataclasses.dataclass
class DeepdreamVAEConfig:
    n_blocks: int
    n_layers_per_block: int


class DeepdreamVAE(torch.nn.Module):
    def __init__(self, config: DeepdreamVAEConfig):
        super().__init__()
        self.config = config
