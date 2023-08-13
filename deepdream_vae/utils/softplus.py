import math

import torch

LN_2 = math.log(2.0)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(1.0 + torch.exp(x)) - LN_2
