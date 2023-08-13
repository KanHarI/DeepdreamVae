import math

import torch

LN_2 = math.log(2.0)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x > 0,
        x + torch.log(1 + torch.exp(-x)) - LN_2,
        torch.log(1 + torch.exp(x)) - LN_2,
    )
