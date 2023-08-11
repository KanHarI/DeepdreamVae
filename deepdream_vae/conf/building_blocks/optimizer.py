import dataclasses
import typing
from typing import Any

import torch.optim.optimizer


@dataclasses.dataclass
class OptimizerConf:
    _optimizer: str
    lr: float
    weight_decay: float
    lr_schedule: str
    beta1: float
    beta2: float
    warmup_iters: int
    max_iters: int

    def create_optimizer(self, params: Any) -> torch.optim.optimizer.Optimizer:
        match self._optimizer:
            case "adamw":
                return torch.optim.AdamW(
                    params=params,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    betas=(self.beta1, self.beta2),
                )
            case "sgd":
                return torch.optim.SGD(
                    params=params,
                    lr=self.lr,
                    momentum=self.beta1,
                    weight_decay=self.weight_decay,
                )
            case _:
                raise ValueError(f"Unknown optimizer: {self._optimizer}")

    def get_lr(self, step: int) -> float:
        match self.lr_schedule:
            case "constant":
                return self.lr
            case "linear":
                if step < self.warmup_iters:
                    return self.lr * step / self.warmup_iters
                else:
                    return self.lr * (
                        1.0
                        - (step - self.warmup_iters)
                        / (self.max_iters - self.warmup_iters)
                    )
            case "invsqrt":
                if step < self.warmup_iters:
                    return self.lr * step / self.warmup_iters
                else:
                    return typing.cast(
                        float,
                        self.lr
                        * (
                            1.0
                            - (step - self.warmup_iters)
                            / (self.max_iters - self.warmup_iters)
                        )
                        ** 0.5,
                    )
            case _:
                raise ValueError(f"Unknown lr schedule: {self.lr_schedule}")
