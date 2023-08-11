import traceback
from typing import Iterator, TypeVar

import torch
import torch.utils.data

T = TypeVar("T")


def loop_dataloader(
    dataloader: torch.utils.data.DataLoader[torch.utils.data.Dataset[T]],
) -> Iterator[T]:
    while True:
        for batch in dataloader:
            try:
                yield batch
            except Exception:
                print(f"Error while processing {traceback.format_exc()}")
                pass
