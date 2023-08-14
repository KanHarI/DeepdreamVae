import dataclasses
import hashlib
import os
import typing

import torch.utils.data
import torchvision.transforms as transforms  # type: ignore
from PIL import Image


@dataclasses.dataclass
class FolderImageLoaderDatasetConfig:
    origin_path: str
    is_train: bool
    image_size: int
    scale_factor: int


class FolderImageLoaderDataset(torch.utils.data.Dataset[torch.Tensor]):
    def __init__(self, config: FolderImageLoaderDatasetConfig):
        self.config = config
        self.transform = transforms.Compose(
            [
                transforms.Lambda(self.scale_images),
                self.tile_small_images,
                transforms.RandomCrop(self.config.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        self.image_filenames = []
        for root, dirs, files in os.walk(self.config.origin_path):
            self.image_filenames.extend(
                [os.path.join(root, file) for file in files if file.endswith(".jpg")]
            )
        # If is_train, keep only files whose sha256 starts with 0-D.
        # Otherwise, keep only files whose sha256 starts with E-F.
        self.image_filenames = list(
            filter(
                lambda x: (self._is_train_filename(x) == config.is_train),
                self.image_filenames,
            )
        )

    def tile_small_images(self, image: Image.Image) -> Image.Image:
        if (
            image.size[0] < self.config.image_size
            or image.size[1] < self.config.image_size
        ):
            # Tile
            new_image = Image.new(
                image.mode,
                (
                    max(image.size[0], self.config.image_size),
                    max(image.size[1], self.config.image_size),
                ),
            )
            for i in range(0, new_image.size[0], image.size[0]):
                for j in range(0, new_image.size[1], image.size[1]):
                    new_image.paste(image, (i, j))
            return new_image
        return image

    def _is_train_filename(self, filename: str) -> bool:
        first_hex = hashlib.sha256(os.path.basename(filename).encode()).hexdigest()[0]
        return first_hex in {
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "a",
            "b",
            "c",
            "d",
        }

    def scale_images(self, img: Image.Image) -> Image.Image:
        return img.resize(
            (
                img.size[0] // self.config.scale_factor,
                img.size[1] // self.config.scale_factor,
            )
        )

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_filenames[index])
        image = self.transform(image) * 2.0 - 1.0
        return typing.cast(torch.Tensor, image)
