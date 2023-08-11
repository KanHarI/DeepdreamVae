import dataclasses
import hashlib
import os

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms  # type: ignore


@dataclasses.dataclass
class Resnet50DeepdreamDatasetConfig:
    processed_path: str
    origin_path: str
    is_train: bool
    image_size: int


# 0--Parade            14--Traffic         19--Couple        23--Shoppers          28--Sports_Fan           32--Worker_Laborer  37--Soccer       41--Swimming    46--Jockey                   50--Celebration_Or_Party  55--Sports_Coach_Trainer  5--Car_Accident      9--Press_Conference
# 10--People_Marching  15--Stock_Market    1--Handshaking    24--Soldier_Firing    29--Students_Schoolkids  33--Running         38--Tennis       42--Car_Racing  47--Matador_Bullfighter      51--Dresses               56--Voter                 61--Street_Battle
# 11--Meeting          16--Award_Ceremony  20--Family_Group  25--Soldier_Patrol    2--Demonstration         34--Baseball        39--Ice_Skating  43--Row_Boat    48--Parachutist_Paratrooper  52--Photographers         57--Angler                6--Funeral
# 12--Group            17--Ceremony        21--Festival      26--Soldier_Drilling  30--Surgeons             35--Basketball      3--Riot          44--Aerobics    49--Greeting                 53--Raid                  58--Hockey                7--Cheering
# 13--Interview        18--Concerts        22--Picnic        27--Spa               31--Waiter_Waitress      36--Football        40--Gymnastics   45--Balloonist  4--Dancing                   54--Rescue                59--people--driving--car  8--Election_Campain

TWO_WORD_NAME_NUMBERS = [
    28,
    32,
    5,
    9,
    10,
    15,
    24,
    29,
    42,
    47,
    61,
    16,
    20,
    25,
    39,
    43,
    48,
    26,
    31,
    8,
]

THREE_WORD_NAME_NUMBERS = [
    50,
    55,
]

SKIP_PHRASES = [
    "59_",
    "peopledrivingcar",
]


class Resnet50DeepdreamDataset(
    torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]
):
    def __init__(self, config: Resnet50DeepdreamDatasetConfig):
        super().__init__()
        self.config = config
        self.processed_files = []
        for root, dirs, files in os.walk(config.processed_path):
            self.processed_files.extend(
                [os.path.join(root, file) for file in files if file.endswith(".jpg")]
            )
        # If is_train, keep only files whose sha256 starts with 0-D.
        # Otherwise, keep only files whose sha256 starts with E-F.
        if config.is_train:
            self.processed_files = list(
                filter(
                    lambda x: self._is_train_filename(x) and not self._is_skip_file(x),
                    self.processed_files,
                )
            )
        else:
            self.processed_files = list(
                filter(
                    lambda x: not self._is_train_filename(x)
                    and not self._is_skip_file(x),
                    self.processed_files,
                )
            )
        self.transform = transforms.Compose(
            [
                self.tile_small_images,
                transforms.RandomCrop(self.config.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

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

    def _is_skip_file(self, filename: str) -> bool:
        for skip_phrase in SKIP_PHRASES:
            if skip_phrase in filename:
                return True
        return False

    def tile_small_images(self, image: Image.Image) -> Image.Image:
        if (
            image.size[0] < self.config.image_size
            or image.size[1] < self.config.image_size
        ):
            factor_x = max(1, self.config.image_size // image.size[0] + 1)
            factor_y = max(1, self.config.image_size // image.size[1] + 1)
            image = image.crop(
                (0, 0, image.size[0] * factor_x, image.size[1] * factor_y)
            )
        return image

    def __len__(self) -> int:
        return len(self.processed_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        processed_file = self.processed_files[index]
        # Processed tile name sample: `0_Parade_marchingband_1_56.jpg___width_600_model_RESNET50_IMAGENET_layer1_layer2_pyrsize_4_pyrratio_1.8_iter_10_lr_0.09_shift_32_smooth_0.5.jpg`
        # Origin file name: `0_Parade_marchingband_1_56.jpg`
        origin_file_name = os.path.basename(processed_file).split("___")[0]
        # Origin dir: `0--Parade`
        origin_file_name_dir_num = origin_file_name.split("_")[0]
        num_words_to_take = 1
        if int(origin_file_name_dir_num) in TWO_WORD_NAME_NUMBERS:
            num_words_to_take = 2
        elif int(origin_file_name_dir_num) in THREE_WORD_NAME_NUMBERS:
            num_words_to_take = 3
        origin_file_name_dir_name = "_".join(
            origin_file_name.split("_")[1 : num_words_to_take + 1]
        )
        origin_file_name_dir = os.path.join(
            self.config.origin_path,
            f"{origin_file_name_dir_num}--{origin_file_name_dir_name}",
        )
        origin_file = os.path.join(origin_file_name_dir, origin_file_name)
        # Load jpgs
        origin_img = Image.open(origin_file)
        processed_img = Image.open(processed_file)
        # Convert to tensors
        origin_img_tensor = self.transform(origin_img)
        processed_img_tensor = self.transform(processed_img)
        return origin_img_tensor, processed_img_tensor
