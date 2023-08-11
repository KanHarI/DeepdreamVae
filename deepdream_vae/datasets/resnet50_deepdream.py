import dataclasses
import os

import torch
import torch.utils.data


@dataclasses.dataclass
class Resnet50DeepdreamDatasetConfig:
    processed_path: str
    origin_path: str


class Resnet50DeepdreamDataset(
    torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]
):
    def __init__(self, config: Resnet50DeepdreamDatasetConfig):
        super().__init__()
        self.config = config
        self.folders = [
            os.path.join(config.processed_path, folder)
            for folder in os.listdir(config.processed_path)
        ]
        self.processed_files = []
        for folder in self.folders:
            self.processed_files += [
                os.path.join(folder, file) for file in os.listdir(folder)
            ]

    def __len__(self) -> int:
        return len(self.processed_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        processed_file = self.processed_files[index]
        # Processed tile name sample: `0_Parade_marchingband_1_56.jpg___width_600_model_RESNET50_IMAGENET_layer1_layer2_pyrsize_4_pyrratio_1.8_iter_10_lr_0.09_shift_32_smooth_0.5.jpg`
        # Origin file name: `0_Parade_marchingband_1_56.jpg`
        origin_file_name = os.path.basename(processed_file).split("___")[0]
        # Origin dir: `0--Parade`
        origin_file_name_dir_num = origin_file_name.split("_")[0]
        origin_file_name_dir_name = origin_file_name.split("_")[1]
        origin_file_name_dir = os.path.join(
            self.config.origin_path,
            f"{origin_file_name_dir_num}--{origin_file_name_dir_name}",
        )
        origin_file = os.path.join(origin_file_name_dir, origin_file_name)
        processed_image = torch.load(processed_file)
        origin_image = torch.load(origin_file)
        return processed_image, origin_image
