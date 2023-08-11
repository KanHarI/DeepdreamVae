import sys
from typing import Any

import dacite
import hydra
import torch
import torch.utils.data
import wandb

from deepdream_vae.conf.resnet50.resnet50_stills import Resnet50StillsExperimentConf
from deepdream_vae.datasets.resnet50_deepdream import (
    Resnet50DeepdreamDataset,
    Resnet50DeepdreamDatasetConfig,
)
from deepdream_vae.models.deepdream_vae import DeepdreamVAE, DeepdreamVAEConfig


@hydra.main(
    config_path="../../conf/resnet50",
    config_name="resnet50_stills.yaml",
    version_base=None,
)
def main(hydra_cfg: dict[Any, Any]) -> int:
    config: Resnet50StillsExperimentConf = dacite.from_dict(
        data_class=Resnet50StillsExperimentConf, data=hydra_cfg
    )
    if config.wandb_log:
        wandb.init(project="deepdream-vae-resnet50-stills", name=config.wandb_run_name)
        wandb.config.update(config)
    print("Creating dataset...")
    # Create dataset
    train_dataset_conf = Resnet50DeepdreamDatasetConfig(
        processed_path=config.processed_files_path,
        origin_path=config.source_files_path,
        is_train=True,
    )
    test_dataset_conf = Resnet50DeepdreamDatasetConfig(
        processed_path=config.processed_files_path,
        origin_path=config.source_files_path,
        is_train=False,
    )
    train_dataset = Resnet50DeepdreamDataset(train_dataset_conf)
    test_dataset = Resnet50DeepdreamDataset(test_dataset_conf)
    print(f"Num train samples: {len(train_dataset)}")
    print(f"Num test samples: {len(test_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.optimizer.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.optimizer.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    batch = next(iter(train_dataloader))
    model_conf = DeepdreamVAEConfig(
        n_layers_per_block=config.unet.n_layers_per_block,
        n_blocks=config.unet.n_blocks,
        n_first_block_channels=config.unet.n_first_block_channels,
        init_std=config.optimizer.init_std,
        activation=config.unet.activation,
        device=config.unet.device,
        dtype=config.unet.dtype,
        ln_eps=config.unet.ln_eps,
    )
    model = DeepdreamVAE(model_conf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
