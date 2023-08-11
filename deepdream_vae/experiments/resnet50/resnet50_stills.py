from typing import Any

import hydra
import sys

import dacite

from deepdream_vae.conf.resnet50.resnet50_stills import Resnet50StillsExperimentConf
import wandb

from deepdream_vae.models.deepdream_vae import DeepdreamVAEConfig


@hydra.main(
    config_path="../../conf/resnet50",
    config_name="resnet50_stills.yaml",
    version_base=None,
)
def main(hydra_cfg: dict[Any, Any]) -> int:
    config: Resnet50StillsExperimentConf = dacite.from_dict(data_class=Resnet50StillsExperimentConf, data=hydra_cfg)
    if config.wandb_log:
        wandb.init(project="deepdream-vae-resnet50-stills", name=config.wandb_run_name)
        wandb.config.update(config)
    model_conf = DeepdreamVAEConfig(
        n_layers_per_block=config.unet.n_layers_per_block,
        n_blocks=config.unet.n_blocks,
        n_first_block_channels=config.unet.n_first_block_channels,
        init_std=config.unet.optimizer.init_std,
        activation=config.unet.activation,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
