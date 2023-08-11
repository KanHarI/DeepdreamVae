import sys
from typing import Any

import dacite
import hydra
import wandb

from deepdream_vae.conf.resnet50.resnet50_stills import Resnet50StillsExperimentConf
from deepdream_vae.models.deepdream_vae import DeepdreamVAEConfig, DeepdreamVAE


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
    x = 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
