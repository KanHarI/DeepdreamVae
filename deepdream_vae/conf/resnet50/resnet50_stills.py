import dataclasses

from deepdream_vae.conf.building_blocks.optimizer import OptimizerConf
from deepdream_vae.conf.building_blocks.unet import UNetConf


@dataclasses.dataclass
class Resnet50StillsExperimentConf:
    unet: UNetConf
    optimizer: OptimizerConf
    source_files_path: str
    processed_files_path: str
    eval_interval: int
    eval_iters: int
    log_interval: int
    wandb_log: bool
    wandb_run_name: str
