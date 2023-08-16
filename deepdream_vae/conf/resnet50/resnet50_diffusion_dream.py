import dataclasses

from deepdream_vae.conf.building_blocks.optimizer import OptimizerConf
from deepdream_vae.conf.building_blocks.unet import UNetConf


@dataclasses.dataclass
class Resnet50DiffusionDreamExperimentConf:
    unet: UNetConf
    optimizer: OptimizerConf
    source_files_path: str
    eval_interval: int
    eval_iters: int
    log_interval: int
    wandb_log: bool
    wandb_run_name: str
    image_size: int
    scale_factor: int
    save_interval: int
    n_blocks_resnet: int
    l2_consistency_loss_weight: float
    num_diffusion_activations: int
    overflow_loss_weight: float
    l2_diffusion_diff_loss_weight: float
