import sys
from typing import Any

import dacite
import hydra
import numpy as np
import torch
import torch.utils.data
import wandb
from PIL import Image
from tqdm import tqdm

from deepdream_vae.conf.resnet50.resnet50_stills import Resnet50StillsExperimentConf
from deepdream_vae.datasets.resnet50_deepdream import (
    Resnet50DeepdreamDataset,
    Resnet50DeepdreamDatasetConfig,
)
from deepdream_vae.models.deepdream_vae import DeepdreamVAE, DeepdreamVAEConfig
from deepdream_vae.models.discriminator import Discriminator, DiscriminatorConfig


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
        image_size=config.image_size,
    )
    test_dataset_conf = Resnet50DeepdreamDatasetConfig(
        processed_path=config.processed_files_path,
        origin_path=config.source_files_path,
        is_train=False,
        image_size=config.image_size,
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
    print("Creating model...")
    model_conf = DeepdreamVAEConfig(
        n_layers_per_block=config.unet.n_layers_per_block,
        n_blocks=config.unet.n_blocks,
        n_first_block_channels=config.unet.n_first_block_channels,
        init_std=config.optimizer.init_std,
        activation=config.unet.activation,
        device=config.unet.device,
        dtype=config.unet.dtype,
        ln_eps=config.unet.ln_eps,
        image_size=config.image_size,
    )
    generative_model = DeepdreamVAE(model_conf)
    generative_model.init_weights()
    generative_model.train()
    discriminator_config = DiscriminatorConfig(
        n_blocks=config.discriminator.n_blocks,
        n_layers_per_block=config.discriminator.n_layers_per_block,
        n_first_block_channels=config.discriminator.n_first_block_channels,
        init_std=config.optimizer.init_std,
        activation=config.discriminator.activation,
        device=config.discriminator.device,
        dtype=config.discriminator.dtype,
        ln_eps=config.discriminator.ln_eps,
        image_size=config.image_size,
        loss_eps=config.discriminator.loss_eps,
    )
    discriminator = Discriminator(discriminator_config)
    if config.compile_model:
        generative_model = torch.compile(generative_model)  # type: ignore
        discriminator = torch.compile(discriminator)  # type: ignore
    print("Creating optimizer...")
    optimizer = config.optimizer.create_optimizer(generative_model.parameters())
    print("Starting training...")
    train_losses = torch.zeros(
        (config.eval_interval), device="cpu", dtype=torch.float32
    )
    train_losses += float("inf")
    for step in range(config.optimizer.max_iters):
        eval_losses = torch.zeros(
            (config.eval_iters), device="cpu", dtype=torch.float32
        )
        eval_transformed_discriminator_losses = torch.zeros(
            (config.eval_iters), device="cpu", dtype=torch.float32
        )
        eval_deepdream_discriminator_losses = torch.zeros(
            (config.eval_iters), device="cpu", dtype=torch.float32
        )
        eval_mixed_losses = torch.zeros(
            (config.eval_iters), device="cpu", dtype=torch.float32
        )
        if step % config.eval_interval == 0:
            generative_model.eval()
            with torch.no_grad():
                for i in tqdm(range(config.eval_iters)):
                    batch = next(iter(test_dataloader))
                    batch = tuple(
                        [
                            t.to(device=config.unet.device, dtype=config.unet.dtype)
                            for t in batch
                        ]
                    )
                    transformed_images = generative_model(*batch)
                    transformed_discriminator_loss = discriminator(
                        transformed_images[0],
                        torch.zeros(
                            (transformed_images[0].shape[0]),
                            device=config.discriminator.device,
                            dtype=config.discriminator.dtype,
                        ),
                    )
                    deepdream_discriminator_loss = discriminator(
                        batch[1],
                        torch.ones(
                            (transformed_images[0].shape[0]),
                            device=config.discriminator.device,
                            dtype=config.discriminator.dtype,
                        ),
                    )
                    random_mixes_ratios = torch.rand(
                        (batch[0].shape[0],),
                        device=config.discriminator.device,
                        dtype=config.discriminator.dtype,
                    )
                    random_mixes = (
                        random_mixes_ratios * batch[1]
                        + (1 - random_mixes_ratios) * transformed_images[0]
                    )
                    mixed_discriminator_loss = discriminator(
                        random_mixes,
                        random_mixes_ratios,
                    )
                    loss = (
                        transformed_discriminator_loss
                        + deepdream_discriminator_loss
                        + mixed_discriminator_loss
                    )
                    eval_losses[i] = loss.item()
                    eval_transformed_discriminator_losses[
                        i
                    ] = transformed_discriminator_loss.item()
                    eval_deepdream_discriminator_losses[
                        i
                    ] = deepdream_discriminator_loss.item()
                    eval_mixed_losses[i] = mixed_discriminator_loss.item()
                sample_output_image = transformed_images[0]
                # Save sample output image to `outputs` directory
                sample_output_image = sample_output_image[0].detach().cpu()
                sample_output_image = sample_output_image.permute(1, 2, 0)
                sample_output_image = sample_output_image.numpy()
                sample_output_image = (sample_output_image * 255).astype(np.uint8)  # type: ignore
                save_path = f"outputs/sample_output_{step}.jpg"
                print(f"Saving sample output image to {save_path}")
                image_to_save = Image.fromarray(sample_output_image)
                image_to_save.save(save_path)
                # Log eval metrics to wandb
                if config.wandb_log:
                    wandb.log(
                        {
                            "eval_loss": eval_losses.mean().item(),
                            "eval_loss_std": eval_losses.std().item(),
                            "eval_transformed_discriminator_loss": eval_transformed_discriminator_losses.mean().item(),
                            "eval_deepdream_discriminator_loss": eval_deepdream_discriminator_losses.mean().item(),
                            "eval_mixed_discriminator_loss": eval_mixed_losses.mean().item(),
                            "train_loss": train_losses.mean().item(),
                            "train_loss_std": train_losses.std().item(),
                            "lr": config.optimizer.get_lr(step),
                        },
                        step=step,
                    )
                # Print eval metrics
                print(
                    f"Step {step} eval loss: {eval_losses.mean().item()} +/- {eval_losses.std().item()}\n"
                    f"train loss: {train_losses.mean().item()} +/- {train_losses.std().item()}, lr: {config.optimizer.get_lr(step)}\n"
                )
            generative_model.train()
        batch = next(iter(train_dataloader))
        batch = tuple(
            [t.to(device=config.unet.device, dtype=config.unet.dtype) for t in batch]
        )
        transformed_images = generative_model(*batch)
        transformed_discriminator_loss = discriminator(
            transformed_images[0],
            torch.zeros(
                (transformed_images[0].shape[0]),
                device=config.discriminator.device,
                dtype=config.discriminator.dtype,
            ),
        )
        deepdream_discriminator_loss = discriminator(
            batch[1],
            torch.ones(
                (transformed_images[0].shape[0]),
                device=config.discriminator.device,
                dtype=config.discriminator.dtype,
            ),
        )
        random_mixes_ratios = torch.rand(
            (batch[0].shape[0],),
            device=config.discriminator.device,
            dtype=config.discriminator.dtype,
        )
        random_mixes = (
            random_mixes_ratios * batch[1]
            + (1 - random_mixes_ratios) * transformed_images[0]
        )
        mixed_discriminator_loss = discriminator(
            random_mixes,
            random_mixes_ratios,
        )
        loss = (
            transformed_discriminator_loss
            + deepdream_discriminator_loss
            + mixed_discriminator_loss
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = config.optimizer.get_lr(step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses[step % config.eval_interval] = loss.item()
        if step % config.log_interval == 0:
            print(f"Step {step} train loss: {loss.item()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
