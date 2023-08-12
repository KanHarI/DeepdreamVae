import os
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
        scale_factor=config.scale_factor,
    )
    test_dataset_conf = Resnet50DeepdreamDatasetConfig(
        processed_path=config.processed_files_path,
        origin_path=config.source_files_path,
        is_train=False,
        image_size=config.image_size,
        scale_factor=config.scale_factor,
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
        noise_proj_init_std_factor=config.unet.noise_proj_init_std_factor,
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
        discriminator_cheat_loss=config.discriminator.discriminator_cheat_loss,
        discriminator_cheat_factor=config.discriminator.discriminator_cheat_factor,
    )
    discriminator = Discriminator(discriminator_config)
    discriminator.init_weights()
    discriminator.train()
    if config.compile_model:
        generative_model = torch.compile(generative_model)  # type: ignore
        discriminator = torch.compile(discriminator)  # type: ignore
    print("Creating optimizer...")
    generator_optimizer = config.optimizer.create_optimizer(
        generative_model.parameters()
    )
    discriminator_optimizer = config.optimizer.create_optimizer(
        discriminator.parameters()
    )
    print("Starting training...")
    train_generator_losses = torch.zeros(
        (config.eval_interval), device="cpu", dtype=torch.float32
    )
    train_discriminator_losses = torch.zeros(
        (config.eval_interval), device="cpu", dtype=torch.float32
    )
    train_generator_losses += float("inf")
    train_discriminator_losses += float("inf")
    for step in range(config.optimizer.max_iters):
        eval_generator_losses = torch.zeros(
            (config.eval_iters), device="cpu", dtype=torch.float32
        )
        eval_discriminator_losses = torch.zeros(
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
            discriminator.eval()
            with torch.no_grad():
                for i in tqdm(range(config.eval_iters)):
                    batch = next(iter(test_dataloader))
                    batch = tuple(
                        [
                            t.to(device=config.unet.device, dtype=config.unet.dtype)
                            for t in batch
                        ]
                    )
                    transformed_images = generative_model(batch[0])
                    transformed_discriminator_loss = discriminator(
                        torch.cat([batch[0], transformed_images], dim=1),
                        torch.zeros(
                            (transformed_images.shape[0]),
                            device=config.discriminator.device,
                            dtype=config.discriminator.dtype,
                        ),
                    )
                    generator_loss = -transformed_discriminator_loss
                    deepdream_discriminator_loss = discriminator(
                        torch.cat([batch[0], batch[1]], dim=1),
                        torch.ones(
                            (transformed_images.shape[0]),
                            device=config.discriminator.device,
                            dtype=config.discriminator.dtype,
                        ),
                    )
                    random_mixes_ratios = torch.rand(
                        (transformed_images.shape[0], 1, 1, 1),
                        device=config.discriminator.device,
                        dtype=config.discriminator.dtype,
                    )
                    random_mixes = batch[
                        1
                    ] * random_mixes_ratios + transformed_images.detach() * (
                        1 - random_mixes_ratios
                    )
                    mixed_discriminator_loss = discriminator(
                        torch.cat([batch[0], random_mixes], dim=1),
                        random_mixes_ratios.squeeze(3).squeeze(2).squeeze(1),
                    )
                    discriminator_loss = (
                        transformed_discriminator_loss
                        + deepdream_discriminator_loss
                        + mixed_discriminator_loss
                    )
                    eval_generator_losses[i] = generator_loss.item()
                    eval_discriminator_losses[i] = discriminator_loss.item()
                    eval_transformed_discriminator_losses[
                        i
                    ] = transformed_discriminator_loss.item()
                    eval_deepdream_discriminator_losses[
                        i
                    ] = deepdream_discriminator_loss.item()
                    eval_mixed_losses[i] = mixed_discriminator_loss.item()
                # Save sample output image to `outputs` directory
                sample_input_image = ((batch[0][0] + 1) / 2).detach().cpu()
                sample_deepdream_image = ((batch[1][0] + 1) / 2).detach().cpu()
                sample_output_image = ((transformed_images[0] + 1) / 2).detach().cpu()
                sample_input_image = sample_input_image.permute(1, 2, 0)
                sample_deepdream_image = sample_deepdream_image.permute(1, 2, 0)
                sample_output_image = sample_output_image.permute(1, 2, 0)
                sample_input_image = sample_input_image.numpy()
                sample_deepdream_image = sample_deepdream_image.numpy()
                sample_output_image = sample_output_image.numpy()
                sample_input_image = (sample_input_image * 255).astype(np.uint8)
                sample_deepdream_image = (sample_deepdream_image * 255).astype(np.uint8)
                sample_output_image = (sample_output_image * 255).astype(np.uint8)
                if not os.path.exists("outputs/" + config.wandb_run_name):
                    os.makedirs("outputs/" + config.wandb_run_name)
                input_save_path = (
                    f"outputs/{config.wandb_run_name}/sample_input_{step}.jpg"
                )
                deepdream_save_path = (
                    f"outputs/{config.wandb_run_name}/sample_deepdream_{step}.jpg"
                )
                output_save_path = (
                    f"outputs/{config.wandb_run_name}/sample_output_{step}.jpg"
                )
                print(f"Saving sample output image to {output_save_path}")
                input_image_to_save = Image.fromarray(sample_input_image)
                deepdream_image_to_save = Image.fromarray(sample_deepdream_image)
                input_image_to_save.save(input_save_path)
                deepdream_image_to_save.save(deepdream_save_path)
                output_image_to_save = Image.fromarray(sample_output_image)
                output_image_to_save.save(output_save_path)
                noise_volume = generative_model.noise_proj.std().item()
                # Log eval metrics to wandb
                if config.wandb_log:
                    wandb.log(
                        {
                            "eval_generator_loss": eval_generator_losses.mean().item(),
                            "eval_generator_loss_std": eval_generator_losses.std().item(),
                            "eval_discriminator_loss": eval_discriminator_losses.mean().item(),
                            "eval_discriminator_losses_std": eval_discriminator_losses.std().item(),
                            "eval_transformed_discriminator_loss": eval_transformed_discriminator_losses.mean().item(),
                            "eval_deepdream_discriminator_loss": eval_deepdream_discriminator_losses.mean().item(),
                            "eval_mixed_discriminator_loss": eval_mixed_losses.mean().item(),
                            "train_generator_loss": train_generator_losses.mean().item(),
                            "train_generator_loss_std": train_generator_losses.std().item(),
                            "train_discriminator_loss": train_discriminator_losses.mean().item(),
                            "train_discriminator_loss_std": train_discriminator_losses.std().item(),
                            "lr": config.optimizer.get_lr(step),
                            "noise_volume": noise_volume,
                            "input_image": wandb.Image(
                                Image.fromarray(sample_input_image)
                            ),
                            "deepdream_image": wandb.Image(
                                Image.fromarray(sample_deepdream_image)
                            ),
                            "output_image": wandb.Image(
                                Image.fromarray(sample_output_image)
                            ),
                        },
                        step=step,
                    )
                # Print eval metrics
                print(
                    f"Step {step}: eval_generator_loss: {eval_generator_losses.mean().item()}, eval_discriminator_loss: {eval_discriminator_losses.mean().item()}, eval_transformed_discriminator_loss: {eval_transformed_discriminator_losses.mean().item()}, eval_deepdream_discriminator_loss: {eval_deepdream_discriminator_losses.mean().item()}, eval_mixed_discriminator_loss: {eval_mixed_losses.mean().item()}\n"
                    f"train_generator_loss: {train_generator_losses.mean().item()}, train_discriminator_loss: {train_discriminator_losses.mean().item()}\n"
                    f"lr: {config.optimizer.get_lr(step)}, noise_volume: {noise_volume}"
                )
            generative_model.train()
            discriminator.train()
        batch = next(iter(train_dataloader))
        batch = tuple(
            [t.to(device=config.unet.device, dtype=config.unet.dtype) for t in batch]
        )
        transformed_images = generative_model(batch[0])
        transformed_discriminator_loss = discriminator(
            torch.cat([batch[0], transformed_images], dim=1),
            torch.zeros(
                (transformed_images.shape[0]),
                device=config.discriminator.device,
                dtype=config.discriminator.dtype,
            ),
        )
        generator_loss = -transformed_discriminator_loss
        deepdream_discriminator_loss = discriminator(
            torch.cat([batch[0], batch[1]], dim=1),
            torch.ones(
                (transformed_images.shape[0]),
                device=config.discriminator.device,
                dtype=config.discriminator.dtype,
            ),
        )
        random_mixes_ratios = torch.rand(
            (transformed_images.shape[0], 1, 1, 1),
            device=config.discriminator.device,
            dtype=config.discriminator.dtype,
        )
        random_mixes = (
            random_mixes_ratios * batch[1]
            + (1 - random_mixes_ratios) * transformed_images
        )
        mixed_discriminator_loss = discriminator(
            torch.cat([batch[0], random_mixes], dim=1),
            random_mixes_ratios.squeeze(3).squeeze(2).squeeze(1),
        )
        discriminator_loss = (
            transformed_discriminator_loss * config.discriminator_generated_loss_factor
            + deepdream_discriminator_loss * config.discriminator_deepdream_loss_factor
            + mixed_discriminator_loss * config.discriminator_mixed_loss_factor
        )
        for param_group in discriminator_optimizer.param_groups:
            param_group["lr"] = (
                config.optimizer.get_lr(step) * config.discriminator_lr_multiplier
            )
        for param_group in generator_optimizer.param_groups:
            param_group["lr"] = (
                config.optimizer.get_lr(step) * config.generator_lr_multiplier
            )
        discriminator_optimizer.zero_grad()
        discriminator_loss.backward(retain_graph=True)
        discriminator_optimizer.step()
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()
        train_generator_losses[step % config.eval_interval] = generator_loss.detach()
        train_discriminator_losses[
            step % config.eval_interval
        ] = discriminator_loss.detach()
        if step % config.log_interval == 0:
            print(
                f"Step {step}, generator_loss: {generator_loss.detach()}, discriminator_loss: {discriminator_loss.detach()}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
