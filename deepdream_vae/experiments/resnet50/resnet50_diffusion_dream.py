import os
import random
import sys
from typing import Any, Iterator

import dacite
import hydra
import numpy as np
import torch
import torch.utils.data
import torchvision.models as models  # type: ignore
import wandb
from PIL import Image
from tqdm import tqdm

from deepdream_vae.conf.resnet50.resnet50_diffusion_dream import (
    Resnet50DiffusionDreamExperimentConf,
)
from deepdream_vae.datasets.folder_image_loader_dataset import (
    FolderImageLoaderDataset,
    FolderImageLoaderDatasetConfig,
)
from deepdream_vae.models.deepdream_vae import DeepdreamVAE, DeepdreamVAEConfig
from deepdream_vae.models.resnet_partial import create_n_layers_resnet
from deepdream_vae.utils.loop_dataloader import loop_dataloader


@hydra.main(
    config_path="../../conf/resnet50",
    config_name="resnet50_diffusion_dream.yaml",
    version_base=None,
)
def main(hydra_cfg: dict[Any, Any]) -> int:
    config: Resnet50DiffusionDreamExperimentConf = dacite.from_dict(
        data_class=Resnet50DiffusionDreamExperimentConf, data=hydra_cfg
    )
    if config.wandb_log:
        wandb.init(project="deepdream_diffusion", name=config.wandb_run_name)
        wandb.config.update(config)
    diffusion_model_config = DeepdreamVAEConfig(
        n_blocks=config.unet.n_blocks,
        n_layers_per_block=config.unet.n_layers_per_block,
        n_layers_mini_block=config.unet.n_layers_mini_block,
        n_first_block_channels=config.unet.n_first_block_channels,
        init_std=config.optimizer.init_std,
        activation=config.unet.activation,
        device=config.unet.device,
        dtype=config.unet.dtype,
        ln_eps=config.unet.ln_eps,
        image_size=config.image_size,
    )
    # Create forward and backward diffusion models
    print("Creating forward and backward diffusion models...")
    forward_diffusion_model = DeepdreamVAE(config=diffusion_model_config)
    forward_diffusion_model.init_weights()
    forward_diffusion_model.train()
    backward_diffusion_model = DeepdreamVAE(config=diffusion_model_config)
    backward_diffusion_model.init_weights()
    backward_diffusion_model.train()
    print("Loading pretrained resnet, and creating partial resnet...")
    resnet = models.resnet50(pretrained=True)
    partial_resnet = create_n_layers_resnet(
        n=config.n_blocks_resnet, original_resnet=resnet
    )
    partial_resnet.to(device=config.unet.device, dtype=config.unet.dtype)  # type: ignore
    partial_resnet.eval()
    print("Creating optimizer...")
    optimizer = config.optimizer.create_optimizer(
        [*forward_diffusion_model.parameters(), *backward_diffusion_model.parameters()]
    )
    print("Creating datasets...")
    train_dataset_config = FolderImageLoaderDatasetConfig(
        origin_path=config.source_files_path,
        is_train=True,
        image_size=config.image_size,
        scale_factor=config.scale_factor,
    )
    train_dataset = FolderImageLoaderDataset(train_dataset_config)
    test_dataset_config = FolderImageLoaderDatasetConfig(
        origin_path=config.source_files_path,
        is_train=False,
        image_size=config.image_size,
        scale_factor=config.scale_factor,
    )
    test_dataset = FolderImageLoaderDataset(test_dataset_config)
    print("Num train samples:", len(train_dataset))
    print("Num test samples:", len(test_dataset))
    train_dataloader: Iterator[torch.Tensor] = loop_dataloader(
        torch.utils.data.DataLoader(
            train_dataset,  # type: ignore
            batch_size=config.optimizer.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )
    )
    test_dataloader: Iterator[torch.Tensor] = loop_dataloader(
        torch.utils.data.DataLoader(
            test_dataset,  # type: ignore
            batch_size=config.optimizer.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )
    )
    best_eval_loss = float("inf")
    train_losses = torch.zeros(
        (config.eval_interval,),
        dtype=torch.float32,
        device="cpu",
    )
    train_losses += float("inf")
    train_resnet_losses = torch.zeros(
        (config.eval_interval,),
        dtype=torch.float32,
        device="cpu",
    )
    train_resnet_losses += float("inf")
    train_consistency_losses = torch.zeros(
        (config.eval_interval,),
        dtype=torch.float32,
        device="cpu",
    )
    train_consistency_losses += float("inf")
    train_diffusions_diffs = torch.zeros(
        (config.eval_interval,),
        dtype=torch.float32,
        device="cpu",
    )
    train_diffusions_diffs += float("inf")
    for step in range(config.optimizer.max_iters):
        if step % config.eval_interval == 0:
            with torch.no_grad():
                forward_diffusion_model.eval()
                backward_diffusion_model.eval()
                eval_losses = torch.zeros(
                    (config.eval_iters,),
                    dtype=torch.float32,
                    device="cpu",
                )
                eval_resnet_losses = torch.zeros(
                    (config.eval_iters,),
                    dtype=torch.float32,
                    device="cpu",
                )
                eval_consistency_losses = torch.zeros(
                    (config.eval_iters,),
                    dtype=torch.float32,
                    device="cpu",
                )
                eval_diffusions_diffs = torch.zeros(
                    (config.eval_iters,),
                    dtype=torch.float32,
                    device="cpu",
                )
                for j in tqdm(range(config.eval_iters)):
                    batch = next(test_dataloader)
                    batch = batch.to(config.unet.device, config.unet.dtype)
                    diffused = [batch]
                    resnet_activations_sums = [partial_resnet(batch).mean()]
                    resnet_activations_deltas = []
                    total_overflow = torch.zeros(
                        1, device=config.unet.device, dtype=config.unet.dtype
                    )
                    for t in range(config.num_diffusion_activations):
                        next_diffused = forward_diffusion_model(diffused[-1])
                        # Clip to [-1, 1]
                        diffused.append(torch.clamp(next_diffused, -1, 1))
                        total_overflow += ((next_diffused - diffused[-1]) ** 2).mean()
                        resnet_activations_sums.append(
                            partial_resnet(diffused[-1]).mean()
                        )
                        resnet_activations_deltas.append(
                            resnet_activations_sums[-1] - resnet_activations_sums[-2]
                        )
                    stacked_resnet_activations_deltas = torch.stack(
                        resnet_activations_deltas, dim=0
                    )
                    diffusion_diff = torch.zeros(
                        (1,), device=config.unet.device, dtype=config.unet.dtype
                    )
                    for i in range(config.num_diffusion_activations):
                        diffusion_diff += (
                            (diffused[i] - diffused[i + 1]) ** 2
                        ).mean() / config.num_diffusion_activations
                    undiffused = [diffused[-1]]
                    for t in range(config.num_diffusion_activations):
                        next_undiffused = backward_diffusion_model(undiffused[-1])
                        undiffused.append(torch.clamp(next_undiffused, -1, 1))
                        total_overflow += (
                            (next_undiffused - undiffused[-1]) ** 2
                        ).mean()
                    undiffused = undiffused[::-1]
                    reconstructions_l2s = torch.stack(
                        [
                            ((diffused[k] - undiffused[k]) ** 2).mean()
                            for k in range(config.num_diffusion_activations)
                        ],
                        dim=0,
                    )
                    # Prioritize the worst diffusion steps
                    resnet_activations_softmins = torch.nn.functional.softmax(
                        -stacked_resnet_activations_deltas, dim=0
                    )
                    eval_resnet_losses[j] = -torch.einsum(
                        "t,t->",
                        resnet_activations_softmins,
                        stacked_resnet_activations_deltas,
                    ).item()
                    eval_consistency_losses[j] = reconstructions_l2s.mean().item()
                    eval_losses[j] = (
                        eval_resnet_losses[j]
                        + eval_consistency_losses[j] * config.l2_consistency_loss_weight
                    )
                    eval_diffusions_diffs[j] = diffusion_diff.item()
                eval_loss = eval_losses.mean().item()
                eval_resnet_loss = eval_resnet_losses.mean().item()
                eval_consistency_loss = eval_consistency_losses.mean().item()
                eval_loss_std = eval_losses.std().item()
                eval_resnet_loss_std = eval_resnet_losses.std().item()
                eval_consistency_loss_std = eval_consistency_losses.std().item()
                train_loss = train_losses.mean().item()
                train_resnet_loss = train_resnet_losses.mean().item()
                train_consistency_loss = train_consistency_losses.mean().item()
                train_loss_std = train_losses.std().item()
                train_resnet_loss_std = train_resnet_losses.std().item()
                train_consistency_loss_std = train_consistency_losses.std().item()
                eval_diffusions_diff = eval_diffusions_diffs.mean().item()
                train_diffusions_diff = train_diffusions_diffs.mean().item()
                print(
                    f"Step {step}: Eval loss: {eval_loss:.3f} +- {eval_loss_std:.3f} (resnet: {eval_resnet_loss:.3f} +- {eval_resnet_loss_std:.3f}, consistency: {eval_consistency_loss:.3f} +- {eval_consistency_loss_std:.3f}),"
                    f"Train loss: {train_loss:.3f} +- {train_loss_std:.3f} (resnet: {train_resnet_loss:.3f} +- {train_resnet_loss_std:.3f}, consistency: {train_consistency_loss:.3f} +- {train_consistency_loss_std:.3f})"
                    f"Total overflow: {total_overflow.item()}, diffused diff: {eval_diffusions_diff}, train diffused diff: {train_diffusions_diff}",
                )
                source_image = batch[0] * 0.5 + 0.5
                diffused_image = diffused[-1][0] * 0.5 + 0.5
                undiffused_image = undiffused[0][0] * 0.5 + 0.5
                # Clip to [0, 1]
                source_image = torch.clamp(source_image, 0, 1)
                diffused_image = torch.clamp(diffused_image, 0, 1)
                undiffused_image = torch.clamp(undiffused_image, 0, 1)
                if config.wandb_log:
                    wandb.log(
                        {
                            "eval_loss": eval_loss,
                            "eval_loss_std": eval_loss_std,
                            "eval_resnet_loss": eval_resnet_loss,
                            "eval_resnet_loss_std": eval_resnet_loss_std,
                            "eval_consistency_loss": eval_consistency_loss,
                            "eval_consistency_loss_std": eval_consistency_loss_std,
                            "train_loss": train_loss,
                            "train_loss_std": train_loss_std,
                            "train_resnet_loss": train_resnet_loss,
                            "train_resnet_loss_std": train_resnet_loss_std,
                            "train_consistency_loss": train_consistency_loss,
                            "train_consistency_loss_std": train_consistency_loss_std,
                            "source_image": wandb.Image(source_image),
                            "diffused_image": wandb.Image(diffused_image),
                            "undiffused_image": wandb.Image(undiffused_image),
                            "total_overflow": total_overflow.item(),
                            "diffused_diff": eval_diffusions_diff,
                            "train_diffused_diff": train_diffusions_diff,
                        },
                        step=step,
                    )
                out_dir = "outputs/" + config.wandb_run_name
                os.makedirs(out_dir, exist_ok=True)
                # Save images
                source_image_np = (
                    source_image.detach().cpu().permute(1, 2, 0).numpy() * 255
                ).astype(np.uint8)
                diffused_image_np = (
                    diffused_image.detach().cpu().permute(1, 2, 0).numpy() * 255
                ).astype(np.uint8)
                undiffused_image_np = (
                    undiffused_image.detach().cpu().permute(1, 2, 0).numpy() * 255
                ).astype(np.uint8)
                source_image_pil = Image.fromarray(source_image_np)
                diffused_image_pil = Image.fromarray(diffused_image_np)
                undiffused_image_pil = Image.fromarray(undiffused_image_np)
                source_image_pil.save(os.path.join(out_dir, f"source_{step}.png"))
                diffused_image_pil.save(os.path.join(out_dir, f"diffused_{step}.png"))
                undiffused_image_pil.save(
                    os.path.join(out_dir, f"undiffused_{step}.png")
                )
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    print("Saving checkpoint to", out_dir)
                    torch.save(
                        {
                            "step": step,
                            "forward_diffusion_model": forward_diffusion_model.state_dict(),
                            "backward_diffusion_model": backward_diffusion_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        os.path.join(out_dir, "checkpoint.pt"),
                    )
        batch = next(train_dataloader)
        batch = batch.to(config.unet.device, config.unet.dtype)
        diffused = [batch]
        resnet_activations_sums = [partial_resnet(batch).mean()]
        resnet_activations_deltas = []
        num_active_in_step = random.randint(1, config.num_diffusion_activations)
        total_overflow = torch.zeros(
            1, device=config.unet.device, dtype=config.unet.dtype
        )
        for t in range(num_active_in_step):
            next_diffused = forward_diffusion_model(diffused[-1])
            diffused.append(torch.clamp(next_diffused, -1, 1))
            total_overflow += ((next_diffused - diffused[-1]) ** 2).mean()
            resnet_activations_sums.append(partial_resnet(diffused[-1]).mean())
            resnet_activations_deltas.append(
                resnet_activations_sums[-1] - resnet_activations_sums[-2]
            )
        diffusion_diff = torch.zeros(
            (1,), device=config.unet.device, dtype=config.unet.dtype
        )
        for i in range(num_active_in_step):
            diffusion_diff += ((diffused[i] - diffused[i + 1]) ** 2).mean() / (
                num_active_in_step
            )
        stacked_resnet_activations_deltas = torch.stack(
            resnet_activations_deltas, dim=0
        )
        undiffused = [diffused[-1]]
        for t in range(num_active_in_step):
            next_undiffused = backward_diffusion_model(undiffused[-1])
            undiffused.append(torch.clamp(next_undiffused, -1, 1))
            total_overflow += ((next_undiffused - undiffused[-1]) ** 2).mean()
        undiffused = undiffused[::-1]
        reconstructions_l2s = torch.stack(
            [
                ((diffused[k] - undiffused[k]) ** 2).mean()
                for k in range(num_active_in_step)
            ],
            dim=0,
        )
        # Prioritize the worst diffusion steps
        resnet_activations_softmins = torch.nn.functional.softmax(
            -stacked_resnet_activations_deltas, dim=0
        )
        resnet_loss = -torch.einsum(
            "t,t->", resnet_activations_softmins, stacked_resnet_activations_deltas
        )
        consistency_loss = reconstructions_l2s.mean()
        loss = (
            resnet_loss
            + consistency_loss * config.l2_consistency_loss_weight
            + total_overflow * config.overflow_loss_weight
            + diffusion_diff * config.l2_diffusion_diff_loss_weight
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = config.optimizer.get_lr(step)
        optimizer.zero_grad()
        loss.backward()  # type: ignore
        # Print gradients
        # for name, param in forward_diffusion_model.named_parameters():
        #     print(name, param.grad)
        # for name, param in backward_diffusion_model.named_parameters():
        #     print(name, param.grad)
        optimizer.step()
        train_losses[step % config.eval_interval] = loss.item()
        train_resnet_losses[step % config.eval_interval] = resnet_loss.item()
        train_consistency_losses[step % config.eval_interval] = consistency_loss.item()
        train_diffusions_diffs[step % config.eval_interval] = diffusion_diff.item()
        if step % config.log_interval == 0:
            print(
                f"Step {step}: Loss: {loss.item():.3f} (resnet: {resnet_loss.item():.3f}, consistency: {consistency_loss.item():.3f}, overflow: {total_overflow.item()}, diffused diff: {diffusion_diff.item()})"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
