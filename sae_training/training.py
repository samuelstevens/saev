import json
import os
import re

import beartype
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Adam
from tqdm import tqdm

import wandb
from sae_training.activations_store import ActivationsStore
from sae_training.config import Config
from sae_training.hooked_vit import HookedVisionTransformer
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import new_session


def train(cfg: Config) -> tuple[SparseAutoencoder, HookedVisionTransformer]:
    vit, sae, activations_loader = new_session(cfg)

    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, config=cfg, name=cfg.run_name)

    # train SAE
    sae = train_sae(sae, activations_loader)

    # save sae to checkpoints folder
    path = f"{cfg.checkpoint_path}/final_{sae.get_name()}.pt"
    sae.save_model(path)

    # upload to wandb
    if cfg.log_to_wandb:
        model_artifact = wandb.Artifact(
            f"{re.sub(r'[^a-zA-Z0-9]', '', sae.get_name())}",
            type="model",
            metadata=dict(cfg.__dict__),
        )
        model_artifact.add_file(path)
        wandb.log_artifact(model_artifact, aliases=["final_model"])

    if cfg.log_to_wandb:
        wandb.finish()

    return sae, vit


@beartype.beartype
def train_sae(
    sae: SparseAutoencoder, activation_store: ActivationsStore
) -> SparseAutoencoder:
    batch_size = sae.cfg.batch_size
    total_training_tokens = sae.cfg.total_training_tokens

    n_steps = 0
    n_training_tokens = 0

    # track active features
    act_freq_scores = torch.zeros(sae.cfg.d_sae, device=sae.cfg.device)
    n_fwd_passes_since_fired = torch.zeros(sae.cfg.d_sae, device=sae.cfg.device)
    n_frac_active_tokens = 0

    optimizer = Adam(sae.parameters(), lr=sae.cfg.lr)
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda steps: min(1.0, (steps + 1) / sae.cfg.lr_warm_up_steps),
    )

    sae.initialize_b_dec(activation_store)

    sae.train()

    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    while n_training_tokens < total_training_tokens:
        # Do a training step.
        sae.train()
        # Make sure the W_dec is still zero-norm
        sae.set_decoder_norm_to_unit_norm()

        # after resampling, reset the sparsity:
        if (n_steps + 1) % sae.cfg.feature_sampling_window == 0:
            # feature_sampling_window divides dead_sampling_window
            feature_sparsity = act_freq_scores / n_frac_active_tokens
            log_feature_sparsity = torch.log10(feature_sparsity + 1e-10).detach().cpu()

            if sae.cfg.log_to_wandb:
                wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())
                wandb.log(
                    {
                        "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
                        "plots/feature_density_line_chart": wandb_histogram,
                    },
                    step=n_steps,
                )

            act_freq_scores = torch.zeros(sae.cfg.d_sae, device=sae.cfg.device)
            n_frac_active_tokens = 0

        scheduler.step()
        optimizer.zero_grad()

        ghost_grad_neuron_mask = (
            n_fwd_passes_since_fired > sae.cfg.dead_feature_window
        ).bool()
        sae_in = activation_store.next_batch()
        # breakpoint()

        # Forward and Backward Passes
        sae_out, feature_acts, loss, mse_loss, l1_loss, ghost_grad_loss = sae(
            sae_in,
            ghost_grad_neuron_mask,
        )
        did_fire = (feature_acts > 0).float().sum(-2) > 0
        n_fwd_passes_since_fired += 1
        # breakpoint()
        n_fwd_passes_since_fired[did_fire] = 0  # TODO: is there a leak here?

        n_training_tokens += batch_size

        with torch.no_grad():
            # Calculate the sparsities, and add it to a list, calculate sparsity metrics
            act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
            n_frac_active_tokens += batch_size
            feature_sparsity = act_freq_scores / n_frac_active_tokens

            if sae.cfg.log_to_wandb and ((n_steps + 1) % sae.cfg.wandb_log_freq == 0):
                # metrics for currents acts
                l0 = (feature_acts > 0).float().sum(-1).mean()
                current_learning_rate = optimizer.param_groups[0]["lr"]

                per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
                total_variance = sae_in.pow(2).sum(-1)
                explained_variance = 1 - per_token_l2_loss / total_variance

                metrics = {
                    # losses
                    "losses/mse_loss": mse_loss.item(),
                    "losses/l1_loss": l1_loss.item()
                    / sae.l1_coefficient,  # normalize by l1 coefficient
                    "losses/ghost_grad_loss": ghost_grad_loss.item(),
                    "losses/overall_loss": loss.item(),
                    # variance explained
                    "metrics/explained_variance": explained_variance.mean().item(),
                    "metrics/explained_variance_std": explained_variance.std().item(),
                    "metrics/l0": l0.item(),
                    # sparsity
                    "sparsity/mean_passes_since_fired": n_fwd_passes_since_fired.mean().item(),
                    "sparsity/n_passes_since_fired_over_threshold": ghost_grad_neuron_mask.sum().item(),
                    "sparsity/below_1e-5": (feature_sparsity < 1e-5)
                    .float()
                    .mean()
                    .item(),
                    "sparsity/below_1e-6": (feature_sparsity < 1e-6)
                    .float()
                    .mean()
                    .item(),
                    "sparsity/dead_features": (
                        feature_sparsity < sae.cfg.dead_feature_threshold
                    )
                    .float()
                    .mean()
                    .item(),
                    "details/n_training_tokens": n_training_tokens,
                    "details/current_learning_rate": current_learning_rate,
                }
                wandb.log(metrics, step=n_steps)

                directory = f"metrics/{wandb.run.id}"
                os.makedirs(directory, exist_ok=True)
                with open(f"{directory}/step{n_steps}.json", "w") as fd:
                    json.dump(metrics, fd)

                try:
                    with open(f"metrics/reference/step{n_steps}.json") as fd:
                        reference = json.load(fd)
                        breakpoint()
                        print("Check that `reference` and `metrics` are similar.")
                        print("reference:", reference)
                        print("metrics:", metrics)
                except FileNotFoundError:
                    pass

            pbar.set_description(
                f"{n_steps}| MSE Loss {mse_loss.item():.3f} | L1 {l1_loss.item():.3f}"
            )
            pbar.update(batch_size)

        loss.backward()
        sae.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()

        n_steps += 1

    return sae
