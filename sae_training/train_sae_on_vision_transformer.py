from functools import partial

import numpy as np
import plotly_express as px
import torch
from torch.optim import Adam
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

import wandb
from sae_training.activations_store import ActivationsStore
from sae_training.vit_activations_store import ViTActivationsStore
from sae_training.optim import get_scheduler
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.hooked_vit import HookedVisionTransformer, Hook
from sae_training.config import ViTSAERunnerConfig


def train_sae_on_vision_transformer(
    model: HookedVisionTransformer,
    sparse_autoencoder: SparseAutoencoder,
    activation_store: ViTActivationsStore,
    batch_size: int = 1024, # Isn't this contained in sparse_autoencoder.cfg.batch_size??
    n_checkpoints: int = 0,
    feature_sampling_method: str = "l2",  # None, l2, or anthropic # This is also contained in the .cfg object!
    feature_sampling_window: int = 1000,  # how many training steps between resampling the features / considiring neurons dead
    feature_reinit_scale: float = 0.2,  # how much to scale the resampled features by
    dead_feature_threshold: float = 1e-8,  # how infrequently a feature has to be active to be considered dead
    dead_feature_window: int = 2000,  # how many training steps before a feature is considered dead
    use_wandb: bool = False,
    wandb_log_frequency: int = 50,
):
    if use_wandb:
        wandb.init(project="mats-hugo")
    if feature_sampling_method is not None:
        feature_sampling_method = feature_sampling_method.lower()

    total_training_tokens = sparse_autoencoder.cfg.total_training_tokens
    total_training_steps = total_training_tokens // batch_size
    n_training_steps = 0
    n_training_tokens = 0
    n_resampled_neurons = 0
    steps_before_reset = 0
    if n_checkpoints > 0:
        checkpoint_thresholds = list(range(0, total_training_tokens, total_training_tokens // n_checkpoints))[1:]
    
    # track active features
    act_freq_scores = torch.zeros(sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device)
    n_forward_passes_since_fired = torch.zeros(sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device)
    n_frac_active_tokens = 0
    
    optimizer = Adam(sparse_autoencoder.parameters(),
                     lr = sparse_autoencoder.cfg.lr)
    scheduler = get_scheduler(
        sparse_autoencoder.cfg.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps = sparse_autoencoder.cfg.lr_warm_up_steps, 
        training_steps=total_training_steps,
        lr_end=sparse_autoencoder.cfg.lr / 10, # heuristic for now. 
    )
    sparse_autoencoder.initialize_b_dec(activation_store)
    sparse_autoencoder.train()
    

    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    while n_training_tokens < total_training_tokens:
        # Do a training step.
        sparse_autoencoder.train()
        # Make sure the W_dec is still zero-norm
        sparse_autoencoder.set_decoder_norm_to_unit_norm()


        if (feature_sampling_method=="anthropic") and ((n_training_steps + 1) % dead_feature_window == 0):
            
            feature_sparsity = act_freq_scores / n_frac_active_tokens
            
            # if reset criterion is frequency in window, then then use that to generate indices.
            if sparse_autoencoder.cfg.dead_feature_estimation_method == "no_fire":
                dead_neuron_indices = (act_freq_scores == 0).nonzero(as_tuple=False)[:, 0]
            elif sparse_autoencoder.cfg.dead_feature_estimation_method == "frequency":
                dead_neuron_indices = (feature_sparsity < sparse_autoencoder.cfg.dead_feature_threshold).nonzero(as_tuple=False)[:, 0]
            
            if len(dead_neuron_indices) > 0:
                
                if len(dead_neuron_indices) > sparse_autoencoder.cfg.resample_batches * sparse_autoencoder.cfg.store_batch_size:
                    print("Warning: more dead neurons than number of tokens. Consider sampling more tokens when resampling.")
                
                sparse_autoencoder.resample_neurons_anthropic(
                    dead_neuron_indices, 
                    model,
                    optimizer, 
                    activation_store
                )

                if use_wandb:
                    n_resampled_neurons = min(len(dead_neuron_indices), sparse_autoencoder.cfg.store_batch_size * sparse_autoencoder.cfg.resample_batches)
                    wandb.log(
                        {
                            "metrics/n_resampled_neurons": n_resampled_neurons,
                        },
                        step=n_training_steps,
                    )
                
                # for now, we'll hardcode this.
                current_lr = scheduler.get_last_lr()[0]
                reduced_lr = current_lr / 10_000
                increment = (current_lr - reduced_lr) / 10_000
                optimizer.param_groups[0]['lr'] = reduced_lr
                steps_before_reset = 10_000
            else:
                print("No dead neurons, skipping resampling")
            
        # Resample dead neurons
        if (feature_sampling_method == "l2") and ((n_training_steps + 1) % dead_feature_window == 0):
            print("no l2 resampling currently. Please use anthropic resampling")
            
        # after resampling, reset the sparsity:
        if (n_training_steps + 1) % feature_sampling_window == 0:
            
            feature_sparsity = act_freq_scores / n_frac_active_tokens
            log_feature_sparsity = torch.log10(feature_sparsity + 1e-10).detach().cpu()

            if use_wandb:
                wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())
                wandb.log(
                    {   
                        "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
                        "plots/feature_density_line_chart": wandb_histogram,
                    },
                    step=n_training_steps,
                )
            
            act_freq_scores = torch.zeros(sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device)
            n_frac_active_tokens = 0



        if (steps_before_reset > 0) and n_training_steps > 0:
            steps_before_reset -= 1
            optimizer.param_groups[0]['lr'] += increment
            if steps_before_reset == 0:
                optimizer.param_groups[0]['lr'] = current_lr
        else:
            scheduler.step()
    
        optimizer.zero_grad()
        
        ghost_grad_neuron_mask = (n_forward_passes_since_fired > sparse_autoencoder.cfg.dead_feature_window).bool()
        sae_in = activation_store.next_batch()
        
        # Forward and Backward Passes
        sae_out, feature_acts, loss, mse_loss, l1_loss, ghost_grad_loss = sparse_autoencoder(
            sae_in,
            ghost_grad_neuron_mask,
        )
        did_fire = ((feature_acts > 0).float().sum(-2) > 0)
        n_forward_passes_since_fired += 1
        n_forward_passes_since_fired[did_fire] = 0
        
        n_training_tokens += batch_size

        with torch.no_grad():
            # Calculate the sparsities, and add it to a list, calculate sparsity metrics
            act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
            n_frac_active_tokens += batch_size
            feature_sparsity = act_freq_scores / n_frac_active_tokens

            if use_wandb and ((n_training_steps + 1) % wandb_log_frequency == 0):
                # metrics for currents acts
                l0 = (feature_acts > 0).float().sum(-1).mean()
                current_learning_rate = optimizer.param_groups[0]["lr"]
                
                per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
                total_variance = sae_in.pow(2).sum(-1)
                explained_variance = 1 - per_token_l2_loss/total_variance
                
                wandb.log(
                    {
                        # losses
                        "losses/mse_loss": mse_loss.item(),
                        "losses/l1_loss": l1_loss.item() / sparse_autoencoder.l1_coefficient, # normalize by l1 coefficient
                        "losses/ghost_grad_loss": ghost_grad_loss.item(),
                        "losses/overall_loss": loss.item(),
                        # variance explained
                        "metrics/explained_variance": explained_variance.mean().item(),
                        "metrics/explained_variance_std": explained_variance.std().item(),
                        "metrics/l0": l0.item(),
                        # sparsity
                        "sparsity/mean_passes_since_fired": n_forward_passes_since_fired.mean().item(),
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
                            feature_sparsity < dead_feature_threshold
                        )
                        .float()
                        .mean()
                        .item(),
                        "details/n_training_tokens": n_training_tokens,
                        "details/current_learning_rate": current_learning_rate,
                    },
                    step=n_training_steps,
                )

            # record loss frequently, but not all the time.
            if use_wandb and ((n_training_steps + 1) % (wandb_log_frequency * 10) == 0):
                if "cuda" in str(model.cfg.device):
                    torch.cuda.empty_cache()
                ###########################
                # Need to sort this out!!
                ###########################
                # sparse_autoencoder.eval()
                # run_evals(sparse_autoencoder, activation_store, model, n_training_steps)
                # sparse_autoencoder.train()
                
            pbar.set_description(
                f"{n_training_steps}| MSE Loss {mse_loss.item():.3f} | L1 {l1_loss.item():.3f}"
            )
            pbar.update(batch_size)

        loss.backward()
        sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()


        # checkpoint if at checkpoint frequency
        if n_checkpoints > 0 and n_training_tokens > checkpoint_thresholds[0]:
            cfg = sparse_autoencoder.cfg
            path = f"{sparse_autoencoder.cfg.checkpoint_path}/{n_training_tokens}_{sparse_autoencoder.get_name()}.pt"
            log_feature_sparsity_path = f"{sparse_autoencoder.cfg.checkpoint_path}/{n_training_tokens}_{sparse_autoencoder.get_name()}_log_feature_sparsity.pt"
            sparse_autoencoder.save_model(path)
            torch.save(log_feature_sparsity, log_feature_sparsity_path)
            checkpoint_thresholds.pop(0)
            if len(checkpoint_thresholds) == 0:
                n_checkpoints = 0
            if cfg.log_to_wandb:
                model_artifact = wandb.Artifact(
                    f"{sparse_autoencoder.get_name()}", type="model", metadata=dict(cfg.__dict__)
                )
                model_artifact.add_file(path)
                wandb.log_artifact(model_artifact)
                
                sparsity_artifact = wandb.Artifact(
                    f"{sparse_autoencoder.get_name()}_log_feature_sparsity", type="log_feature_sparsity", metadata=dict(cfg.__dict__)
                )
                sparsity_artifact.add_file(log_feature_sparsity_path)
                wandb.log_artifact(sparsity_artifact)
                
            
        n_training_steps += 1
        
        
    if n_checkpoints > 0:
        log_feature_sparsity_path = f"{sparse_autoencoder.cfg.checkpoint_path}/final_{sparse_autoencoder.get_name()}_log_feature_sparsity.pt"
        sparse_autoencoder.save_model(path)
        torch.save(log_feature_sparsity, log_feature_sparsity_path)
        if cfg.log_to_wandb:
            sparsity_artifact = wandb.Artifact(
                    f"{sparse_autoencoder.get_name()}_log_feature_sparsity", type="log_feature_sparsity", metadata=dict(cfg.__dict__)
                )
            sparsity_artifact.add_file(log_feature_sparsity_path)
            wandb.log_artifact(sparsity_artifact)
        

    return sparse_autoencoder


@torch.no_grad()
def run_evals(sparse_autoencoder: SparseAutoencoder, activation_store: ViTActivationsStore, model: HookedVisionTransformer, n_training_steps: int):
    # Need to rewrite this function for use with ViTs. What does this function even do?
    pass


def kl_divergence_attention(y_true, y_pred):

    # Compute log probabilities for KL divergence
    log_y_true = torch.log2(y_true + 1e-10)
    log_y_pred = torch.log2(y_pred + 1e-10)

    return y_true * (log_y_true - log_y_pred)