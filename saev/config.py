from dataclasses import dataclass

import beartype
import torch

import wandb


@beartype.beartype
@dataclass
class Config:
    """
    Configuration for training a sparse autoencoder on a vision transformer.
    """

    # Data Generating Function (Model + Training Distibuion)
    image_width: int = 224
    image_height: int = 224
    model_name: str = "openai/clip-vit-large-patch14"
    module_name: str = "resid"
    block_layer: int = -2
    dataset_path: str = "ILSVRC/imagenet-1k"

    # SAE Parameters
    d_in: int = 1024

    # Activation Store Parameters
    total_training_tokens: int = 2_621_440
    n_batches_in_store: int = 15
    store_size: int | None = None
    vit_batch_size: int = 1024

    # SAE Parameters
    expansion_factor: int = 64

    # Training Parameters
    l1_coefficient: float = 0.00008
    lr: float = 0.0004
    lr_warm_up_steps: int = 500
    batch_size: int = 1024

    # Resampling protocol args
    use_ghost_grads: bool = True
    feature_sampling_window: int = 64
    resample_batches: int = 32
    feature_reinit_scale: float = 0.2
    dead_feature_window: int = 64
    dead_feature_estimation_method: str = "no_fire"
    dead_feature_threshold: float = 1e-6

    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "mats-hugo"
    wandb_log_freq: int = 10

    # Misc
    device: str = "cuda"
    seed: int = 42
    dtype: torch.dtype = torch.float32
    checkpoint_path: str = "checkpoints"

    def __post_init__(self):
        self.store_size = self.n_batches_in_store * self.batch_size

        self.d_sae = self.d_in * self.expansion_factor

        self.run_name = f"{self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-Tokens-{self.total_training_tokens:3.3e}"

        self.device = torch.device(self.device)

        unique_id = wandb.util.generate_id()
        self.checkpoint_path = f"{self.checkpoint_path}/{unique_id}"

        print(
            f"Run name: {self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-Tokens-{self.total_training_tokens:3.3e}"
        )
        # Print out some useful info:

        total_training_steps = self.total_training_tokens // self.batch_size
        print(f"Total training steps: {total_training_steps}")

        total_wandb_updates = total_training_steps // self.wandb_log_freq
        print(f"Total wandb updates: {total_wandb_updates}")

        # how many times will we sample dead neurons?
        # assert self.dead_feature_window <= self.feature_sampling_window, "dead_feature_window must be smaller than feature_sampling_window"
        n_feature_window_samples = total_training_steps // self.feature_sampling_window
        print(
            f"n_tokens_per_feature_sampling_window (millions): {(self.feature_sampling_window * self.batch_size) / 10** 6}"
        )
        print(
            f"n_tokens_per_dead_feature_window (millions): {(self.dead_feature_window * self.batch_size) / 10** 6}"
        )

        if self.use_ghost_grads:
            print("Using Ghost Grads.")

        print(
            f"We will reset the sparsity calculation {n_feature_window_samples} times."
        )
        print(
            f"Number of tokens when resampling: {self.resample_batches * self.batch_size}"
        )
        print(
            f"Number tokens in sparsity calculation window: {self.feature_sampling_window * self.batch_size:.2e}"
        )


#################
# COMPATIBILITY #
#################


# For compatibility with older (pickled) checkpoints.
# The classes are the same, just named differently.


ViTSAERunnerConfig = Config
