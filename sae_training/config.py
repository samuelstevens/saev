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
    model_name: str = "openai/clip-vit-base-patch32"
    module_name: str = "resid"
    block_layer: int = 10
    dataset_path: str = "ILSVRC/imagenet-1k"

    # SAE Parameters
    d_in: int = 768

    # Activation Store Parameters
    total_training_tokens: int = 2_000_000
    n_batches_in_store: int = 32
    store_size: int | None = None
    vit_batch_size: int = 1024

    # Misc
    device: str = "cpu"
    seed: int = 42
    dtype: torch.dtype = torch.float32

    # SAE Parameters
    expansion_factor: int = 4
    from_pretrained_path: str | None = None

    # Training Parameters
    l1_coefficient: float = 1e-3
    lr: float = 3e-4
    lr_warm_up_steps: int = 500
    batch_size: int = 4096

    # Resampling protocol args
    use_ghost_grads: bool = True
    feature_sampling_window: int = (
        2000  # May need to change this since by default I will use ghost grads
    )
    resample_batches: int = 32
    feature_reinit_scale: float = 0.2
    dead_feature_window: int = 1000  # unless this window is larger feature sampling,
    dead_feature_estimation_method: str = "no_fire"
    dead_feature_threshold: float = 1e-8

    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "mats-hugo"
    wandb_log_frequency: int = 10

    # Misc
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

        total_wandb_updates = total_training_steps // self.wandb_log_frequency
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
