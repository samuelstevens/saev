import os

import torch

from analysis import get_feature_data
from sae_training.config import Config
from sae_training.training import train

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"


def main():
    cfg = Config(
        # Data Generating Function (Model + Training Distibuion)
        image_width=224,
        image_height=224,
        model_name="openai/clip-vit-large-patch14",
        module_name="resid",
        block_layer=-2,
        dataset_path="ILSVRC/imagenet-1k",
        d_in=1024,
        # SAE Parameters
        expansion_factor=64,
        # Training Parameters
        lr=0.0004,
        l1_coefficient=0.00008,
        batch_size=1024,
        lr_warm_up_steps=500,
        total_training_tokens=2_621_440,
        n_batches_in_store=15,
        # Dead Neurons and Sparsity
        use_ghost_grads=True,
        feature_sampling_window=64,
        dead_feature_window=64,
        dead_feature_threshold=1e-6,
        # WANDB
        log_to_wandb=True,
        wandb_project="mats-hugo",
        wandb_log_freq=10,
        # Misc
        device="cuda",
        seed=42,
        checkpoint_path="checkpoints",
        dtype=torch.float32,
    )

    sae, vit = train(cfg)

    get_feature_data(sae, vit, n_images=524_288, k_top_images=20)


if __name__ == "__main__":
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    main()
