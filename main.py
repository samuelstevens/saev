import os
import tyro

import torch

from analysis import get_feature_data
from sae_training.config import Config
from sae_training.training import train

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"


def main(cfg: Config):
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

    main(tyro.cli(Config))
