import sys

import torch

from vit_sae_analysis.dashboard_fns import get_feature_data

sys.path.append("..")

from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import (
    ViTSparseAutoencoderSessionloader,
)

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

sae_path = ""

loaded_object = torch.load(sae_path)

cfg = loaded_object["cfg"]

state_dict = loaded_object["state_dict"]

sparse_autoencoder = SparseAutoencoder(cfg)

sparse_autoencoder.load_state_dict(state_dict)

sparse_autoencoder.eval()

loader = ViTSparseAutoencoderSessionloader(cfg)

model = loader.get_model(cfg.model_name)

model.to(cfg.device)

get_feature_data(
    sparse_autoencoder,
    model,
    number_of_images=524_288,
    number_of_max_activating_images=20,
)
