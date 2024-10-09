import torch
import tyro

from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import SessionLoader
from vit_sae_analysis.dashboard_fns import get_feature_data


def main(ckpt_path: str):
    loaded_object = torch.load(ckpt_path)
    cfg = loaded_object["cfg"]
    state_dict = loaded_object["state_dict"]

    sparse_autoencoder = SparseAutoencoder(cfg)
    sparse_autoencoder.load_state_dict(state_dict)
    sparse_autoencoder.eval()

    loader = SessionLoader(cfg)
    model = loader.get_model(cfg.model_name)
    model.to(cfg.device)

    get_feature_data(
        sparse_autoencoder,
        model,
        number_of_images=524_288,
        number_of_max_activating_images=20,
    )


if __name__ == "__main__":
    tyro.cli(main)
