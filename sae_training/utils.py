import beartype
import torch

from sae_training.activations_store import ActivationsStore
from sae_training.config import Config
from sae_training.hooked_vit import HookedVisionTransformer
from sae_training.sparse_autoencoder import SparseAutoencoder


@beartype.beartype
def new_session(
    cfg: Config,
) -> tuple[HookedVisionTransformer, SparseAutoencoder, ActivationsStore]:
    vit = HookedVisionTransformer(cfg.model_name)
    vit.eval()
    for parameter in vit.model.parameters():
        parameter.requires_grad_(False)
    vit.to(cfg.device)

    sae = SparseAutoencoder(cfg)
    activations_store = ActivationsStore(cfg, vit)

    return vit, sae, activations_store


@beartype.beartype
def load_session(
    path: str,
) -> tuple[HookedVisionTransformer, SparseAutoencoder, ActivationsStore]:
    if torch.cuda.is_available():
        cfg = torch.load(path, weights_only=False)["cfg"]
    else:
        cfg = torch.load(path, map_location="cpu", weights_only=False)["cfg"]

    vit, _, activations_loader = new_session(cfg)
    sae = SparseAutoencoder.load_from_pretrained(path)

    return vit, sae, activations_loader
