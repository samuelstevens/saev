import collections.abc

import beartype
import torch

from sae_training.activations_store import ActivationsStore
from sae_training.config import Config
from sae_training.hooked_vit import HookedVisionTransformer
from sae_training.sparse_autoencoder import SparseAutoencoder


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


def load_session(
    path: str,
) -> tuple[HookedVisionTransformer, SparseAutoencoder, ActivationsStore]:
    if torch.backends.mps.is_available():
        cfg = torch.load(path, map_location="mps")["cfg"]
        cfg.device = "mps"
    elif torch.cuda.is_available():
        cfg = torch.load(path, map_location="cuda")["cfg"]
    else:
        cfg = torch.load(path, map_location="cpu")["cfg"]

    vit, _, activations_loader = new_session(cfg)
    sae = SparseAutoencoder.load_from_pretrained(path)

    return vit, sae, activations_loader


@beartype.beartype
def batched_idx(
    total_size: int, batch_size: int
) -> collections.abc.Iterator[tuple[int, int]]:
    """
    Iterate over (start, end) indices for total_size examples, where end - start is at most batch_size.

    Args:
        total_size: total number of examples
        batch_size: maximum distance between the generated indices.

    Returns:
        A generator of (int, int) tuples that can slice up a list or a tensor.
    """
    for start in range(0, total_size, batch_size):
        stop = min(start + batch_size, total_size)
        yield start, stop
