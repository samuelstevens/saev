import collections.abc
import logging
import os

import beartype
import torch
import tqdm
import tyro
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

import saev

logger = logging.getLogger("analysis")


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


@jaxtyped(typechecker=beartype.beartype)
def get_vit_acts(
    acts_store: saev.ActivationsStore, n: int
) -> tuple[Float[Tensor, "n d_model"], Int[Tensor, " n"]]:
    """
    Args:

    Returns:
        activation tensor for all images.
    """
    n_seen = 0
    batches, indices = [], []
    while n_seen < n:
        batch, i = acts_store.next_indexed_batch()
        batches.append(batch)
        indices.append(i)
        n_seen += len(batch)
        logger.info("Got batch of size %d.", len(batch))

    batches = torch.cat(batches, dim=0)
    indices = torch.cat(indices, dim=0)
    return batches, indices


@jaxtyped(typechecker=beartype.beartype)
def get_sae_acts(
    vit_acts: Float[Tensor, "n d_model"], sae: saev.SparseAutoencoder
) -> Float[Tensor, "n d_sae"]:
    sae_acts = []
    for start, end in batched_idx(len(vit_acts), sae.cfg.vit_batch_size):
        _, cache = sae.run_with_cache(vit_acts[start:end])
        sae_acts.append(cache["hook_hidden_post"])

    sae_acts = torch.cat(sae_acts, dim=0)
    sae_acts = sae_acts.to(sae.cfg.device)
    return sae_acts


@jaxtyped(typechecker=beartype.beartype)
def get_new_topk(
    first_values: Float[Tensor, "d_sae k"],
    first_indices: Int[Tensor, "d_sae k"],
    second_values: Float[Tensor, "d_sae k"],
    second_indices: Int[Tensor, "d_sae k"],
    k: int,
) -> tuple[Float[Tensor, "d_sae k"], Int[Tensor, "d_sae k"]]:
    total_values = torch.cat([first_values, second_values], dim=1)
    total_indices = torch.cat([first_indices, second_indices], dim=1)
    new_values, indices_of_indices = torch.topk(total_values, k=k, dim=1)
    new_indices = torch.gather(total_indices, 1, indices_of_indices)
    return new_values, new_indices


@beartype.beartype
@torch.inference_mode()
def get_feature_data(
    sae: saev.SparseAutoencoder,
    vit: saev.HookedVisionTransformer,
    *,
    n_images: int = 32_768,
    k_top_images: int = 10,
    images_per_it: int = 16_384,
    directory: str = "data",
):
    """
    Args:
        sae: The sparse autoencoder to use
        vit: The vision transformer to apply the SAE to.
        n_images: How many images to look at.
        k_top_images: How many images per neuron (SAE feature) to keep.
        directory: Where to write this data.
    """
    torch.cuda.empty_cache()
    sae.eval()

    breakpoint()

    acts_store = saev.ActivationsStore(sae.cfg, vit)

    if n_images > len(acts_store.dataset):
        logger.warning(
            "The dataset '%s' only has %d images, but you requested %d images.",
            sae.cfg.dataset_path,
            len(acts_store.dataset),
            n_images,
        )
        n_images = len(acts_store.dataset)

    top_values = torch.zeros((sae.cfg.d_sae, k_top_images)).to(sae.cfg.device)
    top_indices = torch.zeros((sae.cfg.d_sae, k_top_images), dtype=torch.int)
    top_indices = top_indices.to(sae.cfg.device)

    sae_sparsity = torch.zeros((sae.cfg.d_sae,)).to(sae.cfg.device)
    sae_mean_acts = torch.zeros((sae.cfg.d_sae,)).to(sae.cfg.device)

    n_seen = 0

    while n_seen < n_images:
        torch.cuda.empty_cache()

        # tensor of size [batch, d_resid]
        vit_acts, indices = get_vit_acts(acts_store, images_per_it)
        # tensor of size [feature_idx, batch]
        sae_acts = get_sae_acts(vit_acts, sae).transpose(0, 1)
        del vit_acts
        sae_mean_acts += sae_acts.sum(dim=1)
        sae_sparsity += (sae_acts > 0).sum(dim=1)

        values, i = torch.topk(sae_acts, k=k_top_images, dim=1)
        # Convert i, a matrix of indices into this current batch, into indices, a matrix of indices into the global dataset.
        indices = indices.to(sae.cfg.device)[i.view(-1)].view(i.shape)

        top_values, top_indices = get_new_topk(
            top_values, top_indices, values, indices, k_top_images
        )

        n_seen += images_per_it
        logger.info("%d/%d (%.1f%%)", n_seen, n_images, n_seen / n_images * 100)

    sae_mean_acts /= sae_sparsity
    sae_sparsity /= n_images

    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory if it does not exist
        os.makedirs(directory)

    # compute the label tensor
    top_image_label_indices = torch.tensor([
        acts_store.dataset[int(index)]["label"]
        for index in tqdm.tqdm(top_indices.flatten(), desc="Getting labels")
    ])
    # Reshape to original dimensions
    top_image_label_indices = top_image_label_indices.view(top_indices.shape)
    torch.save(top_indices, f"{directory}/max_activating_image_indices.pt")
    torch.save(top_values, f"{directory}/max_activating_image_values.pt")
    torch.save(
        top_image_label_indices,
        f"{directory}/max_activating_image_label_indices.pt",
    )
    torch.save(sae_sparsity, f"{directory}/sae_sparsity.pt")
    torch.save(sae_mean_acts, f"{directory}/sae_mean_acts.pt")
    # Should also save label information tensor here!!!

    n_neurons, n_examples = top_values.shape
    for neuron in tqdm.trange(n_neurons):
        neuron_dead = True
        neuron_dir = os.path.join(directory, str(neuron))
        for i in range(n_examples):
            if top_values[neuron, i].item() <= 0:
                continue

            if neuron_dead:
                if not os.path.exists(neuron_dir):
                    os.makedirs(neuron_dir)
                neuron_dead = False

            index = top_indices[neuron, i].item()
            value = top_values[neuron, i].item()

            image = acts_store.dataset[index]["image"]
            image.save(f"{neuron_dir}/{i}_{index}_{value:.4g}.png", "PNG")


def main(
    ckpt_path: str,
    n_images: int = 524_288,
    k_top_images: int = 20,
    directory: str = "data",
):
    """
    Runs the primary function in this file: `get_feature_data()`.

    Args:
        ckpt_path: The SAE checkpoint to use.
        n_images: number of images to use. Use a smaller number for debugging.
        k_top_images: the number of top images to store per neuron.
    """
    vit, sae, _ = saev.utils.load_session(ckpt_path)
    get_feature_data(
        sae, vit, n_images=n_images, k_top_images=k_top_images, directory=directory
    )


if __name__ == "__main__":
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    tyro.cli(main)
