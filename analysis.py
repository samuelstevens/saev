import collections.abc
import logging
import os

import beartype
import datasets
import torch
import tyro
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor, topk
from tqdm import tqdm, trange

from sae_training.config import Config
from sae_training.hooked_vit import HookedVisionTransformer
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import load_session

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


@beartype.beartype
def get_vit_acts(
    vit: HookedVisionTransformer, inputs: object, cfg: Config
) -> Float[Tensor, "batch d_model"]:
    """
    Args:
        vit: ViT that records activations
        inputs: a batch of inputs in the format expected by `vit`.
        cfg: SAE config.

    Returns:
        activations tensor for the batch.
    """
    hook_loc = (cfg.block_layer, cfg.module_name)
    _, cache = vit.run_with_cache([hook_loc], **inputs)
    acts = cache[hook_loc][:, 0, :]
    return acts


@jaxtyped(typechecker=beartype.beartype)
def get_all_vit_acts(
    vit: HookedVisionTransformer, images: list, cfg: Config
) -> Float[Tensor, "n d_model"]:
    """
    Args:
        vit: ViT that records activations.
        images: list of PIL images.
        cfg: SAE config.

    Returns:
        activation tensor for all images.
    """
    sae_batches = []
    for start, end in batched_idx(len(images), cfg.vit_batch_size):
        logger.info("Starting processing.")
        inputs = vit.processor(
            images=images[start:end], text="", return_tensors="pt", padding=True
        ).to(vit.model.device)
        logger.info("Finished processing.")
        sae_batches.append(get_vit_acts(vit, inputs, cfg))

    sae_batches = torch.cat(sae_batches, dim=0)
    sae_batches = sae_batches.to(cfg.device)
    return sae_batches


@jaxtyped(typechecker=beartype.beartype)
def get_sae_acts(
    vit_acts: Float[Tensor, "n d_model"], sae: SparseAutoencoder
) -> Float[Tensor, "n d_sae"]:
    sae_acts = []
    for start, end in batched_idx(len(vit_acts), sae.cfg.vit_batch_size):
        _, cache = sae.run_with_cache(vit_acts[start:end])
        sae_acts.append(cache["hook_hidden_post"])

    sae_acts = torch.cat(sae_acts, dim=0)
    sae_acts = sae_acts.to(sae.cfg.device)
    return sae_acts


@beartype.beartype
def save_highest_activating_images(
    max_activating_indices: Float[Tensor, "n k"],
    max_activating_values: Float[Tensor, "n k"],
    directory: str,
    dataset: object,
    key: str,
):
    n_neurons, n_examples = max_activating_values.shape
    for neuron in trange(n_neurons):
        neuron_dead = True
        neuron_dir = os.path.join(directory, str(neuron))
        for i in range(n_examples):
            if max_activating_values[neuron, i].item() <= 0:
                continue

            if neuron_dead:
                if not os.path.exists(neuron_dir):
                    os.makedirs(neuron_dir)
                neuron_dead = False

            index = int(max_activating_indices[neuron, i].item())
            value = max_activating_values[neuron, i].item()

            image = dataset[index][key]
            image.save(f"{neuron_dir}/{i}_{index}_{value:.4g}.png", "PNG")


@jaxtyped(typechecker=beartype.beartype)
def get_new_top_k(
    first_values: Float[Tensor, "d_sae k"],
    first_indices: Float[Tensor, "d_sae k"],
    second_values: Float[Tensor, "d_sae k"],
    second_indices: Int[Tensor, "d_sae k"],
    k: int,
) -> tuple[Float[Tensor, "d_sae k"], Float[Tensor, "d_sae k"]]:
    # TODO: change indices to dtype=torch.int.
    total_values = torch.cat([first_values, second_values], dim=1)
    total_indices = torch.cat([first_indices, second_indices], dim=1)
    new_values, indices_of_indices = topk(total_values, k=k, dim=1)
    new_indices = torch.gather(total_indices, 1, indices_of_indices)
    return new_values, new_indices


@beartype.beartype
@torch.inference_mode()
def get_feature_data(
    sae: SparseAutoencoder,
    vit: HookedVisionTransformer,
    n_images: int = 32_768,
    k_top_images: int = 10,
    images_per_it: int = 16_384,
    seed: int = 1,
    directory="dashboard",
):
    """ """
    torch.cuda.empty_cache()
    sae.eval()

    dataset = datasets.load_dataset(sae.cfg.dataset_path, split="train")

    if n_images > len(dataset):
        logger.warning(
            "The dataset '%s' only has %d images, but you requested %d images.",
            sae.cfg.dataset_path,
            len(dataset),
            n_images,
        )
        n_images = len(dataset)

    key = "image"

    dataset = dataset.shuffle(seed=seed)

    top_values = torch.zeros((sae.cfg.d_sae, k_top_images)).to(sae.cfg.device)
    top_indices = torch.zeros((sae.cfg.d_sae, k_top_images), dtype=torch.int)
    top_indices = top_indices.to(sae.cfg.device)

    sae_sparsity = torch.zeros((sae.cfg.d_sae,)).to(sae.cfg.device)
    sae_mean_acts = torch.zeros((sae.cfg.d_sae,)).to(sae.cfg.device)

    for start, end in batched_idx(n_images, images_per_it):
        torch.cuda.empty_cache()
        images = dataset[start:end][key]

        # tensor of size [batch, d_resid]
        vit_acts = get_all_vit_acts(vit, images, sae.cfg)
        # tensor of size [feature_idx, batch]
        sae_acts = get_sae_acts(vit_acts, sae).transpose(0, 1)
        del vit_acts
        sae_mean_acts += sae_acts.sum(dim=1)
        sae_sparsity += (sae_acts > 0).sum(dim=1)

        # Convert the images list to a torch tensor
        # sizes [sae_idx, images] is the size of this matrix correct?
        values, indices = topk(sae_acts, k=k_top_images, dim=1)
        indices += end - start

        top_values, top_indices = get_new_top_k(
            top_values, top_indices, values, indices, k_top_images
        )

    sae_mean_acts /= sae_sparsity
    sae_sparsity /= n_images

    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory if it does not exist
        os.makedirs(directory)

    # compute the label tensor
    top_image_label_indices = torch.tensor([
        dataset[int(index)]["label"]
        for index in tqdm(top_indices.flatten(), desc="getting image labels")
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

    save_highest_activating_images(
        top_indices[:1000, :10],
        top_values[:1000, :10],
        directory,
        dataset,
        key,
    )


def main(ckpt_path: str):
    vit, sae, _ = load_session(ckpt_path)
    get_feature_data(sae, vit, n_images=524_288, k_top_images=20)


if __name__ == "__main__":
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    tyro.cli(main)
