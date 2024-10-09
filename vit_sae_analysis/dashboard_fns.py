import os

import beartype
import datasets
import torch
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor, topk
from tqdm import tqdm, trange

from sae_training.config import Config
from sae_training.hooked_vit import HookedVisionTransformer
from sae_training.sparse_autoencoder import SparseAutoencoder


@beartype.beartype
def get_vit_acts(
    vit: HookedVisionTransformer,
    inputs: object,
    cfg: Config,
) -> Float[Tensor, "batch d_model"]:
    """
    Args:
        vit: ViT that records activations
        inputs: a batch of inputs in the format expected by `vit`.
        cfg: SAE config.

    Returns:
        activations tensor for the batch.
    """
    module_name = cfg.module_name
    block_layer = cfg.block_layer
    list_of_hook_locations = [(block_layer, module_name)]

    activations = vit.run_with_cache(
        list_of_hook_locations,
        **inputs,
    )[1][(block_layer, module_name)]

    activations = activations[:, 0, :]
    return activations


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
    batch_size = cfg.vit_batch_size
    n_batches, remainder = len(images) // batch_size, len(images) % batch_size
    sae_batches = []
    for b in trange(n_batches, desc="Getting ViT activations"):
        image_batch = images[b * batch_size : (b + 1) * batch_size]
        inputs = vit.processor(
            images=image_batch, text="", return_tensors="pt", padding=True
        ).to(vit.model.device)
        sae_batches.append(get_vit_acts(vit, inputs, cfg))

    if remainder > 0:
        image_batch = images[-remainder:]
        inputs = vit.processor(
            images=image_batch, text="", return_tensors="pt", padding=True
        ).to(vit.model.device)
        sae_batches.append(get_vit_acts(vit, inputs, cfg))

    sae_batches = torch.cat(sae_batches, dim=0)
    sae_batches = sae_batches.to(cfg.device)
    return sae_batches


@jaxtyped(typechecker=beartype.beartype)
def get_sae_activations(
    vit_acts: Float[Tensor, "n d_model"], sae: SparseAutoencoder
) -> Float[Tensor, "n d_sae"]:
    hook_name = "hook_hidden_post"
    batch_size = sae.cfg.vit_batch_size  # Use this for the SAE too
    n_batches, remainder = len(vit_acts) // batch_size, len(vit_acts) % batch_size
    sae_activations = []
    for b in trange(n_batches, desc="Getting SAE activations"):
        sae_activations.append(
            sae.run_with_cache(vit_acts[b * batch_size : (b + 1) * batch_size])[1][
                hook_name
            ]
        )

    if remainder > 0:
        sae_activations.append(sae.run_with_cache(vit_acts[-remainder:])[1][hook_name])

    sae_activations = torch.cat(sae_activations, dim=0)
    sae_activations = sae_activations.to(sae.cfg.device)
    return sae_activations


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
    load_pretrained=False,
):
    """ """
    torch.cuda.empty_cache()
    sae.eval()

    dataset = datasets.load_dataset(sae.cfg.dataset_path, split="train")
    key = "image"

    dataset = dataset.shuffle(seed=seed)
    directory = "dashboard"

    top_image_indices = torch.zeros((sae.cfg.d_sae, k_top_images)).to(sae.cfg.device)
    top_image_values = torch.zeros((sae.cfg.d_sae, k_top_images)).to(sae.cfg.device)
    sae_sparsity = torch.zeros((sae.cfg.d_sae,)).to(sae.cfg.device)
    sae_mean_acts = torch.zeros((sae.cfg.d_sae,)).to(sae.cfg.device)
    n_images_seen = 0
    while n_images_seen < n_images:
        torch.cuda.empty_cache()
        try:
            images = dataset[n_images_seen : n_images_seen + images_per_it][key]
        except StopIteration:
            print("All of the images in the dataset have been processed!")
            break

        # tensor of size [batch, d_resid]
        vit_acts = get_all_vit_acts(vit, images, sae.cfg)
        sae_activations = get_sae_activations(vit_acts, sae).transpose(
            0, 1
        )  # tensor of size [feature_idx, batch]
        del vit_acts
        sae_mean_acts += sae_activations.sum(dim=1)
        sae_sparsity += (sae_activations > 0).sum(dim=1)

        # Convert the images list to a torch tensor
        values, indices = topk(
            sae_activations, k=k_top_images, dim=1
        )  # sizes [sae_idx, images] is the size of this matrix correct?
        indices += n_images_seen

        top_image_values, top_image_indices = get_new_top_k(
            top_image_values,
            top_image_indices,
            values,
            indices,
            k_top_images,
        )

        n_images_seen += images_per_it

    sae_mean_acts /= sae_sparsity
    sae_sparsity /= n_images_seen

    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory if it does not exist
        os.makedirs(directory)

    # compute the label tensor
    top_image_label_indices = torch.tensor([
        dataset[int(index)]["label"]
        for index in tqdm(top_image_indices.flatten(), desc="getting image labels")
    ])
    # Reshape to original dimensions
    top_image_label_indices = top_image_label_indices.view(top_image_indices.shape)
    torch.save(top_image_indices, f"{directory}/max_activating_image_indices.pt")
    torch.save(top_image_values, f"{directory}/max_activating_image_values.pt")
    torch.save(
        top_image_label_indices,
        f"{directory}/max_activating_image_label_indices.pt",
    )
    torch.save(sae_sparsity, f"{directory}/sae_sparsity.pt")
    torch.save(sae_mean_acts, f"{directory}/sae_mean_acts.pt")
    # Should also save label information tensor here!!!

    save_highest_activating_images(
        top_image_indices[:1000, :10],
        top_image_values[:1000, :10],
        directory,
        dataset,
        key,
    )
