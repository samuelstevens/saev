import os

import torch
from datasets import load_dataset
from torch import topk
from tqdm import trange

from sae_training.config import ViTSAERunnerConfig
from sae_training.hooked_vit import HookedVisionTransformer
from sae_training.utils import ViTSparseAutoencoderSessionloader


def get_model_activations(model, inputs, cfg):
    module_name = cfg.module_name
    block_layer = cfg.block_layer
    list_of_hook_locations = [(block_layer, module_name)]

    activations = model.run_with_cache(
        list_of_hook_locations,
        **inputs,
    )[1][(block_layer, module_name)]

    activations = activations[:, 0, :]

    return activations


def get_all_model_activations(model, images, cfg):
    max_batch_size = cfg.max_batch_size_for_vit_forward_pass
    number_of_mini_batches = len(images) // max_batch_size
    remainder = len(images) % max_batch_size
    sae_batches = []
    for mini_batch in trange(
        number_of_mini_batches, desc="Dashboard: forward pass images through ViT"
    ):
        image_batch = images[
            mini_batch * max_batch_size : (mini_batch + 1) * max_batch_size
        ]
        inputs = model.processor(
            images=image_batch, text="", return_tensors="pt", padding=True
        ).to(model.model.device)
        sae_batches.append(get_model_activations(model, inputs, cfg))

    if remainder > 0:
        image_batch = images[-remainder:]
        inputs = model.processor(
            images=image_batch, text="", return_tensors="pt", padding=True
        ).to(model.model.device)
        sae_batches.append(get_model_activations(model, inputs, cfg))

    sae_batches = torch.cat(sae_batches, dim=0)
    sae_batches = sae_batches.to(cfg.device)
    return sae_batches


def get_new_top_k(first_values, first_indices, second_values, second_indices, k):
    total_values = torch.cat([first_values, second_values], dim=1)
    total_indices = torch.cat([first_indices, second_indices], dim=1)
    new_values, indices_of_indices = topk(total_values, k=k, dim=1)
    new_indices = torch.gather(total_indices, 1, indices_of_indices)
    return new_values, new_indices


def save_highest_activating_images(
    max_activating_image_indices,
    max_activating_image_values,
    directory,
    dataset,
    image_key,
):
    assert (
        max_activating_image_values.size() == max_activating_image_indices.size()
    ), "size of max activating image indices doesn't match the size of max activing values."
    number_of_neurons, number_of_max_activating_examples = (
        max_activating_image_values.size()
    )
    for neuron in trange(number_of_neurons):
        if not os.path.exists(f"{directory}/{neuron}"):
            os.makedirs(f"{directory}/{neuron}")
        for max_activaitng_image in range(number_of_max_activating_examples):
            image = dataset[
                int(max_activating_image_indices[neuron, max_activaitng_image].item())
            ][image_key]
            image.save(
                f"{directory}/{neuron}/{max_activaitng_image}_{int(max_activating_image_indices[neuron, max_activaitng_image].item())}_{max_activating_image_values[neuron, max_activaitng_image].item():.4g}.png",
                "PNG",
            )


def get_mean(dataset, model, image_key, sae_config, mean_size=16_384):
    images = dataset[:mean_size][image_key]
    acts = get_all_model_activations(model, images, sae_config)
    mean_acts = torch.mean(acts, dim=0)
    return mean_acts


@torch.inference_mode()
def get_feature_data(
    sae_config: ViTSAERunnerConfig,
    model: HookedVisionTransformer,
    number_of_images: int = 32_768,
    number_of_max_activating_images: int = 15,
    max_number_of_images_per_iteration: int = 16_384,
    load_pretrained=False,
    seed=1,
):
    """
    Need to do the following:
    save index and activation level of directions.
    """
    torch.cuda.empty_cache()

    dataset = load_dataset(sae_config.dataset_path, split="train")
    dataset = dataset.shuffle(seed=seed)
    directory = "MLP"

    if sae_config.dataset_path == "cifar100":  # Need to put this in the cfg
        image_key = "img"
    else:
        image_key = "image"
    MLP_size = model.model.vision_model.config.intermediate_size
    MLP_out_weights = model.model.vision_model.encoder.layers[
        sae_config.block_layer
    ].mlp.fc2.weight.detach()  # size [resid_dimension, mlp_dim]
    MLP_out_weights /= torch.norm(MLP_out_weights, dim=0, keepdim=True)
    mean_acts = get_mean(dataset, model, image_key, sae_config)
    if load_pretrained:
        max_activating_image_indices = torch.load(
            f"{directory}/max_activating_image_indices.pt"
        )
        max_activating_image_values = torch.load(
            f"{directory}/max_activating_image_values.pt"
        )
    else:
        # Create two tensors, both of size [MLP, num_max_act]. Stores index and activation value (or projected component).
        max_activating_image_indices = torch.zeros(
            [MLP_size, number_of_max_activating_images]
        ).to(sae_config.device)
        max_activating_image_values = torch.zeros(
            [MLP_size, number_of_max_activating_images]
        ).to(sae_config.device)
        # Get tensor of size [batch, colour, width, height]?

        # Get tesnor of size [batch, reisd]?

        # Get tensor of size [batch, MLP]? NOT by passing through the MLP encoder but instead by multiplying by the normalised decoder matrix.

        # Create a second tensor with the index values in them

        # Concatinate with the 'global' tensors

        # Sort the tensors and then delete the extra dimensions

        # Job done

        number_of_images_processed = 0
        while number_of_images_processed < number_of_images:
            torch.cuda.empty_cache()
            try:
                images = dataset[
                    number_of_images_processed : number_of_images_processed
                    + max_number_of_images_per_iteration
                ][image_key]
            except StopIteration:
                print("All of the images in the dataset have been processed!")
                all_images_processed = True
                break

            model_activations = (
                get_all_model_activations(model, images, sae_config) - mean_acts
            )  # tensor of size [batch, d_resid], with mean removed
            MLP_basis_activations = torch.nn.functional.relu(
                model_activations @ MLP_out_weights
            ).transpose(0, 1)  # size [MLP_dim, batch]
            del model_activations

            values, indices = topk(
                MLP_basis_activations, k=number_of_max_activating_images, dim=1
            )
            indices += number_of_images_processed

            max_activating_image_values, max_activating_image_indices = get_new_top_k(
                max_activating_image_values,
                max_activating_image_indices,
                values,
                indices,
                number_of_max_activating_images,
            )

            number_of_images_processed += max_number_of_images_per_iteration

        # Check if the directory exists
        if not os.path.exists(directory):
            # Create the directory if it does not exist
            os.makedirs(directory)

        torch.save(
            max_activating_image_indices, f"{directory}/max_activating_image_indices.pt"
        )
        torch.save(
            max_activating_image_values, f"{directory}/max_activating_image_values.pt"
        )

    save_highest_activating_images(
        max_activating_image_indices[:1000],
        max_activating_image_values[:1000],
        directory,
        dataset,
        image_key,
    )


cfg = ViTSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    class_token=True,
    image_width=224,
    image_height=224,
    model_name="openai/clip-vit-large-patch14",
    module_name="resid",
    block_layer=-2,
    dataset_path="evanarlian/imagenet_1k_resized_256",
    use_cached_activations=False,
    cached_activations_path=None,
    d_in=1024,
    # SAE Parameters
    expansion_factor=64,
    b_dec_init_method="mean",
    # Training Parameters
    lr=0.0004,
    l1_coefficient=0.00008,
    lr_scheduler_name="constantwithwarmup",
    batch_size=1024,
    lr_warm_up_steps=500,
    total_training_tokens=2_097_152,
    n_batches_in_store=15,
    # Dead Neurons and Sparsity
    use_ghost_grads=True,
    feature_sampling_method=None,
    feature_sampling_window=100,
    dead_feature_window=5000,
    dead_feature_threshold=1e-6,
    # WANDB
    log_to_wandb=True,
    wandb_project="mats-hugo",
    wandb_entity=None,
    wandb_log_frequency=20,
    # Misc
    device="cuda",
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype=torch.float32,
)

loader = ViTSparseAutoencoderSessionloader(cfg)
model = loader.get_model(cfg.model_name)
model.to(cfg.device)

get_feature_data(
    cfg,
    model,
    load_pretrained=False,
)
