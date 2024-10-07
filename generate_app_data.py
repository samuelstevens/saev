import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image, ImageFilter
from tqdm import tqdm

from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import ViTSparseAutoencoderSessionloader

save_neurons = True
save_images = True

expansion_factor = 64
directory = f"expansion {expansion_factor}"  # "dashboard"
sparsity = torch.load(f"{directory}/sae_sparsity.pt").to("cpu")  # size [n]
max_activating_image_indices = (
    torch.load(f"{directory}/max_activating_image_indices.pt").to("cpu").to(torch.int32)
)
max_activating_image_values = torch.load(
    f"{directory}/max_activating_image_values.pt"
).to("cpu")  # size [n, num_max_act]
max_activating_image_label_indices = (
    torch.load(f"{directory}/max_activating_image_label_indices.pt")
    .to("cpu")
    .to(torch.int32)
)  # size [n, num_max_act]
sae_mean_acts = max_activating_image_values.mean(dim=-1)
sae_path = f"checkpoints/{expansion_factor}_expansion/final_sparse_autoencoder_openai/clip-vit-large-patch14_-2_resid_{expansion_factor*1024}.pt"
loaded_object = torch.load(sae_path)
cfg = loaded_object["cfg"]
state_dict = loaded_object["state_dict"]
sparse_autoencoder = SparseAutoencoder(cfg)
sparse_autoencoder.load_state_dict(state_dict)
sparse_autoencoder.eval()
loader = ViTSparseAutoencoderSessionloader(cfg)
model = loader.get_model(cfg.model_name)
model.to(cfg.device)
dataset = load_dataset(cfg.dataset_path, split="train")
dataset = dataset.shuffle(seed=1)
encoder_weights = (
    sparse_autoencoder.W_enc.clone().detach().transpose(0, 1)
)  # [d_sae, resid]
encoder_weights /= torch.norm(encoder_weights, dim=0, keepdim=True)
mlp_out_weights = (
    model.model.vision_model.encoder.layers[cfg.block_layer]
    .mlp.fc2.weight.clone()
    .detach()
    .transpose(0, 1)
)  # size [hidden_mlp_dimemsion, resid_dimension]
mlp_out_weights /= torch.norm(mlp_out_weights, dim=1, keepdim=True)
cosine_similarities = encoder_weights @ mlp_out_weights.transpose(
    0, 1
)  # size [d_sae, hidden_mlp]
cosine_similarities = cosine_similarities.to("cpu")

number_of_neurons = max_activating_image_values.size()[0]
entropy_list = torch.zeros(number_of_neurons)

for i in range(number_of_neurons):
    # Get unique labels and their indices for the current sample
    unique_labels, _ = max_activating_image_label_indices[i].unique(return_inverse=True)
    unique_labels = unique_labels[
        unique_labels != 949
    ]  # ignore label 949 = dataset[0]['label'] - the default label index
    if len(unique_labels) != 0:
        counts = 0
        for label in unique_labels:
            counts += (max_activating_image_label_indices[i] == label).sum()
        if counts < 10:
            entropy_list[i] = -1  # discount as too few datapoints!
        else:
            # Sum probabilities based on these labels
            summed_probs = torch.zeros_like(
                unique_labels, dtype=max_activating_image_values.dtype
            )
            for j, label in enumerate(unique_labels):
                summed_probs[j] = (
                    max_activating_image_values[i][
                        max_activating_image_label_indices[i] == label
                    ]
                    .sum()
                    .item()
                )
            # Calculate entropy for the summed probabilities
            summed_probs = (
                summed_probs / summed_probs.sum()
            )  # Normalize to make it a valid probability distribution
            entropy = -torch.sum(
                summed_probs * torch.log(summed_probs + 1e-9)
            )  # small epsilon to avoid log(0)
            entropy_list[i] = entropy
    else:
        entropy_list[i] = -1

# Mask all neurons in the dense cluster
mask = (
    (torch.log10(sparsity) > -4)
    & (torch.log10(sae_mean_acts) > -0.7)
    & (entropy_list > -1)
)
indices = torch.tensor([i for i in range(number_of_neurons)])
indices = list(indices[mask])


def save_highest_activating_images(neuron_index, neuron_directory):
    image_indices = max_activating_image_indices[neuron_index][:16]
    images = []
    for image_index in image_indices:
        images.append(dataset[int(image_index)]["image"])
    # Resize images and ensure they are in RGB
    resized_images = [img.resize((224, 224)).convert("RGB") for img in images]

    # Create an image grid
    grid_size = 4
    image_width, image_height = 224, 224
    border_size = 2  # White border thickness

    # Create a new image with white background
    total_width = grid_size * image_width + (grid_size - 1) * border_size
    total_height = grid_size * image_height + (grid_size - 1) * border_size
    new_im = Image.new("RGB", (total_width, total_height), "white")

    # Paste images in the grid
    x_offset, y_offset = 0, 0
    for i, img in enumerate(resized_images):
        new_im.paste(img, (x_offset, y_offset))
        x_offset += image_width + border_size
        if (i + 1) % grid_size == 0:
            x_offset = 0
            y_offset += image_height + border_size

    # Save the new image
    new_im.save(f"{neuron_directory}/highest_activating_images.png")


def save_MLP_cosine_similarity(neuron_index, neuron_directory):
    new_cosine_similarities = cosine_similarities[neuron_index].clone()
    df = pd.DataFrame(
        {
            "X": range(len(new_cosine_similarities)),
            "Y": new_cosine_similarities.numpy(),  # Convert tensor to numpy array
        }
    )
    df.to_feather(f"{neuron_directory}/MLP.feather")

    # fig = px.line(df, x='X', y='Y', labels={
    #         'X': 'MLP index',  # Custom x-axis label
    #         'Y': 'Cosine similarity'  # Custom y-axis label
    #     })
    # fig.update_layout(
    #     yaxis=dict(range=[-0.3, 0.6])  # Set the y-axis range
    # )


def save_activations_and_neurons(image, image_directory):
    module_name = cfg.module_name
    block_layer = cfg.block_layer
    list_of_hook_locations = [(block_layer, module_name)]
    inputs = model.processor(
        images=[image], text="", return_tensors="pt", padding=True
    ).to(cfg.device)
    model_activations = model.run_with_cache(
        list_of_hook_locations,
        **inputs,
    )[1][(block_layer, module_name)]
    model_activations = model_activations[:, 0, :]
    _, feature_acts, _, _, _, _ = sparse_autoencoder(model_activations)
    feature_acts = feature_acts.to("cpu")
    feature_acts = feature_acts[0].detach()
    _, sae_indices = torch.topk(feature_acts, 5)
    if (torch.log10(sparsity)[sae_indices] > -1.16).sum() > 0:
        raise Exception("Image is invalid!")
    df = pd.DataFrame(
        {
            "X": range(len(feature_acts)),
            "Y": feature_acts.numpy(),  # Convert tensor to numpy array
        }
    )
    df.to_feather(f"{image_directory}/activations.feather")

    # fig = px.line(df, x='X', y='Y', labels={
    #         'X': 'SAE index',  # Custom x-axis label
    #         'Y': 'Activation value'  # Custom y-axis label
    #     })

    sae_indices = sae_indices.squeeze().tolist()
    with open(f"{image_directory}/top_five_indices.json", "w") as json_file:
        json.dump(sae_indices, json_file)
    for sae_index in sae_indices:
        if not os.path.isdir(f"web_app/neurons/{sae_index}"):
            raise Exception("This sae feature has not yet been saved!")


def is_valid_image(image):
    module_name = cfg.module_name
    block_layer = cfg.block_layer
    list_of_hook_locations = [(block_layer, module_name)]
    inputs = model.processor(
        images=[image], text="", return_tensors="pt", padding=True
    ).to(cfg.device)
    model_activations = model.run_with_cache(
        list_of_hook_locations,
        **inputs,
    )[1][(block_layer, module_name)]
    model_activations = model_activations[:, 0, :]
    _, feature_acts, _, _, _, _ = sparse_autoencoder(model_activations)
    feature_acts = feature_acts.to("cpu")
    feature_acts = feature_acts[0].detach()
    _, sae_indices = torch.topk(feature_acts, 5)
    if (torch.log10(sparsity)[sae_indices] > -1.16).sum() > 0:
        return False
    return True


if save_neurons:
    new_directory = "web_app/neurons"
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    torch.save(entropy_list, "web_app/neurons/entropy.pt")
    for index in tqdm(indices, desc="saving highest activating grids"):
        index = int(index.item())
        new_directory = f"web_app/neurons/{index}"
        external_directory = f"saeexplorer/neurons/{index}"
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
        if not os.path.exists(external_directory):
            os.makedirs(external_directory)
        save_highest_activating_images(index, external_directory)
        save_MLP_cosine_similarity(index, new_directory)
        meta_data = {
            "neuron index": index,
            "log 10 sparsity": torch.log10(sparsity)[index].item(),
            "mean activation": sae_mean_acts[index].item(),
            "label entropy": entropy_list[index].item(),
        }
        with open(f"{new_directory}/meta_data.pkl", "wb") as pickle_file:
            pickle.dump(meta_data, pickle_file)

if save_images:
    indices = np.random.permutation(np.arange(500000, 1000001)).tolist()
    num_images = 0
    for i in tqdm(indices, desc="saving images for web app"):
        image = dataset[i]["image"]
        if is_valid_image(image):
            num_images += 1
            new_directory = f"web_app/images/{i}"
            if not os.path.exists(new_directory):
                os.makedirs(new_directory)
            external_directory = f"saeexplorer/images/{i}"
            if not os.path.exists(external_directory):
                os.makedirs(external_directory)
            save_activations_and_neurons(image, new_directory)
            image = image.resize((224, 224)).convert("RGB")
            image.save(f"{external_directory}/image.png")
            image = image.filter(ImageFilter.GaussianBlur(radius=40))
            image.save(f"{external_directory}/blurred_image.png")
            if num_images > 1000:
                break
