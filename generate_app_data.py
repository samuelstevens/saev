import os
import pickle

import torch
import tyro
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from sae_training.sparse_autoencoder import SparseAutoencoder


def main(ckpt_path: str):
    directory = "dashboard"
    sparsity = torch.load(f"{directory}/sae_sparsity.pt").to("cpu")  # size [n]
    max_activating_image_indices = (
        torch.load(f"{directory}/max_activating_image_indices.pt")
        .to("cpu")
        .to(torch.int32)
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
    loaded_object = torch.load(ckpt_path)
    cfg = loaded_object["cfg"]
    sparse_autoencoder = SparseAutoencoder(cfg)
    sparse_autoencoder.load_state_dict(loaded_object["state_dict"])
    sparse_autoencoder.eval()
    dataset = load_dataset(cfg.dataset_path, split="train")
    dataset = dataset.shuffle(seed=1)

    number_of_neurons = max_activating_image_values.size()[0]
    entropy_list = torch.zeros(number_of_neurons)

    for i in range(number_of_neurons):
        # Get unique labels and their indices for the current sample
        unique_labels, _ = max_activating_image_label_indices[i].unique(
            return_inverse=True
        )
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
        meta_data = {
            "neuron index": index,
            "log 10 sparsity": torch.log10(sparsity)[index].item(),
            "mean activation": sae_mean_acts[index].item(),
            "label entropy": entropy_list[index].item(),
        }
        with open(f"{new_directory}/meta_data.pkl", "wb") as pickle_file:
            pickle.dump(meta_data, pickle_file)


if __name__ == "__main__":
    tyro.cli(main)
