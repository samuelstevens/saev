import sys

import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from tqdm import trange

sys.path.append("..")


from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import ViTSparseAutoencoderSessionloader
from vit_sae_analysis.dashboard_fns import (
    get_all_model_activations,
    get_sae_activations,
)


def convert_images_to_tensor(images, device="cuda"):
    """
    Convert a list of PIL images to a PyTorch tensor in RGB format with shape [B, C, H, W].

    Parameters:
    - images: List of PIL.Image objects.
    - device: The device to store the tensor on ('cpu' or 'cuda').

    Returns:
    - A PyTorch tensor with shape [B, C, H, W].
    """
    # Define a transform to convert PIL images (in RGB) to tensors
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),  # Convert image to RGB
            transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
            transforms.ToTensor(),  # Convert the image to a torch tensor
        ]
    )

    # Ensure each image is in RGB format, apply the transform, and move to the specified device
    tensor_list = [transform(img).to(device) for img in images]
    tensor_output = torch.stack(tensor_list, dim=0)

    return tensor_output


def get_model_and_sae(sae_path):
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loaded_object = torch.load(sae_path)

    cfg = loaded_object["cfg"]

    state_dict = loaded_object["state_dict"]

    sparse_autoencoder = SparseAutoencoder(cfg)

    sparse_autoencoder.load_state_dict(state_dict)

    sparse_autoencoder.eval()

    loader = ViTSparseAutoencoderSessionloader(cfg)

    model = loader.get_model(cfg.model_name)
    model.to(cfg.device)

    torch.cuda.empty_cache()
    sparse_autoencoder.eval()
    return model, sparse_autoencoder


def get_dataset(sparse_autoencoder):
    dataset = load_dataset(sparse_autoencoder.cfg.dataset_path, split="train")
    dataset = dataset.shuffle()
    iterable_dataset = iter(dataset)
    return dataset, iterable_dataset


def load_random_images_and_activations(sae_path, num_images):
    model, sparse_autoencoder = get_model_and_sae(sae_path)

    if sparse_autoencoder.cfg.dataset_path == "cifar100":
        image_key = "img"
    else:
        image_key = "image"

    dataset, iterable_dataset = get_dataset(sparse_autoencoder)

    images = []
    for image in trange(num_images, desc="Getting images for dashboard"):
        with torch.no_grad():
            try:
                images.append(next(iterable_dataset)[image_key])
            except StopIteration:
                iterable_dataset = iter(dataset.shuffle())
                images.append(next(iterable_dataset)[image_key])

    model_activations = get_all_model_activations(
        model, images, sparse_autoencoder.cfg
    )  # tensor of size [batch, d_resid]
    sae_activations = get_sae_activations(
        model_activations,
        sparse_autoencoder,
        torch.tensor(range(sparse_autoencoder.cfg.d_sae)),
    )  # tensor of size [batch, feature_idx]
    del model_activations

    images = convert_images_to_tensor(images)

    return (images, sae_activations)
