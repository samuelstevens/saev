import os

import torch
from datasets import load_dataset
from PIL import Image
from torch.nn import functional as F

from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import ViTSparseAutoencoderSessionloader

"""
Need to load the sae
Iterate through the dashboard files - get the neuron idx
save the sae encoder directions
map to the multimodal space - save that direction too
map text of the form 'a photo of a {label}'. left multiply by the projection matrix save this
map the sae directions to the label text space with probabilities.
"""


# returns something of size [1000, 768]
def get_imagenet_text_embeddings(model, processor, labels, cfg):
    text = [f"A photo of a {label}." for label in labels]
    inputs = processor(text=text, return_tensors="pt", padding=True).to(cfg.device)
    output = model.text_model(**inputs).pooler_output
    embeddings = torch.matmul(
        output, model.text_projection.weight.transpose(0, 1)
    )  # size [1000, 768]
    embeddings /= torch.norm(embeddings, dim=-1, keepdim=True)
    return embeddings


def find_neuron_idxs(directory_path="dashboard"):
    subdirs = []  # List to store subdirectories meeting the criteria
    # Iterate through each subdirectory in the given directory path
    entries = os.listdir(directory_path)
    # Filter out the subdirectories
    subdirectories = [
        int(entry)
        for entry in entries
        if os.path.isdir(os.path.join(directory_path, entry))
    ]
    return subdirectories


neuron_idxs = find_neuron_idxs()
sae_path = "checkpoints/pcy601zk/final_sparse_autoencoder_openai/clip-vit-large-patch14_-2_resid_65536.pt"
temperature = 0.015
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
if cfg.dataset_path == "cifar100":  # Need to put this in the cfg
    label_key = "fine_label"
else:
    label_key = "label"
labels = dataset.features[label_key].names

text_embeddings = get_imagenet_text_embeddings(
    model.model, model.processor, labels, cfg
)
sae_directions = 13 * sparse_autoencoder.W_dec.data.clone()  # d_sae, d_in
sae_directions += sparse_autoencoder.b_dec.data.clone().unsqueeze(dim=0)

sae_directions = model.model.visual_projection(
    model.model.vision_model.post_layernorm(sae_directions)
)
sae_directions /= torch.norm(sae_directions, dim=-1, keepdim=True)


inner_products = torch.matmul(sae_directions, text_embeddings.transpose(0, 1))
all_probabilities = F.softmax(inner_products / temperature, dim=-1)
_, indices = torch.topk(all_probabilities, k=10, dim=-1)

for index, neuron_idx in enumerate(neuron_idxs):
    text = [
        f"A photo of a {labels[label_index]}.\nProbability: {all_probabilities[index, label_index]* 100:.4g}% \n\n"
        for label_index in indices[index]
    ]

    # Open a file for writing. Creates a new file if it doesn't exist or truncates the file if it exists.
    with open(f"dashboard/{neuron_idx}/text_explanation.txt", "w") as file:
        for line in text:
            file.write(line)


sparse_autoencoder = sparse_autoencoder.eval()
vision_model = model.model.vision_model
activation_cache = []


def hook_fn(module, module_input, module_output):
    activation_cache.append(module_output[0][:, 0, :])


handle = vision_model.encoder.layers[-2].register_forward_hook(hook_fn)

images = [Image.open("dashboard/133/max_activating/0_0.1902.png")]
inputs = model.processor(
    images=images, text=[""], return_tensors="pt", padding=True
).to(cfg.device)
vision_model(inputs["pixel_values"])
handle.remove()
image_activation = activation_cache[0]
sae_out = sparse_autoencoder(image_activation)[0]
image_activation = image_activation.squeeze()
image_activation = sae_out.squeeze()
image_embedding = model.model.visual_projection(
    model.model.vision_model.post_layernorm(image_activation)
)
image_embedding /= torch.norm(image_embedding)

inner_products = torch.matmul(image_embedding, text_embeddings.transpose(0, 1))
all_probabilities = F.softmax(inner_products / temperature, dim=-1)
_, indices = torch.topk(all_probabilities, k=10)


text = [
    f"A photo of a {labels[label_index]}.\nProbability: {all_probabilities[label_index]* 100:.4g}% \n\n"
    for label_index in indices
]
for sentence in text:
    print(sentence)


# print(sae_out)
# print(f'SAE encoder size: {torch.norm(sparse_autoencoder.W_enc.data[:,133])}')
# print(f'SAE bias: {sparse_autoencoder.b_enc.data[133]}')
# print(f'Decoder bias size: {torch.norm(sparse_autoencoder.b_dec).item()}')
# print(f'Image activation size: {torch.norm(image_activation).item()}')
# image_activation -= sparse_autoencoder.b_dec.clone()
# print(f'Image activation minus decoder bias size: {torch.norm(image_activation).item()}')
# sae_activation = sparse_autoencoder.W_enc.data[:,133].clone()
# activation_value = torch.dot(sae_activation, image_activation) + sparse_autoencoder.b_enc.data[133].clone()
# print(f'Reconstructed activation value: {activation_value}')
# image_activation /= torch.norm(image_activation)
# sae_activation /= torch.norm(sae_activation)
# cos_similarity = torch.dot(sae_activation,image_activation).item()
# print(f'Cos similarity between image activation minus decoder bias and sae encoder: {cos_similarity}')
