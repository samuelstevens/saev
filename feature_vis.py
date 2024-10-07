import glob
import os
import shutil

import torch
from leap_ie.vision import engine
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor


def find_alive_idxs(directory_path="dashboard", threshold=10):
    subdirs_with_pngs = []  # List to store subdirectories meeting the criteria

    # Iterate through each subdirectory in the given directory path
    for root, dirs, files in os.walk(directory_path):
        for subdir in dirs:
            subdir_path = os.path.join(
                root, subdir + "/max_activating"
            )  # Construct path to subdirectory

            # Use glob to find all .png files within the subdirectory
            png_files = glob.glob(os.path.join(subdir_path, "*.png"))

            # Check if there are more than 6 .png files
            if len(png_files) >= threshold:
                # Add the subdirectory path to the list
                subdirs_with_pngs.append(int(subdir))

    return subdirs_with_pngs


class partial_model(nn.Module):
    def __init__(self, sae_config, sae):
        super().__init__()
        self.clip_model, self.processor = self.get_ViT(sae_config.model_name)
        self.sae_config = sae_config
        self.sae = sae

    def get_ViT(self, model_name):
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        return model, processor

    def forward(self, x):
        x = self.clip_model.vision_model.embeddings(x)
        x = self.clip_model.vision_model.pre_layrnorm(x)
        num_layers = len(self.clip_model.vision_model.encoder.layers)
        for idx, layer in enumerate(self.clip_model.vision_model.encoder.layers):
            if idx <= (self.sae_config.block_layer % num_layers):
                x = layer(x, None, None)[0]
        x = self.sae(x[:, 0, :])
        return x


class new_sae(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W_enc = nn.Parameter(
            torch.zeros((config.d_in, config.d_sae)), requires_grad=True
        )
        self.b_enc = nn.Parameter(torch.zeros((config.d_sae)), requires_grad=True)
        self.b_dec = nn.Parameter(torch.zeros((config.d_in)), requires_grad=True)

    def forward(self, x):
        x -= self.b_dec
        x = x @ self.W_enc.data
        x += self.b_enc
        # x = nn.functional.relu(x)
        return x

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


target_indices = [380, 886, 1773, 2297, 4482, 11059, 22604, 26958, 30387]

# 774, 10057, 25081, 14061, 2681
sae_path = "checkpoints/pcy601zk/final_sparse_autoencoder_openai/clip-vit-large-patch14_-2_resid_65536.pt"
loaded_object = torch.load(sae_path)
cfg = loaded_object["cfg"]
state_dict = loaded_object["state_dict"]
sparse_autoencoder = new_sae(cfg)
sparse_autoencoder.load_my_state_dict(state_dict)
sparse_autoencoder = sparse_autoencoder.to(cfg.device)
partial_model_instance = partial_model(cfg, sparse_autoencoder)
partial_model_instance = partial_model_instance.to(cfg.device)


max_steps = 3000
lr = [0.001, 0.005, 0.01, 0.05]
# LEAP config
for target_index in target_indices:
    print(f"Starting feature viz for neuron {target_index}!")
    for learning_rate in lr:
        config = {
            "leap_api_key": "LEAPIE908A240083F2956D8A4CF8B8C0689EB",
            "lr": 0.06,
            "leap_logging": False,
            "max_steps": max_steps,
            "use_blur": True,
        }

        res = engine.generate(
            project_name="MATS-cs_objective",
            model=partial_model_instance,
            class_list=[str(i) for i in range(cfg.d_sae)],
            preprocessing=[
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                )
            ],
            config=config,
            target_classes=[target_index],
            samples=None,
            device="cuda",
            mode="pt",
        )

        # Load images:
        current_file_path = f"leap_files/prototype_{max_steps}_0_{target_index}.png"
        images = [Image.open(f"leap_files/prototype_{max_steps}_0_{target_index}.png")]
        inputs = partial_model_instance.processor(
            images=images, return_tensors="pt", padding=True
        ).to(cfg.device)

        temperature = 1
        output = nn.functional.relu(partial_model_instance(inputs["pixel_values"]))
        post_relu_logits = output[0, target_index]
        print(f"post relu logits: {post_relu_logits}")
        probabilities = nn.functional.softmax(output / temperature, dim=-1)
        print(f"probability of target class: {probabilities[0,target_index]}")
        print(f"l1 value: {output.sum(dim = -1).detach()}")
        print(f"l0 value: {(output>0).sum(dim = -1).detach()}")
        target_file_path = f"dashboard/{target_index}/feature_vis/{learning_rate}_{post_relu_logits:.4g}.png"
        if post_relu_logits > 1:
            new_directory = os.path.dirname(target_file_path)

            # Check if the new directory exists, create it if it doesn't
            if not os.path.exists(new_directory):
                os.makedirs(new_directory)

            # Move the file
            shutil.move(current_file_path, target_file_path)
    directory_path = "leap_files"
    items = glob.glob(directory_path)

    for item in items:
        if os.path.isfile(item):
            os.remove(item)
        elif os.path.isdir(item):
            shutil.rmtree(item)
