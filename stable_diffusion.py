import os

import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel

# For video display:
from PIL import Image
from torch import autocast
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import ViTSparseAutoencoderSessionloader

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=True
)

# Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=True
)

# The noise scheduler
scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
)

# To the GPU we go!
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)

prompt = [""]
neuron_idx = 133
directory_path = f"stable_diffusion/{neuron_idx}"

# Make a folder to store results
os.makedirs(directory_path, exist_ok=True)

# Some settings
# prompt = ["a futuristic city, abandoned and overgrown with plants, dystopia, bathed in sunlight"]
height = 512  # default height of Stable Diffusion
width = 768  # default width of Stable Diffusion
num_inference_steps = 50  # Number of denoising steps
guidance_scale = 8.0  # Scale for classifier-free guidance
batch_size = 1
temperature = 0.01

sae_path = "checkpoints/pcy601zk/final_sparse_autoencoder_openai/clip-vit-large-patch14_-2_resid_65536.pt"

loaded_object = torch.load(sae_path)
cfg = loaded_object["cfg"]
state_dict = loaded_object["state_dict"]
sparse_autoencoder = SparseAutoencoder(cfg)
sparse_autoencoder.load_state_dict(state_dict)
sparse_autoencoder.eval()
loader = ViTSparseAutoencoderSessionloader(cfg)
model = loader.get_model(cfg.model_name)
model.to(cfg.device)

image = Image.open("dashboard/60133/max_activating/2_9.481.png")
inputs = model.processor(
    images=[image], text=[""], return_tensors="pt", padding=True
).to(torch_device)
output = model.model(**inputs).image_embeds[0]
image_vector = torch.matmul(
    output, torch.inverse(model.model.text_projection.weight.transpose(0, 1))
)
image_vector /= torch.norm(image_vector)


# Prep text
text_input = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

target_index = torch.argmax(text_input.input_ids.to(torch_device), dim=-1).item()

for i in range(5):
    text_embeddings[0, i + 1, :] = image_vector * torch.norm(
        text_embeddings[0, i + 1, :]
    )

max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])


for idx in range(10):
    scheduler.set_timesteps(num_inference_steps)
    generator = torch.manual_seed(idx)

    # Prep latents
    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.sigmas[0]  # Need to scale to match k

    # Loop
    with autocast("cuda"):
        for i, t in tqdm(enumerate(scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                )["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents)["prev_sample"]

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents

    with torch.no_grad():
        image = vae.decode(latents)

    # Display
    image = (image.sample / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(f"{directory_path}/{idx:04}.png")
