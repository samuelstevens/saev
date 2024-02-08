import os
import sys
import torch
import tqdm
import wandb
import json
import plotly.express as px
from transformer_lens import utils
from datasets import load_dataset
from typing import  Dict
from pathlib import Path

from functools import partial

sys.path.append("..")

from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_analysis.visualizer import data_fns, html_fns
from sae_analysis.visualizer.data_fns import get_feature_data, FeatureData
from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.lm_runner import language_model_sae_runner
from sae_training.train_sae_on_language_model import train_sae_on_language_model

if torch.backends.mps.is_available():
    device = "mps" 
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

def imshow(x, **kwargs):
    x_numpy = utils.to_numpy(x)
    px.imshow(x_numpy, **kwargs).show()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

cfg = LanguageModelSAERunnerConfig(

    # Data Generating Function (Model + Training Distibuion)
    model_name = "gpt2-small",
    hook_point = "blocks.10.hook_resid_pre",
    hook_point_layer = 10,
    d_in = 768,
    dataset_path = "Skylion007/openwebtext",
    is_dataset_tokenized=False,
    
    # SAE Parameters
    expansion_factor = 64,
    b_dec_init_method = "geometric_median",
    
    # Training Parameters
    lr = 0.0004,
    l1_coefficient = 0.00008,
    lr_scheduler_name="cosineannealingwarmup",
    train_batch_size = 4096,
    context_size = 128,
    lr_warm_up_steps=5000,
    
    # Activation Store Parameters
    n_batches_in_buffer = 128,
    total_training_tokens = 100_000,
    store_batch_size = 32,
    
    # Dead Neurons and Sparsity
    use_ghost_grads=True,
    feature_sampling_method = None,
    feature_sampling_window = 1000,
    dead_feature_window=5000,
    dead_feature_threshold = 1e-6,
    
    # WANDB
    log_to_wandb = True,
    wandb_project= "mats-hugo",
    wandb_entity = None,
    wandb_log_frequency=100,
    
    # Misc
    device = "cuda",
    seed = 42,
    n_checkpoints = 0,
    checkpoint_path = "checkpoints",
    dtype = torch.float32,
    )

sparse_autoencoder = language_model_sae_runner(cfg)

#Create an activation store with the correct database
session_loader = LMSparseAutoencoderSessionloader(cfg)
model = session_loader.get_model(cfg.model_name)
activations_store = session_loader.get_activations_loader(cfg, model)

sparse_autoencoder = train_sae_on_language_model(
    model,
    sparse_autoencoder,
    activations_store,
    batch_size = 1024,
    feature_sampling_window = 10,
    feature_sampling_method = None,
    use_wandb = True,
)

#Evaluate the SAE in terms of the models loss and compare to zero/mean ablation.
wandb.init(project='mats-hugo', entity='hugo-fry', job_type="inference")

def reconstr_hook(mlp_out, hook, new_mlp_out):
    return new_mlp_out

def zero_abl_hook(mlp_out, hook):
    return torch.zeros_like(mlp_out)

batch_tokens = activations_store.get_batch_tokens()
_, cache = model.run_with_cache(batch_tokens, prepend_bos=True)
_, cache = model.run_with_cache(batch_tokens, prepend_bos=True)
sae_out, feature_acts, loss, mse_loss, l1_loss = sparse_autoencoder(
    cache[sparse_autoencoder.cfg.hook_point]
    )
del cache

original_loss = model(batch_tokens, return_type="loss").item()
sae_reconstruction_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[
            (
                utils.get_act_name("resid_pre", 10),
                partial(reconstr_hook, new_mlp_out=sae_out),
            )
        ],
        return_type="loss",
    ).item(),

zero_ablation_loss = model.run_with_hooks(
        batch_tokens,
        return_type="loss",
        fwd_hooks=[(utils.get_act_name("resid_pre", 10), zero_abl_hook)],
    ).item()

wandb.log({'original loss':original_loss, 'sae loss':sae_reconstruction_loss, 'zero ablation loss':zero_ablation_loss})
wandb.finish()


vocab_dict = model.tokenizer.vocab
vocab_dict = {v: k.replace("Ġ", " ").replace("\n", "\\n") for k, v in vocab_dict.items()}

vocab_dict_filepath = Path(os.getcwd()) / "vocab_dict.json"
if not vocab_dict_filepath.exists():
    with open(vocab_dict_filepath, "w") as f:
        json.dump(vocab_dict, f)
        

os.environ["TOKENIZERS_PARALLELISM"] = "false"
data = activations_store.dataset
tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
tokenized_data = tokenized_data.shuffle(42)
all_tokens = tokenized_data["tokens"]


# Currently, don't think much more time can be squeezed out of it. Maybe the best saving would be to
# make the entire sequence indexing parallelized, but that's possibly not worth it right now.

max_batch_size = 512
total_batch_size = 4096*5
feature_idx = list(range(sparse_autoencoder.d_sae))
# max_batch_size = 512
# total_batch_size = 16384
# feature_idx = list(range(1000))

tokens = all_tokens[:total_batch_size]

feature_data: Dict[int, FeatureData] = get_feature_data(
    encoder=sparse_autoencoder,
    # encoder_B=sparse_autoencoder,
    model=model,
    hook_point=sparse_autoencoder.cfg.hook_point,
    hook_point_layer=sparse_autoencoder.cfg.hook_point_layer,
    tokens=tokens,
    feature_idx=feature_idx,
    max_batch_size=max_batch_size,
    left_hand_k = 3,
    buffer = (5, 5),
    n_groups = 10,
    first_group_size = 20,
    other_groups_size = 5,
    verbose = True,
)

pbar=tqdm(total = len(feature_idx))

for test_idx in feature_idx:
    html_str = feature_data[test_idx].get_all_html()
    with open(f"data_{test_idx:04}.html", "w") as f:
        f.write(html_str)
    pbar.update(1)
    
pbar.close()

for i in range(4):
    print()
print("*****Done*****")