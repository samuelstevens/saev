import re

import wandb

# from sae_training.activation_store import ActivationStore
from sae_training.train_sae_on_vision_transformer import train_sae_on_vision_transformer
from sae_training.utils import ViTSparseAutoencoderSessionloader


def vision_transformer_sae_runner(cfg):
    if cfg.from_pretrained_path is not None:
        model, sparse_autoencoder, activations_loader = (
            ViTSparseAutoencoderSessionloader.load_session_from_pretrained(
                cfg.from_pretrained_path
            )
        )
        cfg = sparse_autoencoder.cfg
    else:
        loader = ViTSparseAutoencoderSessionloader(cfg)
        model, sparse_autoencoder, activations_loader = loader.load_session()

    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, config=cfg, name=cfg.run_name)

    # train SAE
    sparse_autoencoder = train_sae_on_vision_transformer(
        model,
        sparse_autoencoder,
        activations_loader,
    )

    # save sae to checkpoints folder
    path = f"{cfg.checkpoint_path}/final_{sparse_autoencoder.get_name()}.pt"
    sparse_autoencoder.save_model(path)

    # upload to wandb
    if cfg.log_to_wandb:
        model_artifact = wandb.Artifact(
            f"{re.sub(r'[^a-zA-Z0-9]', '', sparse_autoencoder.get_name())}",
            type="model",
            metadata=dict(cfg.__dict__),
        )
        model_artifact.add_file(path)
        wandb.log_artifact(model_artifact, aliases=["final_model"])

    if cfg.log_to_wandb:
        wandb.finish()

    return sparse_autoencoder, model
