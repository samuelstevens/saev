import beartype
import torch
from transformer_lens import HookedTransformer

from sae_training.activations_store import ActivationsStore
from sae_training.config import Config
from sae_training.hooked_vit import HookedVisionTransformer
from sae_training.sparse_autoencoder import SparseAutoencoder


@beartype.beartype
class SessionLoader:
    """
    Responsible for loading all required
    artifacts and files for training
    a sparse autoencoder on a language model
    or analysing a pretraining autoencoder
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load_session(
        self,
    ) -> tuple[HookedVisionTransformer, SparseAutoencoder, ActivationsStore]:
        """
        Loads a session for training a sparse autoencoder on a language model.
        """

        model = self.get_model(self.cfg.model_name)
        model.to(self.cfg.device)  # May need to include a .to() method.
        activations_loader = self.get_activations_loader(self.cfg, model)
        sparse_autoencoder = self.initialize_sparse_autoencoder(self.cfg)

        return model, sparse_autoencoder, activations_loader

    @classmethod
    def load_session_from_pretrained(
        cls, path: str
    ) -> tuple[HookedTransformer, SparseAutoencoder, ActivationsStore]:
        """
        Loads a session for analysing a pretrained sparse autoencoder.
        """
        if torch.backends.mps.is_available():
            cfg = torch.load(path, map_location="mps")["cfg"]
            cfg.device = "mps"
        elif torch.cuda.is_available():
            cfg = torch.load(path, map_location="cuda")["cfg"]
        else:
            cfg = torch.load(path, map_location="cpu")["cfg"]

        model, _, activations_loader = cls(cfg).load_session()
        sparse_autoencoder = SparseAutoencoder.load_from_pretrained(path)

        return model, sparse_autoencoder, activations_loader

    def get_model(self, model_name: str):
        """
        Loads a model from transformer lens
        """

        # Todo: add check that model_name is valid

        model = HookedVisionTransformer(model_name)
        model.eval()

        return model

    def initialize_sparse_autoencoder(self, cfg):
        """
        Initializes a sparse autoencoder
        """

        sparse_autoencoder = SparseAutoencoder(cfg)

        return sparse_autoencoder

    def get_activations_loader(self, cfg: Config, model: HookedVisionTransformer):
        """
        Loads a DataLoaderBuffer for the activations of a language model.
        """

        activations_loader = ActivationsStore(cfg, model)

        return activations_loader
