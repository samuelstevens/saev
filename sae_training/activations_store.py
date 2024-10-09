import beartype
import datasets
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from sae_training.config import Config
from sae_training.hooked_vit import HookedVisionTransformer


@jaxtyped(typechecker=beartype.beartype)
class ActivationsStore:
    """
    Class for streaming tokens and generating and storing activations while training SAEs.

    Uses a multiprocess torch DataLoader to preprocess images (resize, crop, etc) faster, but still uses the main process to convert images (as tensors) into ViT activations.
    """

    cfg: Config
    vit: HookedVisionTransformer
    hook_loc: tuple[int, str]
    batch_size: int

    def __init__(
        self,
        cfg: Config,
        vit: HookedVisionTransformer,
    ):
        self.cfg = cfg
        self.vit = vit
        self.dataset = datasets.load_dataset(self.cfg.dataset_path, split="train")

        self.labels = self.dataset.features["label"].names
        self.dataset = self.dataset.shuffle(cfg.seed)

        self.dataset_it = iter(self.dataset)
        self.hook_loc = (cfg.block_layer, cfg.module_name)
        assert cfg.vit_batch_size == cfg.batch_size
        self.batch_size = cfg.batch_size

    def get_sae_batches(self) -> Float[Tensor, "store_size d_model"]:
        n_examples = 0
        examples = []
        while n_examples < self.cfg.store_size:
            batch = self.next_batch()
            examples.append(batch)
            n_examples += len(batch)

        examples = torch.cat(examples, dim=0)
        examples = examples[: self.cfg.store_size]
        examples = examples.to(self.cfg.device)
        return examples

    @torch.inference_mode
    def next_batch(self) -> Float[Tensor, "batch d_model"]:
        # 1. Load images from disk.
        images = []
        for _ in range(self.batch_size):
            try:
                images.append(next(self.dataset_it)["image"])
            except StopIteration:
                self.dataset_it = iter(self.dataset.shuffle())
                images.append(next(self.dataset_it)["image"])

        # 2. Preprocess images to tensors.
        inputs = self.vit.processor(
            images=images, text="", return_tensors="pt", padding=True
        ).to(self.cfg.device)

        # 3. Get ViT activations.
        activations = self.vit.run_with_cache([self.hook_loc], **inputs)[1][
            self.hook_loc
        ]

        # Only keep the class token.
        activations = activations[:, 0, :]
        # Move to device
        activations = activations.to(self.cfg.device)
        return activations
