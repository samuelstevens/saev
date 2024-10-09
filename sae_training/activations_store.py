import beartype
import datasets
import torch
import tqdm
from jaxtyping import Float, jaxtyped
from torch import Tensor
import PIL.Image

from sae_training.config import Config
from sae_training.hooked_vit import HookedVisionTransformer


@jaxtyped(typechecker=beartype.beartype)
class ActivationsStore:
    """
    Class for streaming tokens and generating and storing activations while training SAEs.
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
        self.batch_size = cfg.batch_size
        self.dataloader_it = self.get_dataloader_it()

    @torch.no_grad()
    def get_image_batches(self) -> list[PIL.Image.Image]:
        """
        A batch of tokens from a dataset.
        """
        batch_of_images = []
        for _ in tqdm.trange(
            self.cfg.store_size, desc="Filling activation store with images"
        ):
            try:
                batch_of_images.append(next(self.dataset_it)["image"])
            except StopIteration:
                self.dataset_it = iter(self.dataset.shuffle())
                batch_of_images.append(next(self.dataset_it)["image"])
        return batch_of_images

    def get_activations(
        self, image_batches: list[PIL.Image.Image]
    ) -> Float[Tensor, "batch d_model"]:
        inputs = self.vit.processor(
            images=image_batches, text="", return_tensors="pt", padding=True
        ).to(self.cfg.device)

        activations = self.vit.run_with_cache([self.hook_loc], **inputs)[1][
            self.hook_loc
        ]

        # Only keep the class token
        # See the forward(), foward_head() methods of the VisionTransformer class in timm.
        # Eg "x = x[:, 0]  # class token" - the [:,0] indexes the batch dimension then the token dimension
        activations = activations[:, 0, :]
        return activations

    def get_sae_batches(self) -> Float[Tensor, "..."]:
        image_batches = self.get_image_batches()
        batch_size = self.cfg.vit_batch_size
        n_batches = len(image_batches) // batch_size
        remainder = len(image_batches) % batch_size
        sae_batches = []
        for batch in tqdm.trange(n_batches, desc="Getting batches for SAE"):
            activations = self.get_activations(
                image_batches[batch * batch_size : (batch + 1) * batch_size]
            )
            sae_batches.append(activations)

        if remainder > 0:
            sae_batches.append(self.get_activations(image_batches[-remainder:]))

        sae_batches = torch.cat(sae_batches, dim=0)
        sae_batches = sae_batches.to(self.cfg.device)
        breakpoint()
        return sae_batches

    def get_dataloader_it(self):
        """
        Return a torch.utils.dataloader which you can get batches from.

        Should automatically refill the buffer when it gets to n % full.
        (better mixing if you refill and shuffle regularly).

        """

        dataloader = torch.utils.data.DataLoader(
            self.get_sae_batches(), batch_size=self.batch_size, shuffle=True
        )

        return iter(dataloader)

    def next_batch(self):
        """
        Get the next batch from the current DataLoader.
        If the DataLoader is exhausted, refill the buffer and create a new DataLoader.
        """
        try:
            # Try to get the next batch
            return next(self.dataloader_it)
        except StopIteration:
            # If the DataLoader is exhausted, create a new one
            self.dataloader_it = self.get_dataloader_it()
            return next(self.dataloader_it)
