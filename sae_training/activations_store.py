import beartype
import datasets
import torch
import tqdm
from jaxtyping import Float, jaxtyped
from torch import Tensor

from sae_training.config import Config
from sae_training.hooked_vit import HookedVisionTransformer


@jaxtyped(typechecker=beartype.beartype)
class ActivationsStore:
    """
    Class for streaming tokens and generating and storing activations while training SAEs.
    """

    def __init__(
        self,
        cfg: Config,
        model: HookedVisionTransformer,
    ):
        self.cfg = cfg
        self.model = model
        self.dataset = datasets.load_dataset(self.cfg.dataset_path, split="train")

        self.image_key = "image"
        self.label_key = "label"

        self.labels = self.dataset.features[self.label_key].names

        self.dataset = self.dataset.shuffle(seed=42)

        self.iterable_dataset = iter(self.dataset)

        print("Making dataloader.")
        self.dataloader = self.get_dataloader()
        print("Made dataloader.")

    @torch.no_grad()
    def get_image_batches(self):
        """
        A batch of tokens from a dataset.
        """
        batch_of_images = []
        for _ in tqdm.trange(
            self.cfg.store_size, desc="Filling activation store with images"
        ):
            try:
                batch_of_images.append(next(self.iterable_dataset)[self.image_key])
            except StopIteration:
                self.iterable_dataset = iter(self.dataset.shuffle())
                batch_of_images.append(next(self.iterable_dataset)[self.image_key])
        return batch_of_images

    def get_activations(self, image_batches: list) -> Float[Tensor, "batch d_model"]:
        module_name = self.cfg.module_name
        block_layer = self.cfg.block_layer
        list_of_hook_locations = [(block_layer, module_name)]

        inputs = self.model.processor(
            images=image_batches, text="", return_tensors="pt", padding=True
        ).to(self.cfg.device)

        activations = self.model.run_with_cache(
            list_of_hook_locations,
            **inputs,
        )[1][(block_layer, module_name)]

        # Only keep the class token
        # See the forward(), foward_head() methods of the VisionTransformer class in timm.
        # Eg "x = x[:, 0]  # class token" - the [:,0] indexes the batch dimension then the token dimension
        activations = activations[:, 0, :]
        return activations

    def get_sae_batches(self):
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
        return sae_batches

    def get_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Return a torch.utils.dataloader which you can get batches from.

        Should automatically refill the buffer when it gets to n % full.
        (better mixing if you refill and shuffle regularly).

        """
        batch_size = self.cfg.batch_size

        sae_batches = self.get_sae_batches()

        dataloader = torch.utils.data.DataLoader(
            sae_batches, batch_size=batch_size, shuffle=True
        )

        return iter(dataloader)

    def next_batch(self):
        """
        Get the next batch from the current DataLoader.
        If the DataLoader is exhausted, refill the buffer and create a new DataLoader.
        """
        try:
            # Try to get the next batch
            return next(self.dataloader)
        except StopIteration:
            # If the DataLoader is exhausted, create a new one
            self.dataloader = self.get_dataloader()
            return next(self.dataloader)
