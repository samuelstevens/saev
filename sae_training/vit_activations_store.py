import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import trange

from sae_training.config import ViTSAERunnerConfig
from sae_training.hooked_vit import HookedVisionTransformer


class ViTActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs.
    """

    def __init__(
        self,
        cfg: ViTSAERunnerConfig,
        model: HookedVisionTransformer,
        create_dataloader: bool = True,
        train=True,
    ):
        self.cfg = cfg
        self.model = model
        self.dataset = load_dataset(self.cfg.dataset_path, split="train")

        if self.cfg.dataset_path == "cifar100":  # Need to put this in the cfg
            self.image_key = "img"
            self.label_key = "fine_label"
        else:
            self.image_key = "image"
            self.label_key = "label"

        self.labels = self.dataset.features[self.label_key].names

        self.dataset = self.dataset.shuffle(seed=42)

        self.iterable_dataset = iter(self.dataset)

        if self.cfg.use_cached_activations:
            """
            Need to implement this. loads stored activations from a file.
            """
            pass

        if create_dataloader:
            if self.cfg.class_token:
                print("Starting to create the data loader!!!")
                self.dataloader = self.get_data_loader()
                print("Data loader created!!!")
            else:
                """
              Need to implement a buffer for the image patch training.
              """
                pass

    def get_batch_of_images_and_labels(self):
        batch_size = self.cfg.max_batch_size_for_vit_forward_pass
        images = []
        labels = []
        for _ in range(batch_size):
            try:
                data = next(self.iterable_dataset)
            except StopIteration:
                self.iterable_dataset = iter(self.dataset.shuffle())
                data = next(self.iterable_dataset)
            image = data[self.image_key]
            label_index = data[self.label_key]
            images.append(image)
            labels.append(f"A photo of a {self.labels[label_index]}.")
        inputs = self.model.processor(
            images=images, text=labels, return_tensors="pt", padding=True
        ).to(self.cfg.device)
        return inputs

    def get_image_batches(self):
        """
        A batch of tokens from a dataset.
        """
        device = self.cfg.device
        batch_of_images = []
        with torch.no_grad():
            for _ in trange(
                self.cfg.store_size, desc="Filling activation store with images"
            ):
                try:
                    batch_of_images.append(next(self.iterable_dataset)[self.image_key])
                except StopIteration:
                    self.iterable_dataset = iter(self.dataset.shuffle())
                    batch_of_images.append(next(self.iterable_dataset)[self.image_key])
        return batch_of_images

    def get_activations(self, image_batches):
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

        if self.cfg.class_token:
            # Only keep the class token
            activations = activations[
                :, 0, :
            ]  # See the forward(), foward_head() methods of the VisionTransformer class in timm.
            # Eg "x = x[:, 0]  # class token" - the [:,0] indexes the batch dimension then the token dimension

        return activations

    def get_sae_batches(self):
        image_batches = self.get_image_batches()
        max_batch_size = self.cfg.max_batch_size_for_vit_forward_pass
        number_of_mini_batches = len(image_batches) // max_batch_size
        remainder = len(image_batches) % max_batch_size
        sae_batches = []
        for mini_batch in trange(
            number_of_mini_batches, desc="Getting batches for SAE"
        ):
            sae_batches.append(
                self.get_activations(
                    image_batches[
                        mini_batch * max_batch_size : (mini_batch + 1) * max_batch_size
                    ]
                )
            )

        if remainder > 0:
            sae_batches.append(self.get_activations(image_batches[-remainder:]))

        sae_batches = torch.cat(sae_batches, dim=0)
        sae_batches = sae_batches.to(self.cfg.device)
        return sae_batches

    def get_data_loader(self) -> DataLoader:
        """
        Return a torch.utils.dataloader which you can get batches from.

        Should automatically refill the buffer when it gets to n % full.
        (better mixing if you refill and shuffle regularly).

        """
        batch_size = self.cfg.batch_size

        sae_batches = self.get_sae_batches()

        dataloader = iter(DataLoader(sae_batches, batch_size=batch_size, shuffle=True))

        return dataloader

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
            self.dataloader = self.get_data_loader()
            return next(self.dataloader)
