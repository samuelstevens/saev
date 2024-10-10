import beartype
import datasets
import torch
import tqdm
from jaxtyping import Float, jaxtyped
from torch import Tensor

from sae_training.config import Config
from sae_training.hooked_vit import HookedVisionTransformer


@beartype.beartype
def make_img_dataloader(
    cfg: Config, dataset, preprocess
) -> tuple[object, torch.utils.data.DataLoader]:
    n_workers = 0

    def batched_transform(example):
        processed = preprocess(
            images=example["image"],
            text=[""] * len(example["image"]),
            return_tensors="pt",
            padding=True,
        )
        example.update(**processed)
        return example

    dataset = (
        dataset.to_iterable_dataset(num_shards=n_workers or 8)
        .shuffle(seed=cfg.seed)
        .map(
            batched_transform,
            batched=True,
            batch_size=cfg.batch_size,
            remove_columns=list(dataset.features.keys()),
        )
        .with_format("torch")
    )

    return dataset, torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        drop_last=False,
        num_workers=n_workers,
        persistent_workers=False,
        shuffle=False,
        pin_memory=False,
    )


@jaxtyped(typechecker=beartype.beartype)
class ActivationsStore:
    """
    Class for streaming tokens and generating and storing activations while training SAEs.

    Uses a multiprocess torch DataLoader to preprocess images (resize, crop, etc) faster, but still uses the main process to convert images (as tensors) into ViT activations.
    """

    cfg: Config
    vit: HookedVisionTransformer
    hook_loc: tuple[int, str]
    _epoch: int = 0

    def __init__(
        self,
        cfg: Config,
        vit: HookedVisionTransformer,
    ):
        self.cfg = cfg
        self.vit = vit
        self.dataset = datasets.load_dataset(
            self.cfg.dataset_path, split="train", trust_remote_code=True
        )
        self.labels = self.dataset.features["label"].names
        self.dataset, self.dataloader = make_img_dataloader(
            cfg, self.dataset, vit.processor
        )
        self.dataloader_it = iter(self.dataloader)

        self.hook_loc = (cfg.block_layer, cfg.module_name)
        assert cfg.vit_batch_size == cfg.batch_size

    def get_sae_batches(self) -> Float[Tensor, "store_size d_model"]:
        n_examples = 0
        examples = []
        pbar = tqdm.tqdm(total=self.cfg.store_size, desc="Getting SAE batches")
        while n_examples < self.cfg.store_size:
            batch = self.next_batch().cpu()
            examples.append(batch)
            n_examples += len(batch)
            pbar.update(len(batch))

        examples = torch.cat(examples, dim=0)
        examples = examples[: self.cfg.store_size]
        return examples

    @torch.inference_mode
    def next_batch(self) -> Float[Tensor, "batch d_model"]:
        try:
            inputs = next(self.dataloader_it)
        except StopIteration:
            self._epoch += 1
            self.dataset.set_epoch(self._epoch)
            self.dataloader_it = iter(self.dataloader)
            inputs = next(self.dataloader_it)

        # 3. Get ViT activations.
        inputs = {key: value.to(self.cfg.device) for key, value in inputs.items()}
        _, cache = self.vit.run_with_cache([self.hook_loc], **inputs)
        activations = cache[self.hook_loc]

        # Only keep the class token.
        activations = activations[:, 0, :]
        return activations
