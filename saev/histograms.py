"""
Evaluates SAE quality by counting the number of dead features and the number of dense features.
Optionally makes histogram plots to help human qualitative comparison.


.. todo:: Develop automatic methods to use histogram and feature frequencies to evaluate quality with a single number.
"""

import beartype
import matplotlib.pyplot as plt
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from . import activations, config, helpers, nn


@jaxtyped(typechecker=beartype.beartype)
def plot_log10_hist(frequencies: Float[Tensor, " d_sae"]):
    """
    Plot the histogram of feature frequency.
    """
    fig, ax = plt.subplots()
    frequencies = torch.log10(frequencies)
    ax.hist(frequencies, bins=50)
    fig.tight_layout()
    return fig


@beartype.beartype
@torch.inference_mode()
def evaluate(cfg: config.HistogramsEvaluate) -> float:
    """
    Count the number of feature activations on the training set.
    """
    sae = nn.load(cfg.ckpt_path)
    sae.eval()

    dataset = activations.Dataset(cfg.shard_root)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, num_workers=cfg.n_workers, shuffle=False
    )

    n_fired = torch.zeros(sae.d_sae)

    for vit_acts, _ in helpers.progress(dataloader):
        _, f_x, *_ = sae(vit_acts)
        n_fired += (f_x > 0).sum(axis=0)
        print((n_fired == 0).sum())  # number of dead features
        # TODO: do the same for dense features (>1/100), plot histograms, use CUDA device, etc.