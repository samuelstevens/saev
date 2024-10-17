# 10/17/2024

I trained a couple SAEs using both my updated codebase and the original codebase.
Like, the original codebase, without even ruff/isort added.
I did this because I figured there were four possible problems:

1. The updated training code is wrong.
2. The updated analysis.py file is wrong.
3. The updated get_app_data.py file is wrong.
4. The random seed was causing problems.

The key is that I now have several checkpoints, trained with the original codebase or with the udpated codebase.
I also have updated and original versions of the other scripts.
So I can take each piece through step by step and see where the mistake is.

| Checkpoint | Wandb | Training Code |
|---|---|---|
| 2dlebd60 | [wawwh1rj](https://wandb.ai/samuelstevens/mats-hugo/runs/wawwh1rj) | original |
