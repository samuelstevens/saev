# To Do

A list of everything I need to do for a release.

## Code

* [ ] Guh

## Preprint

* [ ] Experiments
* [ ] Writing

## Release

* [ ] Interactive dashboards
* [ ] Module docstrings
* [ ] Class docstrings
* [ ] Method docstrings
* [ ] Function docstrings

## Preprint - Experiments

* [ ] DINOv2 vs CLIP
* [ ] BioCLIP vs CLIP
* [x] Image classification control
* [ ] Image segmentation control
* [ ] Image generation control (MAE)
* [ ] Image captioning control (Moondream)

## Preprint - Writing

All of these are just "draft" status.

* [ ] Introduction
* [ ] Background & Related Work
* [ ] Methodology
* [ ] Experimental Results
* [ ] Discussion

## Experiments - Understanding DINOv2 vs CLIP

* [x] Compute ImageNet-1K train activations for DINOv2 ViT-B/14
* [x] Compute ImageNet-1K train activations for CLIP ViT-B/16
* [ ] Train SAE on patch-level activations of ImageNet-1K train from DINOv2 ViT-B/14
* [ ] Train SAE on patch-level activations of ImageNet-1K train from CLIP ViT-B/16
* [ ] Visualize features for DINOv2
* [ ] Visualize features for CLIP
* [ ] Find an interesting trait for DINOv2
* [ ] Find an interesting trait for CLIP

## Experiments - Understanding BioCLIP vs CLIP

* [x] Compute iNat21 train-mini activations for BioCLIP ViT-B/16
* [x] Compute iNat21 train-mini activations for CLIP ViT-B/16
* [ ] Train SAE on patch-level activations of iNat21 train-mini from BioCLIP ViT-B/14
* [ ] Train SAE on patch-level activations of iNat21 train-mini from CLIP ViT-B/16
* [ ] Visualize features for BioCLIP
* [ ] Visualize features for CLIP
* [ ] Find an interesting trait for BioCLIP
* [ ] Find an interesting trait for CLIP

## Experiments - Image Classification Control

* [x] Train SAE on [CLS] activations of ImageNet-1K train from CLIP ViT-B/16
* [x] Compute Caltech-101 train activations for CLIP ViT-B/16
* [x] Compute Caltech-101 test activations for CLIP ViT-B/16
* [x] Train linear probe for Caltech-101 classification
* [x] Calculate 99th percentile of feature activation for each feature.
* [x] Develop interactive Marimo dashboard
* [x] Find something neat.
* [ ] Calculate logit relationship

## Experiments - Image Segmentation Control

* [ ] Train SAE on patch-level activations of ImageNet-1K train from DINOv2 ViT-B/14
* [x] Compute ADE20K train activations for DINOv2 ViT-B/14
* [x] Compute ADE20K validation activations for DINOv2 ViT-B/14
* [x] Train linear probe for ADE20K semantic segmentation (`checkpoints/contrib/semseg/lr_0_001__wd_0_001/model_step8000.pt`)
* [x] What percentage of patches meet the 90% threshold?
* [ ] Develop interactive Marimo dashboard
* [ ] Find something neat.
* [ ] Quantitative results

## Experiments - Image Generation Control

* [x] Compute ImageNet-1K train activations for MAE ViT-B/16 
* [ ] Train SAE on patch-level activations of ImageNet-1K train from MAE ViT-B/16
* [ ] Develop interactive Marimo dashboard
* [ ] Find something neat.

## Experiments - Image Caption Control

* [ ] Compute ImageNet-1K train activations for Moondream's vision encoder
* [ ] Train SAE on patch-level activations of ImageNet-1K train from Moondream's vision encoder 
* [ ] Develop interactive Marimo dashboard
* [ ] Find something neat.


