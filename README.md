# Folding as Projection

This repository contains code and experiments for exploring **model compression through projection-based folding**.  
It accompanies our ICLR'26 submission (anonymized):  
> *"Cut Less, Fold More: Model Compression through the Lens of Projection Geometry"*

---

## Checkpoints

Trained model checkpoints used in this project can either be generated locally or downloaded from external sources:

- **ResNet18 on CIFAR-10**  
  - Train using scripts in [`train/run.sh`](train/run.sh)

- **CLIP ViT-B/32 model soups (M. Wortsman)**  
  - Paper: [Model Soups](https://arxiv.org/abs/2203.05482)  
  - Checkpoints: [GitHub Release](https://github.com/mlfoundations/model-soups/releases/)

- **PreActResNet18 & ViT experiments (M. Andriushchenko)**  
  - Paper: [Sharpness vs. Generalization](https://arxiv.org/pdf/2302.07011)  
  - Code: [GitHub Repository](https://github.com/tml-epfl/sharpness-vs-generalization/tree/main)  
  - Checkpoints: [Google Drive](https://drive.google.com/drive/folders/1LmthJCb3RXBFWjeTOC4UOOl7Ppgg2h7n)  
  - Notes:
    - Install `vit-pytorch==0.40.2`  
    - See [commit](https://github.com/tml-epfl/sharpness-vs-generalization/commit/6d73be94eb88dae6d3096647bb24b92244fae18f) for compatibility  
    - Training metrics available on Google Drive  

---

## Evaluation on Individual Checkpoints

Install dependencies from `requirements.txt`.  
Assuming checkpoints are located under `../checkpoints/<model_name>/<optimizer_name>`, run:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m pipelines.eval_resnet_compression
CUDA_VISIBLE_DEVICES=0 python3 -m pipelines.eval_preact_resnet_compression
CUDA_VISIBLE_DEVICES=0 python3 -m pipelines.eval_vit_compression
CUDA_VISIBLE_DEVICES=0 python3 -m pipelines.eval_clip_compression
```

## Testing pipelines 
The following scripts reproduce the log files used to generate the plots in our submission.
Logs are automatically stored in the `output` folder.

*ResNet18 (CIFAR-10):*
```
CUDA_VISIBLE_DEVICES=0 python3 -m pipelines.test_resnet_compression --ckpt_dir ../checkpoints/resnet18/adam --method fold
CUDA_VISIBLE_DEVICES=0 python3 -m pipelines.test_resnet_compression --ckpt_dir ../checkpoints/resnet18/adam --method mag-l1
CUDA_VISIBLE_DEVICES=0 python3 -m pipelines.test_resnet_compression --ckpt_dir ../checkpoints/resnet18/adam --method mag-l2
CUDA_VISIBLE_DEVICES=3 python3 -m pipelines.test_resnet_sharpness --ckpt_dir ../checkpoints/resnet18/sgd --method mag-l1 >>output/resnet18/sgd-mag-l1-sharpness
```

*PreActResNet18:*
```
CUDA_VISIBLE_DEVICES=0 python3 -m pipelines.test_preact_resnet_compression --ckpt_dir ../checkpoints/preactresnet18 --method fold --epochs 5
CUDA_VISIBLE_DEVICES=7 python3 -m pipelines.test_preact_resnet_sharpness --ckpt_dir ../checkpoints/preactresnet18 --method mag-l2 --sharp-layer-pattern "visual.transformer.resblocks.11" >>output/preactresnet18/mag-l2-sharpness
CUDA_VISIBLE_DEVICES=3 python3 -m pipelines.test_preact_resnet_zeroshot --ckpt_dir ../checkpoints/preactresnet18 --method mag-l2 >>output/preactresnet18/mag-l2-zeroshot
CUDA_VISIBLE_DEVICES=1 python3 -m pipelines.test_preact_resnet_compression --ckpt_dir ../checkpoints/preactresnet18 --method fisher --epochs 0 >>output/preactresnet18/fisher
```
*ViT experiments:*
```
CUDA_VISIBLE_DEVICES=0 python3 -m pipelines.test_vit_compression --ckpt_dir ../checkpoints/vit-exp --method fold --epochs 5
```
*CLIP model soups:*
```
CUDA_VISIBLE_DEVICES=0 python3 -m pipelines.test_clip_compression --ckpt_dir ../checkpoints/clipvit-b32-model-soups --method fold --epochs 5
CUDA_VISIBLE_DEVICES=7 python3 -m pipelines.test_clip_sharpness --ckpt_dir ../checkpoints/clipvit-b32-model-soups --method mag-l2 >>output/clip/mag-l2-sharpness
CUDA_VISIBLE_DEVICES=2 python3 -m pipelines.test_clip_zeroshot --ckpt_dir ../checkpoints/clipvit-b32-model-soups --method mag-l2 >>output/clip/mag-l2-zeroshot
```

## Reproducing Plots
- Full experiment logs are available in the `output` folder.
- Use the Jupyter notebooks in `notebooks` to reproduce the plots shown in the paper.

