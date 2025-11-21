import os
import sys
import glob
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.clip_vit import CLIPViT_B32
from compression.fold import CLIPViT_ModelFolding
from compression.mag_prune import CLIPViT_MagnitudePruning

from utils.eval_utils import test, count_parameters, get_outputs
from utils.tune_utils import retune_layernorm

# --------------------------------------------------------
# Utils
# --------------------------------------------------------
def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_line(ratio, event, **kwargs):
    """Unified concise logging"""
    parts = [f"RATIO={ratio:.1f}", f"EVENT={event}"]
    parts += [f"{k}={v}" for k, v in kwargs.items()]
    print(" ".join(parts))

# --------------------------------------------------------
# Model loading
# --------------------------------------------------------
def load_clip_vit_model(num_classes, checkpoint_path, device):
    """
    Load CLIP ViT-B/32 model and preprocessing transform.
    """
    clip_loader = CLIPViT_B32(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        device=device
    )
    return clip_loader.load()  # returns (model, preprocess)

# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP ViT-B/32 compression across ratios/checkpoints")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory with CLIP checkpoints (.pt)")
    parser.add_argument("--method", type=str, default="fold",
                        choices=["fold", "mag-l1", "mag-l2"])
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tuning epochs after compression")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate for fine-tuning")
    parser.add_argument("--imagenet_root", type=str, default="../data", help="Path to ImageNet root")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    fix_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 1000

    # Load checkpoints
    ckpt_paths = sorted(glob.glob(f"{args.ckpt_dir}/*.pt"))
    prune_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Load preprocess from first checkpoint
    _, preprocess = load_clip_vit_model(num_classes, ckpt_paths[0], device)

    # Prepare datasets
    train_dataset = ImageNet(root=args.imagenet_root, split="train", transform=preprocess)
    val_dataset = ImageNet(root=args.imagenet_root, split="val", transform=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Map pruning methods
    pruner_map = {
        "fold":      lambda m, r: CLIPViT_ModelFolding(m, compression_ratio=r),
        "mag-l1":    lambda m, r: CLIPViT_MagnitudePruning(m, compression_ratio=r, p=1),
        "mag-l2":    lambda m, r: CLIPViT_MagnitudePruning(m, compression_ratio=r, p=2),
    }

    for i, ckpt_path in enumerate(ckpt_paths):
        model_name = os.path.basename(ckpt_path)
        for ratio in prune_ratios:
            # Load model fresh per ratio
            model, _ = load_clip_vit_model(num_classes, ckpt_path, device)
            orig_params = count_parameters(model)

            # Baseline evaluation
            if ratio == 0.0:
                acc = test(model, val_loader, device)
                if acc < 10.0:  # sanity check threshold for ImageNet
                    break
                print(f"\n[MODEL] {i + 1}/{len(ckpt_paths)} {model_name}")
                log_line(ratio, "BASE", params=orig_params, acc=f"{acc:.2f}")
                orig_outputs = get_outputs(model.eval(), val_loader, device)
                continue

            # Apply compression
            pruner = pruner_map[args.method](model, ratio)
            model = pruner.apply().to(device)
            pruned_params = count_parameters(model)

            # Test after pruning
            acc = test(model, val_loader, device)
            log_line(ratio, "PRUNE", params=pruned_params, acc=f"{acc:.2f}")

            # # Retune LayerNorm
            # retune_layernorm(model, train_loader, device=device, lr=5e-5)
            # acc = test(model, val_loader, device)
            # log_line(ratio, "REPAIR", acc=f"{acc:.2f}")

            # # Functional deviation
            # fd = torch.norm(orig_outputs - get_outputs(model.eval(), val_loader, device), dim=1).mean().item()
            # log_line(ratio, "FD", value=f"{fd:.4f}")

            # Optional fine-tune on ImageNet training set
            if args.epochs > 0:
                opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
                loss_fn = nn.CrossEntropyLoss()
                for epoch in range(args.epochs):
                    model.train()
                    for x, y in train_loader:
                        x, y = x.to(device), y.to(device)
                        opt.zero_grad()

                        # Forward pass (CLIP ViT vs ResNet)
                        if hasattr(model, "classification_head") and hasattr(model, "visual"):
                            out = model.classification_head(model.visual(x))
                        else:
                            out = model(x)
                        if isinstance(out, tuple):
                            out = out[0]

                        loss = loss_fn(out, y)
                        loss.backward()
                        opt.step()

                    acc = test(model, val_loader, device)
                    log_line(ratio, f"FINETUNE_EPOCH{epoch + 1}", acc=f"{acc:.2f}")

if __name__ == "__main__":
    main()
