import os
import sys
import random
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

from models.clip_vit import CLIPViT_B32

from compression.fold import CLIPViT_ModelFolding
from compression.mag_prune import CLIPViT_MagnitudePruning

from utils.tune_utils import retune_layernorm
from utils.eval_utils import test, count_parameters

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATASET = "ImageNet"
CHECKPOINT_PATH = "../checkpoints/clipvit-b32-model-soups/model_1.pt"
IMAGENET_ROOT = "../data"
BATCH_SIZE = 32
COMPRESSION_RATIO = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
fix_seed(42)

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Main folding evaluation
# -----------------------------------------------------------------------------
def main():
    num_classes = 1000  # ImageNet classes
    model, preprocess = load_clip_vit_model(num_classes, CHECKPOINT_PATH, DEVICE)

    # Prepare validation loader
    val_dataset = ImageNet(root=IMAGENET_ROOT, split="val", transform=preprocess)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Evaluate BEFORE folding
    print("\n=== Evaluation BEFORE compression ===")
    # acc_before = test(model, val_loader, device=DEVICE)
    # print(f"🔹 Top-1 Accuracy: {acc_before:.2f}%")
    original_params = count_parameters(model)
    print(f"Original Parameters: {original_params}")

    # Apply folding
    print("\n[INFO] Applying CLIP ViT-B/32 model compression...")
    # pruner = CLIPViT_ModelFolding(model, compression_ratio=COMPRESSION_RATIO)
    pruner = CLIPViT_MagnitudePruning(model, compression_ratio=COMPRESSION_RATIO, p=1)

    pruned_model = pruner.apply()

    # Evaluate AFTER folding (before LN re-tune)
    print("\n=== Evaluation AFTER compression (before LN re-tune) ===")
    pruned_params = count_parameters(pruned_model)
    acc_after = test(pruned_model, val_loader, device=DEVICE)
    print(f"🔹 Top-1 Accuracy: {acc_after:.2f}%")

    # Re-tune LayerNorm
    print("\n[INFO] Re-tuning LayerNorm parameters...")
    retune_layernorm(pruned_model, val_loader, device=DEVICE, lr=1e-4)

    # Evaluate AFTER LN re-tune
    print("\n=== Evaluation AFTER compression ===")
    acc_after_ln = test(pruned_model, val_loader, device=DEVICE)
    print(f"🔹 Top-1 Accuracy (after LN re-tune): {acc_after_ln:.2f}%")
    print(f"Pruned Parameters: {pruned_params}")
    print(f"🔥 Compression Ratio: {(original_params - pruned_params) / original_params:.2%}")

if __name__ == "__main__":
    main()
