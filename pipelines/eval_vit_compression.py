import os
import sys
import random
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from compression.fold import ViT_ModelFolding
from compression.mag_prune import ViT_MagnitudePruning
from compression.wanda import ViT_WandaPruning

from utils.tune_utils import retune_layernorm
from utils.eval_utils import test, count_parameters


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CHECKPOINT_PATH = "../checkpoints/vit-exp/2023-01-14 23_42_12.896 dataset=cifar10 model=vit_exp epochs=200 lr_max=0.056401 model_width=512 l2_reg=0.0 sam_rho=0.0 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=True randaug=True seed=0 epoch=200.pth"
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

# ============================================================
# Dataset and transforms
# ============================================================
norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(*norm),
])
transform_val = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(*norm),
])
train_dataset = datasets.CIFAR10("../data", train=True, download=True, transform=transform_train)
val_dataset = datasets.CIFAR10("../data", train=False, download=True, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

# -----------------------------------------------------------------------------
# Main compression evaluation
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Load pretrained ViT model
    from vit_pytorch import SimpleViT
    model = SimpleViT(image_size=32, patch_size=4, num_classes=10, dim=512, depth=6, heads=16,
                      mlp_dim=512 * 2)
    model_dict = torch.load(CHECKPOINT_PATH)['last']
    model.load_state_dict({k: v for k, v in model_dict.items()})
    model = model.to(DEVICE).eval()

    # print("\n=== Evaluation BEFORE compression ===")
    # acc_before = test(model, val_loader, device=DEVICE)
    # print(f"🔹 Top-1 Accuracy: {acc_before:.2f}%")
    original_params = count_parameters(model)
    print(f"Original Parameters: {original_params}")

    # Apply folding
    print("\n[INFO] Applying ResNet18 model compression...")
    # pruner = ViT_ModelFolding(model, compression_ratio=COMPRESSION_RATIO)
    # pruner = ViT_MagnitudePruning(model, compression_ratio=COMPRESSION_RATIO, p=2)

    pruner = ViT_WandaPruning(model, compression_ratio=COMPRESSION_RATIO)
    pruner.run_calibration(train_loader, DEVICE, num_batches=50)

    pruned_model = pruner.apply()

    print("\n=== Evaluation AFTER compression (before REPAIR) ===")
    pruned_params = count_parameters(pruned_model)
    acc_after = test(pruned_model, val_loader, device=DEVICE)
    print(f"🔹 Top-1 Accuracy: {acc_after:.2f}%")

    # Re-tune LayerNorm
    print("\n[INFO] Re-tuning LayerNorm parameters...")
    retune_layernorm(pruned_model, val_loader, device=DEVICE, lr=1e-4)

    # Evaluate after folding
    print("\n=== Evaluation AFTER compression ===")
    acc_after = test(pruned_model, val_loader, device=DEVICE)
    pruned_params = count_parameters(pruned_model)
    print(f"🔹 Top-1 Accuracy: {acc_after:.2f}%")
    print(f"Pruned Parameters: {pruned_params}")
    print(f"🔥 Compression Ratio: {(original_params - pruned_params) / original_params:.2%}")
