import os
import sys
import random
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.resnet import ResNet18

from compression.fold import ResNet18_ModelFolding
from compression.mag_prune import ResNet18_MagnitudePruning
from compression.rand_fold import ResNet18_RandomFolding
from compression.rand_prune import ResNet18_RandomPruning
from compression.singleton import ResNet18_Singleton

from utils.eval_utils import test, count_parameters
from utils.tune_utils import repair_bn


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# CHECKPOINT_PATH = "../checkpoints/resnet18/adam/2025-06-08_08-18-22_dataset=cifar10_arch=resnet18_opt=adam_seed=42_lr=0.01_batch_size=128_momentum=0.0_wd=0.0_epochs=200_l1=1e-05_l2=0.0_sam=False_sam_rho=0.05_rand_aug=False_lr_schedule=True.pth"
# CHECKPOINT_PATH = "../checkpoints/resnet18/adam/2025-06-07_17-27-38_dataset=cifar10_arch=resnet18_opt=adam_seed=42_lr=0.1_batch_size=128_momentum=0.0_wd=0.0_epochs=200_l1=0.0_l2=0.0_sam=False_sam_rho=0.05_rand_aug=False_lr_schedule=True.pth"
CHECKPOINT_PATH = "../checkpoints/resnet18/adam/2025-06-07_17-27-59_dataset=cifar10_arch=resnet18_opt=adam_seed=42_lr=0.1_batch_size=128_momentum=0.0_wd=0.0_epochs=200_l1=0.0_l2=0.0_sam=False_sam_rho=0.05_rand_aug=False_lr_schedule=False.pth"
BATCH_SIZE = 128
COMPRESSION_RATIO = 0.1
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
# Dataset and transforms
# -----------------------------------------------------------------------------
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
# Model loading
# -----------------------------------------------------------------------------
def load_resnet18_model(num_classes, checkpoint_path=None):
    """
    Load our custom ResNet18 model and optionally load pretrained weights.
    """
    model = ResNet18(num_classes=num_classes)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        print("[WARNING] No checkpoint found, using randomly initialized model.")

    return model

# -----------------------------------------------------------------------------
# Main compression evaluation
# -----------------------------------------------------------------------------
def main():
    num_classes = 10
    model = load_resnet18_model(num_classes, CHECKPOINT_PATH).to(DEVICE)

    # Evaluate before folding
    # print("\n=== Evaluation BEFORE compression ===")
    # acc_before = test(model, val_loader, device=DEVICE)
    # print(f"🔹 Top-1 Accuracy: {acc_before:.2f}%")
    original_params = count_parameters(model)
    print(f"Original Parameters: {original_params}")

    # Apply folding
    print("\n[INFO] Applying ResNet18 model compression...")
    # pruner = ResNet18_ModelFolding(model, compression_ratio=COMPRESSION_RATIO)
    pruner = ResNet18_MagnitudePruning(model, compression_ratio=COMPRESSION_RATIO, p=2)
    # pruner = ResNet18_RandomFolding(model, compression_ratio=COMPRESSION_RATIO)
    # pruner = ResNet18_RandomPruning(model, compression_ratio=COMPRESSION_RATIO)
    # pruner = ResNet18_Singleton(model, compression_ratio=COMPRESSION_RATIO)

    pruned_model = pruner.apply()

    print("\n=== Evaluation AFTER compression (before REPAIR) ===")
    pruned_params = count_parameters(pruned_model)
    acc_after = test(pruned_model, val_loader, device=DEVICE)
    print(f"🔹 Top-1 Accuracy: {acc_after:.2f}%")

    # Apply REPAIR
    print("\n[INFO] Re-tuning BatchNorm parameters...")
    repair_bn(pruned_model, train_loader)

    # Evaluate after folding
    print("\n=== Evaluation AFTER compression ===")
    acc_after = test(pruned_model, val_loader, device=DEVICE)
    pruned_params = count_parameters(pruned_model)
    print(f"🔹 Top-1 Accuracy: {acc_after:.2f}%")
    print(f"Pruned Parameters: {pruned_params}")
    print(f"🔥 Compression Ratio: {(original_params - pruned_params) / original_params:.2%}")

if __name__ == "__main__":
    main()
