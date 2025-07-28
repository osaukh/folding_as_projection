import os
import sys
import random
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.preact_resnet import PreActResNet18
from compression.fold import PreActResNet18_ModelFolding

from utils.eval_utils import test, count_parameters
from utils.tune_utils import repair_bn


CHECKPOINT_PATH = "../checkpoints/preactresnet18/2023-01-15 14:02:36.368 dataset=cifar10 model=resnet18 epochs=200 lr_max=0.4633774 model_width=64 l2_reg=0.0 sam_rho=0.05 batch_size=128 frac_train=1 p_label_noise=0.0 lr_schedule=cyclic augm=True randaug=True seed=0 epoch=200.pth"
BATCH_SIZE = 128
COMPRESSION_RATIO = 0.5
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load PreActResNet18 ----
    model = PreActResNet18(num_classes=10).to(device)

    # ---- Load checkpoint (adjust path) ----
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['last'])

    # ---- Evaluation BEFORE compression ----
    # print("=== Evaluation BEFORE compression ===")
    # acc_before = test(model, test_loader, device)
    # print(f"Top-1 Accuracy: {acc_before:.2f}%")
    original_params = count_parameters(model)
    print(f"Original Parameters: {original_params}")

    # ---- Apply compression (migrate to PreAct variant later) ----
    # pruner = PreActResNet18_MagnitudePruning(model, compression_ratio=0.5, p=2)
    pruner = PreActResNet18_ModelFolding(model, compression_ratio=0.5)

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
