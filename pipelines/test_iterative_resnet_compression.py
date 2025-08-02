import os
import sys
import glob
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.resnet import ResNet18
from compression.fold import ResNet18_ModelFolding
from compression.mag_prune import ResNet18_MagnitudePruning
from utils.eval_utils import test, count_parameters

# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------
def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_line(iter_idx, params, acc):
    """Concise log format for easy parsing."""
    print(f"ITER={iter_idx} PARAMS={params} ACC={acc:.2f}")

# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(*norm),
])
transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(*norm),
])

def get_dataloaders():
    train_ds = datasets.CIFAR10("../data", train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10("../data", train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, test_loader

# -------------------------------------------------------------------------
# Model loader
# -------------------------------------------------------------------------
def load_resnet18_model(num_classes, checkpoint_path):
    model = ResNet18(num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model

# -------------------------------------------------------------------------
# Iterative compression
# -------------------------------------------------------------------------
def iterative_compression(model, method, prune_fraction, iterations, epochs, lr, train_loader, test_loader, device):
    """
    Iteratively prune/fold fraction of weights/channels and fine-tune.
    Logs PARAM count and accuracy after each iteration.
    """
    # Select pruner factory
    if method == "fold":
        pruner_factory = lambda m, ratio: ResNet18_ModelFolding(m, compression_ratio=ratio)
    elif method == "mag-l1":
        pruner_factory = lambda m, ratio: ResNet18_MagnitudePruning(m, compression_ratio=ratio, p=1)
    elif method == "mag-l2":
        pruner_factory = lambda m, ratio: ResNet18_MagnitudePruning(m, compression_ratio=ratio, p=2)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Iterative process
    for it in range(1, iterations + 1):
        # Apply pruning/folding
        pruner = pruner_factory(model, prune_fraction)
        model = pruner.apply().to(device)

        # Fine-tune for given epochs
        if epochs > 0:
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()
            for _ in range(epochs):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    loss = loss_fn(model(x), y)
                    loss.backward()
                    opt.step()

        # Evaluate and log
        acc = test(model.eval(), test_loader, device=device)
        params = count_parameters(model)
        log_line(it, params, acc)

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Iterative ResNet18 compression evaluation")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory with checkpoints (.pth)")
    parser.add_argument("--method", type=str, required=True, choices=["fold", "mag-l1", "mag-l2"])
    parser.add_argument("--prune_fraction", type=float, default=0.2, help="Fraction pruned/folded per iteration")
    parser.add_argument("--iterations", type=int, default=7, help="Number of prune/fold iterations")
    parser.add_argument("--epochs", type=int, default=1, help="Fine-tune epochs per iteration")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning")
    args = parser.parse_args()

    fix_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare data
    train_loader, test_loader = get_dataloaders()

    # Process checkpoints
    ckpt_paths = sorted(glob.glob(f"{args.ckpt_dir}/*.pth"))
    for i, ckpt_path in enumerate(ckpt_paths):
        model_name = os.path.basename(ckpt_path)
        print(f"\n[MODEL] {i + 1}/{len(ckpt_paths)} {model_name}")

        # Load fresh model
        model = load_resnet18_model(10, ckpt_path).to(device)

        # Evaluate baseline
        acc_before = test(model.eval(), test_loader, device=device)
        params_before = count_parameters(model)
        print(f"BASE PARAMS={params_before} ACC={acc_before:.2f}")

        # Run iterative pruning/folding
        iterative_compression(
            model, args.method,
            prune_fraction=args.prune_fraction,
            iterations=args.iterations,
            epochs=args.epochs,
            lr=args.lr,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device
        )

if __name__ == "__main__":
    main()
