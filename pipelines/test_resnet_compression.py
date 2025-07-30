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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.resnet import ResNet18
from compression.fold import ResNet18_ModelFolding
from compression.mag_prune import ResNet18_MagnitudePruning
from compression.rand_fold import ResNet18_RandomFolding
from compression.rand_prune import ResNet18_RandomPruning
from compression.singleton import ResNet18_Singleton
from utils.eval_utils import test, count_parameters
from utils.tune_utils import repair_bn

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
# Data
# --------------------------------------------------------
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

# --------------------------------------------------------
# Model
# --------------------------------------------------------
def load_resnet18_model(num_classes, checkpoint_path):
    model = ResNet18(num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model

# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate ResNet18 compression across ratios/checkpoints")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--method", type=str, default="fold",
                        choices=["fold", "mag-l1", "mag-l2", "rand-fold", "rand-prune", "singleton"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    fix_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = get_dataloaders()
    ckpt_paths = sorted(glob.glob(f"{args.ckpt_dir}/*.pth"))
    prune_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    pruner_map = {
        "fold":      lambda m, r: ResNet18_ModelFolding(m, compression_ratio=r),
        "mag-l1":    lambda m, r: ResNet18_MagnitudePruning(m, compression_ratio=r, p=1),
        "mag-l2":    lambda m, r: ResNet18_MagnitudePruning(m, compression_ratio=r, p=2),
        "rand-fold": lambda m, r: ResNet18_RandomFolding(m, compression_ratio=r),
        "rand-prune":lambda m, r: ResNet18_RandomPruning(m, compression_ratio=r),
        "singleton": lambda m, r: ResNet18_Singleton(m, compression_ratio=r),
    }

    for i, ckpt_path in enumerate(ckpt_paths):
        model_name = os.path.basename(ckpt_path)
        for ratio in prune_ratios:
            # Load fresh model per ratio
            model = load_resnet18_model(10, ckpt_path).to(device)
            orig_params = count_parameters(model)

            # Baseline (only for ratio=0.0)
            if ratio == 0.0:
                acc = test(model, test_loader, device)
                if acc < 50.0:
                    break
                print(f"\n[MODEL] {i + 1}/{len(ckpt_paths)} {model_name}")
                log_line(ratio, "BASE", params=orig_params, acc=f"{acc:.2f}")
                continue

            # Apply pruning/folding
            pruner = pruner_map[args.method](model, ratio)
            model = pruner.apply().to(device)
            pruned_params = count_parameters(model)

            acc = test(model, test_loader, device)
            log_line(ratio, "PRUNE", params=pruned_params, acc=f"{acc:.2f}")

            # Repair BN
            repair_bn(model, train_loader)
            acc = test(model, test_loader, device)
            log_line(ratio, "REPAIR", acc=f"{acc:.2f}")

            # Optional fine-tune
            if args.epochs > 0:
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                loss_fn = nn.CrossEntropyLoss()
                for epoch in range(args.epochs):
                    model.train()
                    for x, y in train_loader:
                        x, y = x.to(device), y.to(device)
                        opt.zero_grad()
                        loss = loss_fn(model(x), y)
                        loss.backward()
                        opt.step()
                    acc = test(model, test_loader, device)
                    log_line(ratio, f"FINETUNE_EPOCH{epoch+1}", acc=f"{acc:.2f}")

if __name__ == "__main__":
    main()
