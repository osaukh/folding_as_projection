import os
import sys
import glob
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.preact_resnet import PreActResNet18
from compression.fold import PreActResNet18_ModelFolding
from compression.mag_prune import PreActResNet18_MagnitudePruning
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
    parts = [f"[BASE] Ratio={ratio:.1f}", f"Eval={event}"]
    if "Params" in kwargs:
        parts.insert(1, f"Params={kwargs.pop('Params')}")
    if "RESULT" in kwargs:
        # pass already formatted RESULT string
        parts.append(f"[RESULT] {kwargs.pop('RESULT')}")
    parts += [f"{k}={v}" for k, v in kwargs.items()]
    print(" ".join(parts))


# --------------------------------------------------------
# CIFAR-10 data
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


def get_dataloaders(data_root="../data", batch_size=128, num_workers=8):
    train_ds = datasets.CIFAR10(data_root, train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(data_root, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# --------------------------------------------------------
# CIFAR-10-C dataset
# --------------------------------------------------------
CIFAR10C_CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
    "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
    "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise",
    "snow", "spatter", "speckle_noise", "zoom_blur",
]


class CIFAR10CDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # numpy array (N, 32, 32, 3), uint8
        self.labels = labels  # numpy array (N,)
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]  # HWC, uint8
        lbl = int(self.labels[idx])

        # ToTensor expects HWC uint8 ndarray
        if self.transform is not None:
            img = self.transform(img)
        return img, lbl


def evaluate_cifar10c(model, cifar10c_root, device, batch_size=128, num_workers=4):
    """
    Evaluate model on CIFAR-10-C across all corruptions and severities.
    Returns:
      results: dict[(corruption, severity)] -> accuracy
      mean_acc: float
    """
    model.eval()
    transform_c = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*norm),
    ])

    labels_path = os.path.join(cifar10c_root, "labels.npy")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Could not find CIFAR-10-C labels at {labels_path}")
    labels_all = np.load(labels_path)

    results = {}
    all_accs = []

    for corruption in CIFAR10C_CORRUPTIONS:
        corr_path = os.path.join(cifar10c_root, f"{corruption}.npy")
        if not os.path.exists(corr_path):
            print(f"[WARN] Corruption file not found: {corr_path}, skipping.")
            continue

        data_all = np.load(corr_path)  # shape (50000, 32, 32, 3)

        for severity in range(1, 6):
            start = (severity - 1) * 10000
            end = severity * 10000

            data = data_all[start:end]
            labels = labels_all[start:end]

            dataset = CIFAR10CDataset(data, labels, transform=transform_c)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            acc = test(model, loader, device)
            results[(corruption, severity)] = acc
            all_accs.append(acc)

            print(f"[CIFAR10-C] corruption={corruption:>18s} severity={severity} acc={acc:.2f}")

    if len(all_accs) > 0:
        mean_acc = float(np.mean(all_accs))
    else:
        mean_acc = float("nan")

    print(f"[CIFAR10-C] MEAN_ACC={mean_acc:.2f}")
    return results, mean_acc


# --------------------------------------------------------
# Model
# --------------------------------------------------------
def load_preact_resnet18_model(num_classes, checkpoint_path, device):
    model = PreActResNet18(
        n_cls=num_classes, model_width=64, half_prec=False,
        activation='relu', droprate=0.0,
        normalize_features=False, normalize_logits=False
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    # For checkpoints saved with key 'last'
    if isinstance(state_dict, dict) and 'last' in state_dict:
        model.load_state_dict(state_dict['last'])
    else:
        model.load_state_dict(state_dict)
    return model


# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PreActResNet18 compression & CIFAR-10-C robustness across ratios/checkpoints"
    )
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Directory with CIFAR-10 PreActResNet18 checkpoints (.pth)")
    parser.add_argument("--method", type=str, default="fold",
                        choices=["fold", "mag-l1", "mag-l2"],
                        help="Compression method: fold, mag-l1, mag-l2")
    parser.add_argument("--data-root", type=str, default="../data",
                        help="Root directory containing CIFAR-10 and CIFAR-10-C")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    fix_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CIFAR-10 loaders (for clean accuracy + BN repair)
    train_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # CIFAR-10-C root
    cifar10c_root = os.path.join(args.data_root, "CIFAR-10-C")

    # Checkpoints
    ckpt_paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pth")))
    if len(ckpt_paths) == 0:
        print(f"No checkpoints found in {args.ckpt_dir}")
        return

    # Compression setup
    compression_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    prune_ratios = [0.0] + compression_ratios  # include baseline

    pruner_map = {
        "fold":   lambda m, r: PreActResNet18_ModelFolding(m, compression_ratio=r),
        "mag-l1": lambda m, r: PreActResNet18_MagnitudePruning(m, compression_ratio=r, p=1),
        "mag-l2": lambda m, r: PreActResNet18_MagnitudePruning(m, compression_ratio=r, p=2),
    }

    loss_fn = nn.CrossEntropyLoss()

    for i, ckpt_path in enumerate(ckpt_paths):
        model_name = os.path.basename(ckpt_path)
        print(f"\n{model_name} ========")

        # We reload per ratio so each compression starts from a clean model
        for ratio in prune_ratios:
            # Load fresh model per ratio
            model = load_preact_resnet18_model(10, ckpt_path, device)
            orig_params = count_parameters(model)

            # Baseline (only for ratio=0.0)
            if ratio == 0.0:
                acc_clean = test(model, test_loader, device)
                # skip terrible models
                if acc_clean < 50.0:
                    print(f"Skipping model {model_name} (clean acc={acc_clean:.2f} < 50)")
                    break

                print(f"[MODEL] {i + 1}/{len(ckpt_paths)} {model_name}")
                log_line(
                    ratio,
                    "BeforePruning_Clean",
                    Params=orig_params,
                    RESULT=f"TestAcc={acc_clean:.2f}"
                )

                # Zero-shot CIFAR-10-C robustness for baseline model
                _, mean_acc_c = evaluate_cifar10c(
                    model, cifar10c_root, device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers
                )
                print(f"[CIFAR10-C_BASE] Ratio={ratio:.1f} MeanAcc={mean_acc_c:.2f}")
                continue

            # --------------------------------------------------------
            # Apply compression (fold / mag-l1 / mag-l2)
            # --------------------------------------------------------
            pruner = pruner_map[args.method](model, ratio)
            model = pruner.apply().to(device)
            pruned_params = count_parameters(model)

            # Eval immediately after pruning/folding on clean CIFAR-10
            acc_pruned = test(model, test_loader, device)
            log_line(
                ratio,
                "AfterPruning_Clean",
                Params=pruned_params,
                RESULT=f"TestAcc={acc_pruned:.2f}"
            )

            # --------------------------------------------------------
            # Repair BN (on CIFAR-10 train)
            # --------------------------------------------------------
            repair_bn(model, train_loader)
            acc_repair = test(model, test_loader, device)
            log_line(
                ratio,
                "AfterRepair_Clean",
                Params=pruned_params,
                RESULT=f"TestAcc={acc_repair:.2f}"
            )

            # --------------------------------------------------------
            # Zero-shot CIFAR-10-C robustness (no adaptation on C)
            # --------------------------------------------------------
            _, mean_acc_c = evaluate_cifar10c(
                model, cifar10c_root, device,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            print(f"[CIFAR10-C_COMPRESSED] Ratio={ratio:.1f} Method={args.method} MeanAcc={mean_acc_c:.2f}")


if __name__ == "__main__":
    main()
