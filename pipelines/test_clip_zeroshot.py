import os
import sys
import glob
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageNet
from PIL import Image, ImageFile

# Allow loading truncated images instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.clip_vit import CLIPViT_B32
from compression.fold import CLIPViT_ModelFolding
from compression.mag_prune import CLIPViT_MagnitudePruning
from utils.eval_utils import test, count_parameters
from utils.tune_utils import retune_layernorm  # optional, currently unused


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
    """Unified concise logging (keep CLIP-style)."""
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
# ImageNet-C dataset & evaluation
# --------------------------------------------------------
IMAGENETC_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression",
]


class ImageNetCDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        assert len(image_paths) == len(labels), \
            f"len(image_paths)={len(image_paths)} != len(labels)={len(labels)}"
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = int(self.labels[idx])
        try:
            img = Image.open(path).convert("RGB")
        except OSError:
            # Gracefully handle truncated / corrupted images:
            # fallback to a dummy black image (size will be handled by transform)
            # You can also skip instead, but that complicates indexing.
            print(f"[WARN] Truncated or unreadable image: {path}, using dummy image.")
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform is not None:
            img = self.transform(img)
        return img, label


def quick_check_imagenet_c(imagenet_c_root):
    """
    Light sanity check: at least one corruption folder exists.
    Do NOT require all corruptions (some may be missing / partial).
    """
    existing = [c for c in IMAGENETC_CORRUPTIONS
                if os.path.isdir(os.path.join(imagenet_c_root, c))]
    if not existing:
        raise FileNotFoundError(
            f"ImageNet-C not found under {imagenet_c_root}.\n"
            f"Expected structure like {os.path.join(imagenet_c_root, 'gaussian_noise', '1', '<wnid>', 'ILSVRC2012_val_00000001.JPEG')}"
        )
    print(f"[INFO] Found ImageNet-C corruptions: {existing}")


def build_filename_to_label_map(val_dataset):
    """
    Build mapping:
        'ILSVRC2012_val_00000001.JPEG' -> label index
    using the torchvision ImageNet val dataset.
    """
    mapping = {}
    # ImageNet (torchvision) subclasses ImageFolder, so .samples exists
    for path, target in val_dataset.samples:
        fname = os.path.basename(path)
        mapping[fname] = int(target)
    print(f"[INFO] Built filename->label map for {len(mapping)} ImageNet val images.")
    return mapping


def evaluate_imagenet_c(
    model,
    imagenet_c_root,
    transform,
    device,
    filename_to_label,
    batch_size=32,
    num_workers=8,
    max_per_severity=1000,
):
    """
    Evaluate model on ImageNet-C across all corruptions and severities.

    Expected layout (what you showed):
      imagenet_c_root/
        gaussian_noise/
          1/
            n01440764/
              ILSVRC2012_val_00000001.JPEG, ...
            n01443537/
            ...
          2/
          ...
        shot_noise/
          ...
        ...

    We derive labels from *filenames* using filename_to_label from the clean val set.

    Speed knob:
      max_per_severity: if > 0, randomly subsample at most this many images
                        per (corruption, severity).
    """
    model.eval()

    results = {}
    mean_accs = []

    for corruption in IMAGENETC_CORRUPTIONS:
        corr_dir = os.path.join(imagenet_c_root, corruption)
        if not os.path.isdir(corr_dir):
            # Quietly skip completely missing corruptions
            continue

        for severity in range(1, 6):
            sev_dir = os.path.join(corr_dir, str(severity))
            if not os.path.isdir(sev_dir):
                continue

            # Collect all images under severity across all wnids
            img_paths = sorted(
                glob.glob(os.path.join(sev_dir, "*", "*.JPEG")) +
                glob.glob(os.path.join(sev_dir, "*", "*.jpg")) +
                glob.glob(os.path.join(sev_dir, "*", "*.png"))
            )

            if not img_paths:
                continue

            labels = []
            filtered_paths = []
            missing_fnames = []

            for p in img_paths:
                fname = os.path.basename(p)
                if fname in filename_to_label:
                    filtered_paths.append(p)
                    labels.append(filename_to_label[fname])
                else:
                    missing_fnames.append(fname)

            # Aggregate missing-filename warnings into a single line
            if missing_fnames:
                uniq = sorted(set(missing_fnames))
                show = ", ".join(uniq[:5])
                print(
                    f"[WARN] {corruption} severity {severity}: "
                    f"{len(uniq)} filenames not in val set mapping (showing up to 5): {show}"
                )

            if len(filtered_paths) == 0:
                print(f"[WARN] No valid images for {corruption} severity {severity}, skipping.")
                continue

            # Optional subsampling for speed
            if max_per_severity is not None and max_per_severity > 0 and len(filtered_paths) > max_per_severity:
                idx = np.random.choice(len(filtered_paths), size=max_per_severity, replace=False)
                filtered_paths = [filtered_paths[i] for i in idx]
                labels = [labels[i] for i in idx]

            labels = np.array(labels, dtype=np.int64)

            dataset = ImageNetCDataset(filtered_paths, labels, transform=transform)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            acc = test(model, loader, device)
            results[(corruption, severity)] = acc
            mean_accs.append(acc)

            print(
                f"[IMAGENET-C] corruption={corruption:>18s} "
                f"severity={severity} acc={acc:.2f}  (n={len(filtered_paths)})"
            )

    if len(mean_accs) > 0:
        mean_acc = float(np.mean(mean_accs))
    else:
        mean_acc = float("nan")

    print(f"[IMAGENET-C] MEAN_ACC={mean_acc:.2f}")
    return results, mean_acc


# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP ViT-B/32 compression & ImageNet-C robustness across ratios/checkpoints"
    )
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Directory with CLIP checkpoints (.pt)")
    parser.add_argument("--method", type=str, default="fold",
                        choices=["fold", "mag-l1", "mag-l2"],
                        help="Compression method: fold, mag-l1, mag-l2")
    parser.add_argument("--imagenet_root", type=str, default="../data",
                        help="Path to ImageNet root (clean and ImageNet-C under it)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--max_c_per_severity",
        type=int,
        default=1000,
        help="Max #images per (corruption, severity) for ImageNet-C (0 or negative = use all).",
    )
    args = parser.parse_args()

    fix_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 1000

    # Load checkpoints
    ckpt_paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pt")))
    if len(ckpt_paths) == 0:
        print(f"No checkpoints found in {args.ckpt_dir}")
        return

    prune_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Load preprocess from first checkpoint
    _, preprocess = load_clip_vit_model(num_classes, ckpt_paths[0], device)

    # Prepare dataset (clean ImageNet val only – skip train for speed)
    val_dataset = ImageNet(root=args.imagenet_root, split="val", transform=preprocess)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # Filename -> label mapping from clean val set
    filename_to_label = build_filename_to_label_map(val_dataset)

    # ImageNet-C root
    imagenet_c_root = os.path.join(args.imagenet_root, "ImageNet-C")
    quick_check_imagenet_c(imagenet_c_root)

    # Map pruning methods
    pruner_map = {
        "fold":   lambda m, r: CLIPViT_ModelFolding(m, compression_ratio=r),
        "mag-l1": lambda m, r: CLIPViT_MagnitudePruning(m, compression_ratio=r, p=1),
        "mag-l2": lambda m, r: CLIPViT_MagnitudePruning(m, compression_ratio=r, p=2),
    }

    for i, ckpt_path in enumerate(ckpt_paths):
        model_name = os.path.basename(ckpt_path)

        for ratio in prune_ratios:
            # Load model fresh per ratio
            model, _ = load_clip_vit_model(num_classes, ckpt_path, device)
            orig_params = count_parameters(model)

            # Baseline evaluation
            if ratio == 0.0:
                acc_clean = test(model, val_loader, device)
                if acc_clean < 10.0:  # sanity check threshold for ImageNet
                    print(f"Skipping model {model_name} (clean acc={acc_clean:.2f} < 10)")
                    break

                print(f"\n[MODEL] {i + 1}/{len(ckpt_paths)} {model_name}")
                log_line(ratio, "BASE_CLEAN", params=orig_params, acc=f"{acc_clean:.2f}")

                # Zero-shot ImageNet-C robustness (no adaptation on C)
                _, mean_acc_c = evaluate_imagenet_c(
                    model,
                    imagenet_c_root,
                    preprocess,
                    device,
                    filename_to_label,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    max_per_severity=args.max_c_per_severity if args.max_c_per_severity > 0 else None,
                )
                print(f"[IMAGENET-C_BASE] RATIO={ratio:.1f} MeanAcc={mean_acc_c:.2f}")
                continue

            # --------------------------------------------------------
            # Apply compression
            # --------------------------------------------------------
            pruner = pruner_map[args.method](model, ratio)
            model = pruner.apply().to(device)
            pruned_params = count_parameters(model)

            # Test after pruning on clean ImageNet
            acc_pruned = test(model, val_loader, device)
            log_line(ratio, "PRUNE_CLEAN", params=pruned_params, acc=f"{acc_pruned:.2f}")

            # --------------------------------------------------------
            # Zero-shot ImageNet-C robustness after compression
            # --------------------------------------------------------
            _, mean_acc_c = evaluate_imagenet_c(
                model,
                imagenet_c_root,
                preprocess,
                device,
                filename_to_label,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_per_severity=args.max_c_per_severity if args.max_c_per_severity > 0 else None,
            )
            print(f"[IMAGENET-C_COMPRESSED] RATIO={ratio:.1f} METHOD={args.method} MeanAcc={mean_acc_c:.2f}")


if __name__ == "__main__":
    main()
