import os
import sys
import time
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

# -------------------------------------------------------------------------
# Optional FLOPs dependency (thop)
# -------------------------------------------------------------------------
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
CHECKPOINT_PATH = "../checkpoints/clipvit-b32-model-soups/model_1.pt"
BATCH_SIZE = 32
COMPRESSION_RATIO = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLIP ViT-B/32 default image size (adjust if your CLIP wrapper uses a different crop)
CLIP_INPUT_SIZE = (3, 224, 224)  # (C, H, W)


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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


def build_model(device=DEVICE, num_classes=1000):
    model, _ = load_clip_vit_model(num_classes, CHECKPOINT_PATH, device)
    return model


# -----------------------------------------------------------------------------
# Resource measurement helpers (inference)
# -----------------------------------------------------------------------------
def measure_resources(
    model,
    device,
    input_size=(3, 224, 224),
    batch_size=32,
    num_warmup=5,
    num_iters=20,
):
    """
    Measure latency, FLOPs, and peak memory for a forward pass.

    Peak memory is measured with cuDNN disabled to avoid counting
    large backend workspaces; this makes numbers comparable across
    models and shapes.
    """
    model.to(device)
    model.eval()
    C, H, W = input_size

    # ---------------- Latency ----------------
    x = torch.randn(batch_size, C, H, W, device=device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency_per_batch_ms = (elapsed / num_iters) * 1000.0
    latency_per_image_ms = latency_per_batch_ms / batch_size

    # ---------------- FLOPs ----------------
    flops_per_image = None
    flops_ok = False
    flops_msg = ""

    if not HAS_THOP:
        flops_msg = "thop not installed (pip install thop)."
    else:
        try:
            x_flop = torch.randn(1, C, H, W, device=device)
            model.eval()
            with torch.no_grad():
                flops, params = profile(model, inputs=(x_flop,), verbose=False)
            flops_per_image = float(flops)  # for one image
            flops_ok = True
            flops_msg = "OK"
        except Exception as e:
            flops_per_image = None
            flops_ok = False
            flops_msg = f"thop.profile failed: {repr(e)}"

    # ---------------- Peak memory (GPU only, cuDNN disabled) ----------------
    peak_memory_mb = None
    if device.type == "cuda":
        # Make sure no stale allocations are counted
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # Temporarily disable cuDNN to avoid counting its workspace
        prev_cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        x_mem = torch.randn(batch_size, C, H, W, device=device)
        with torch.no_grad():
            _ = model(x_mem)
        torch.cuda.synchronize()

        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory_bytes / (1024 ** 2)

        # Restore cuDNN state
        torch.backends.cudnn.enabled = prev_cudnn_enabled

    return {
        "latency_ms_per_batch": latency_per_batch_ms,
        "latency_ms_per_image": latency_per_image_ms,
        "flops_per_image": flops_per_image,
        "flops_ok": flops_ok,
        "flops_msg": flops_msg,
        "peak_memory_mb": peak_memory_mb,
    }



def print_resources(name, resources):
    print(f"\n=== Resource usage: {name} ===")
    print(
        f"Latency: {resources['latency_ms_per_batch']:.3f} ms / batch, "
        f"{resources['latency_ms_per_image']:.4f} ms / image"
    )

    if resources["flops_ok"]:
        print(
            f"FLOPs: {resources['flops_per_image'] / 1e6:.2f} MFLOPs / image "
            f"(thop: {resources['flops_msg']})"
        )
    else:
        print(f"FLOPs: unavailable ({resources['flops_msg']})")

    if resources["peak_memory_mb"] is not None:
        print(f"Peak memory (forward): {resources['peak_memory_mb']:.2f} MB")
    else:
        print("Peak memory: only measured on CUDA.")


# -----------------------------------------------------------------------------
# Resource measurement helpers (compression)
# -----------------------------------------------------------------------------
def measure_compression(
    make_pruner,
    build_model_fn,
    device,
    num_classes=1000,
):
    """
    Measure compression time and peak memory during compression (pruner.apply).

    Returns:
        pruned_model, comp_stats

        comp_stats keys:
            - compression_time_s
            - compression_peak_memory_mb  (None on CPU)
    """
    model = build_model_fn(device=device, num_classes=num_classes)

    # Track GPU memory during compression
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    pruner = make_pruner(model)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    pruned_model = pruner.apply()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    comp_time = t1 - t0

    comp_peak_mem_mb = None
    if device.type == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated(device)
        comp_peak_mem_mb = peak_bytes / (1024 ** 2)

    return pruned_model, {
        "compression_time_s": comp_time,
        "compression_peak_memory_mb": comp_peak_mem_mb,
    }



def print_compression_stats(name, stats):
    print(f"\n>>> Compression stats for method '{name}':")
    print(f"Compression time: {stats['compression_time_s']:.3f} s")
    if stats["compression_peak_memory_mb"] is not None:
        print(f"Peak memory during compression: {stats['compression_peak_memory_mb']:.2f} MB")
    else:
        print("Peak memory during compression: only measured on CUDA.")


# -----------------------------------------------------------------------------
# Main CLIP profiling
# -----------------------------------------------------------------------------
def main():
    num_classes = 1000  # ImageNet classes

    # Load model & preprocess once for baseline + val loader
    base_model, preprocess = load_clip_vit_model(num_classes, CHECKPOINT_PATH, DEVICE)

    # Prepare validation loader
    val_dataset = ImageNet(root="../data", split="val", transform=preprocess)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Evaluate BEFORE compression (optional, can be slow on full val set)
    print("\n=== Evaluation BEFORE compression ===")
    # acc_before = test(base_model, val_loader, device=DEVICE)
    # print(f"🔹 Top-1 Accuracy: {acc_before:.2f}%")
    original_params = count_parameters(base_model)
    print(f"Original Parameters: {original_params}")

    base_res = measure_resources(
        base_model,
        device=DEVICE,
        input_size=CLIP_INPUT_SIZE,
        batch_size=BATCH_SIZE,
        num_warmup=3,
        num_iters=10,
    )
    print_resources("CLIP ViT-B/32 (original)", base_res)

    # Move baseline off GPU so it doesn't affect later measurements
    if DEVICE.type == "cuda":
        base_model.to("cpu")
        torch.cuda.empty_cache()
    del base_model

    # -------------------------------------------------------------------------
    # Define methods
    # -------------------------------------------------------------------------
    METHODS = {
        "fold": lambda m: CLIPViT_ModelFolding(m, compression_ratio=COMPRESSION_RATIO),
        "mag":  lambda m: CLIPViT_MagnitudePruning(m, compression_ratio=COMPRESSION_RATIO, p=2),
    }

    # -------------------------------------------------------------------------
    # Run all methods
    # -------------------------------------------------------------------------
    for name, make_pruner in METHODS.items():
        print("\n" + "=" * 80)
        print(f"=== Method: {name} ===")

        # Clean GPU state *before* building/compressing this method
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(DEVICE)

        # ---------------- Compression profiling ----------------
        pruned_model, comp_stats = measure_compression(
            make_pruner=make_pruner,
            build_model_fn=build_model,
            device=DEVICE,
            num_classes=num_classes,
        )
        pruned_model = pruned_model.to(DEVICE)

        print_compression_stats(name, comp_stats)

        # ---------------- Evaluation AFTER compression (before LN re-tune) ----
        print("\n=== Evaluation AFTER compression (before LN re-tune) ===")
        pruned_params = count_parameters(pruned_model)
        acc_after = test(pruned_model, val_loader, device=DEVICE)
        print(f"🔹 Top-1 Accuracy: {acc_after:.2f}%")
        print(f"Pruned Parameters: {pruned_params}")
        print(f"🔥 Compression Ratio: {(original_params - pruned_params) / original_params:.2%}")

        # ---------------- Resource usage AFTER compression --------------------
        pruned_res = measure_resources(
            pruned_model,
            device=DEVICE,
            input_size=CLIP_INPUT_SIZE,
            batch_size=BATCH_SIZE,
            num_warmup=3,
            num_iters=10,
        )
        print_resources(f"CLIP ViT-B/32 (compressed, method={name}, before LN re-tune)", pruned_res)

        # ---------------- Optional: Re-tune LayerNorm -------------------------
        # print("\n[INFO] Re-tuning LayerNorm parameters...")
        # retune_layernorm(pruned_model, val_loader, device=DEVICE, lr=1e-4)
        #
        # print("\n=== Evaluation AFTER compression (after LN re-tune) ===")
        # acc_after_ln = test(pruned_model, val_loader, device=DEVICE)
        # print(f"🔹 Top-1 Accuracy (after LN re-tune): {acc_after_ln:.2f}%")
        # print(f"Pruned Parameters: {pruned_params}")
        # print(f"🔥 Compression Ratio: {(original_params - pruned_params) / original_params:.2%}")

        # Move this pruned model off GPU before next method (safety)
        if DEVICE.type == "cuda":
            pruned_model.to("cpu")
            torch.cuda.empty_cache()
        del pruned_model


if __name__ == "__main__":
    main()
