import os
import sys
import time
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn  # === NEW: for type checks ===
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from collections import OrderedDict  # === NEW: for deterministic layer order ===

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
# Per-layer stats: params, FLOPs, activations (same idea as for PreActResNet18)
# -----------------------------------------------------------------------------
def _is_counted_module(m):
    """Count Conv2d and Linear layers (ViT mainly uses Linear; Conv2d for patch embedding)."""
    return isinstance(m, (nn.Conv2d, nn.Linear))


def collect_layer_stats(model, device, input_size=(3, 224, 224)):
    """
    Collect per-layer statistics for Conv2d / Linear layers:
    - params: # learnable parameters
    - flops: MACs for one image (no factor 2 for mul+add)
    - act_elems: number of output elements
    - act_nonzero: number of non-zero output activations ("effective activations")

    Works for CLIP ViT:
      * Conv2d: patch embedding
      * Linear: attention projections, MLPs, classification head, etc.
    """
    model = model.to(device)
    model.eval()

    C, H, W = input_size
    x = torch.randn(1, C, H, W, device=device)  # single image

    layer_stats = OrderedDict()
    handles = []

    # Pre-fill parameter counts
    for name, module in model.named_modules():
        if _is_counted_module(module):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            layer_stats[name] = {
                "params": params,
                "flops": 0,
                "act_elems": 0,
                "act_nonzero": 0,
            }

    def make_hook(layer_name):
        def hook(module, inp, out):
            # out can be Tensor or tuple
            if isinstance(out, (tuple, list)):
                out_tensor = out[0]
            else:
                out_tensor = out

            info = layer_stats[layer_name]

            # activations
            act_elems = out_tensor.numel()
            act_nonzero = (out_tensor != 0).sum().item()
            info["act_elems"] = act_elems
            info["act_nonzero"] = act_nonzero

            # FLOPs (MACs) per image
            if isinstance(module, nn.Conv2d):
                in_tensor = inp[0]
                out_c, out_h, out_w = out_tensor.shape[1:]
                kernel_h, kernel_w = module.kernel_size
                in_c = in_tensor.shape[1]
                groups = module.groups
                # MACs = out_c * out_h * out_w * (in_c / groups) * k_h * k_w
                flops = (
                    out_c
                    * out_h
                    * out_w
                    * (in_c // groups)
                    * kernel_h
                    * kernel_w
                )
            elif isinstance(module, nn.Linear):
                # For ViT, Linear is applied per token: multiply by #token positions.
                # Batch size is 1 here, so positions = act_elems / out_features.
                out_features = module.out_features
                n_positions = act_elems // out_features if out_features > 0 else 1
                flops = module.in_features * module.out_features * n_positions
            else:
                flops = 0

            info["flops"] = flops

        return hook

    # Register hooks
    for name, module in model.named_modules():
        if _is_counted_module(module) and name in layer_stats:
            h = module.register_forward_hook(make_hook(name))
            handles.append(h)

    # Run one forward pass to populate stats
    with torch.no_grad():
        _ = model(x)

    # Clean hooks
    for h in handles:
        h.remove()

    return layer_stats


def compare_layer_stats(stats_a, stats_b, name_a="fold", name_b="mag"):
    """
    Compare per-layer stats between two methods and print a table:
    - params
    - FLOPs
    - activation elements
    - non-zero activations (effective activations)
    """
    print("\n" + "=" * 80)
    print(f"PER-LAYER COMPARISON: {name_a} vs {name_b} (CLIP ViT-B/32)")
    print("=" * 80)

    all_layers = sorted(set(stats_a.keys()) | set(stats_b.keys()))

    header = (
        f"{'Layer':60s}"
        f"{'params_'+name_a:>12s}{'params_'+name_b:>12s}{'Δp':>8s}"
        f"{'FLOPs_'+name_a:>16s}{'FLOPs_'+name_b:>16s}{'ΔF':>12s}"
        f"{'Act_'+name_a:>14s}{'Act_'+name_b:>14s}{'Δa':>12s}"
        f"{'NZ_'+name_a:>14s}{'NZ_'+name_b:>14s}{'Δnz':>12s}"
    )
    print(header)
    print("-" * len(header))

    total_params_a = total_params_b = 0
    total_flops_a = total_flops_b = 0
    total_act_a = total_act_b = 0
    total_nz_a = total_nz_b = 0

    for layer in all_layers:
        sa = stats_a.get(layer)
        sb = stats_b.get(layer)

        if sa is None or sb is None:
            print(f"{layer:60s}  MISSING IN ONE MODEL")
            continue

        pa, pb = sa["params"], sb["params"]
        fa, fb = sa["flops"], sb["flops"]
        aa, ab = sa["act_elems"], sb["act_elems"]
        nza, nzb = sa["act_nonzero"], sb["act_nonzero"]

        total_params_a += pa
        total_params_b += pb
        total_flops_a += fa
        total_flops_b += fb
        total_act_a += aa
        total_act_b += ab
        total_nz_a += nza
        total_nz_b += nzb

        print(
            f"{layer:60s}"
            f"{pa:12d}{pb:12d}{(pa-pb):8d}"
            f"{fa:16d}{fb:16d}{(fa-fb):12d}"
            f"{aa:14d}{ab:14d}{(aa-ab):12d}"
            f"{nza:14d}{nzb:14d}{(nza-nzb):12d}"
        )

    print("-" * len(header))
    print(
        "TOTALS".ljust(60)
        + f"{total_params_a:12d}{total_params_b:12d}{(total_params_a-total_params_b):8d}"
        + f"{total_flops_a:16d}{total_flops_b:16d}{(total_flops_a-total_flops_b):12d}"
        + f"{total_act_a:14d}{total_act_b:14d}{(total_act_a-total_act_b):12d}"
        + f"{total_nz_a:14d}{total_nz_b:14d}{(total_nz_a-total_nz_b):12d}"
    )

    # Quantify impact of any mismatches
    if total_params_a != total_params_b:
        rel = abs(total_params_a - total_params_b) / max(total_params_a, total_params_b)
        print(f"\n[INFO] Parameter mismatch: {total_params_a} vs {total_params_b} "
              f"({rel*100:.6f}% relative diff).")
    if total_flops_a != total_flops_b:
        rel = abs(total_flops_a - total_flops_b) / max(total_flops_a, total_flops_b)
        print(f"[INFO] FLOPs mismatch: {total_flops_a} vs {total_flops_b} "
              f"({rel*100:.6f}% relative diff).")
    if total_act_a != total_act_b:
        rel = abs(total_act_a - total_act_b) / max(total_act_a, total_act_b)
        print(f"[INFO] Activation-size mismatch: {total_act_a} vs {total_act_b} "
              f"({rel*100:.6f}% relative diff).")
    if total_nz_a != total_nz_b:
        rel = abs(total_nz_a - total_nz_b) / max(total_nz_a, total_nz_b)
        print(f"[INFO] Effective-activation mismatch (non-zero): {total_nz_a} vs {total_nz_b} "
              f"({rel*100:.6f}% relative diff).")

    if (
        total_params_a == total_params_b
        and total_flops_a == total_flops_b
        and total_act_a == total_act_b
        and total_nz_a == total_nz_b
    ):
        print("\n[OK] Parameters, FLOPs, activations, and effective activations "
              "are exactly matched globally for CLIP ViT-B/32.")


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
    # print(f"Top-1 Accuracy: {acc_before:.2f}%")
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

    # Store per-layer stats for each method (for reviewer table)
    per_method_layer_stats = {}

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

        # ---------------- Per-layer stats AFTER compression ------------------
        layer_stats = collect_layer_stats(
            pruned_model,
            device=DEVICE,
            input_size=CLIP_INPUT_SIZE,
        )
        per_method_layer_stats[name] = layer_stats

        # ---------------- Evaluation AFTER compression (before LN re-tune) ----
        print("\n=== Evaluation AFTER compression (before LN re-tune) ===")
        pruned_params = count_parameters(pruned_model)
        acc_after = test(pruned_model, val_loader, device=DEVICE)
        print(f"Top-1 Accuracy: {acc_after:.2f}%")
        print(f"Pruned Parameters: {pruned_params}")
        print(f"Compression Ratio: {(original_params - pruned_params) / original_params:.2%}")

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
        # print(f"Top-1 Accuracy (after LN re-tune): {acc_after_ln:.2f}%")
        # print(f"Pruned Parameters: {pruned_params}")
        # print(f"Compression Ratio: {(original_params - pruned_params) / original_params:.2%}")

        # Move this pruned model off GPU before next method (safety)
        if DEVICE.type == "cuda":
            pruned_model.to("cpu")
            torch.cuda.empty_cache()
        del pruned_model

    # -------------------------------------------------------------------------
    # Per-layer comparison between folding and magnitude pruning (CLIP)
    # -------------------------------------------------------------------------
    if "fold" in per_method_layer_stats and "mag" in per_method_layer_stats:
        compare_layer_stats(
            per_method_layer_stats["fold"],
            per_method_layer_stats["mag"],
            name_a="fold",
            name_b="mag",
        )
    else:
        print("\n[WARN] Missing per-layer stats for 'fold' or 'mag'; "
              "cannot perform per-layer comparison for CLIP.")


if __name__ == "__main__":
    main()
