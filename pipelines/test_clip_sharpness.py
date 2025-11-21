import os
import sys
import glob
import argparse
import random
import copy
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.clip_vit import CLIPViT_B32
from compression.fold import CLIPViT_ModelFolding
from compression.mag_prune import CLIPViT_MagnitudePruning
from utils.eval_utils import test, count_parameters
from utils.tune_utils import retune_layernorm  # not used but fine to keep
from utils.sharpness import random_init_lw, weight_ascent_step_momentum


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
# Wrapper to ensure model(x) -> ImageNet logits
# --------------------------------------------------------
class CLIPLogitsWrapper(nn.Module):
    """
    Wrap a CLIPViT model so that:
      - forward(x) returns 1000-dim logits for ImageNet,
      - .parameters() only includes the visual encoder + classification head,
        i.e., only parameters that actually influence the loss.
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()

        # Register only the parts used for classification.
        if hasattr(base_model, "visual"):
            self.visual = base_model.visual
        else:
            # Fallback: treat the whole model as the visual encoder
            self.visual = base_model

        if hasattr(base_model, "classification_head"):
            self.classification_head = base_model.classification_head
        else:
            # If there is no explicit classification head, just identity
            self.classification_head = nn.Identity()

    def forward(self, x):
        feats = self.visual(x)
        logits = self.classification_head(feats)
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits


# --------------------------------------------------------
# APGD ℓ∞-sharpness
# --------------------------------------------------------
def eval_APGD_sharpness(
        model, batches, loss_f, train_err, train_loss, rho=0.01,
        step_size_mult=1, n_iters=200, layer_name_pattern='all',
        n_restarts=1, min_update_ratio=0.75, rand_init=True,
        no_grad_norm=False, verbose=False, return_output=False,
        adaptive=False, version='default', norm='linf', **kwargs,
):
    """Computes worst-case sharpness for every batch independently, and returns
    the average values.
    """
    assert n_restarts == 1 or rand_init, 'Restarts need random init.'
    del train_err
    del train_loss
    gradient_step_kwargs = kwargs.get('gradient_step_kwargs', {})

    init_fn = partial(random_init_lw, norm=norm, adaptive=adaptive)

    def get_loss_and_err(m, loss_fn, x, y):
        """Compute loss and class. error on a single batch."""
        with torch.no_grad():
            out = m(x)  # CLIPLogitsWrapper will give logits here
            loss = loss_fn(out, y)
            err = (out.max(1)[1] != y).float().mean()
        return loss.cpu().item(), err.cpu().item()

    orig_model_state_dict = copy.deepcopy(model.state_dict())
    orig_param_dict = {param: param.clone() for param in model.parameters()}

    n_batches, delta_norm = 0, 0.
    avg_loss, avg_err, avg_init_loss, avg_init_err = 0., 0., 0., 0.
    output = ""

    if version == 'default':
        p = [0, 0.22]
        w = [0, math.ceil(n_iters * 0.22)]

        while w[-1] < n_iters and w[-1] != w[-2]:
            p.append(p[-1] + max(p[-1] - p[-2] - 0.03, 0.06))
            w.append(math.ceil(p[-1] * n_iters))

        w = w[1:]  # No check needed at the first iteration.
        step_size_scaler = .5
    else:
        raise ValueError(f'Unknown version {version}')

    device = next(model.parameters()).device

    for i_batch, batch in enumerate(batches):
        if len(batch) == 2:
            x, y = batch
        else:
            x, _, y, _, _ = batch

        x = x.to(device)
        y = y.to(device)

        # Loss and err on the unperturbed model.
        init_loss, init_err = get_loss_and_err(model, loss_f, x, y)

        # Accumulate over batches.
        avg_init_loss += init_loss
        avg_init_err += init_err

        worst_loss_over_restarts = init_loss
        worst_err_over_restarts = init_err
        worst_delta_norm_over_restarts = 0.

        for restart in range(n_restarts):
            if rand_init:
                delta_dict = {param: torch.zeros_like(param) for param in model.parameters()}
                delta_dict = init_fn(delta_dict, rho, orig_param_dict=orig_param_dict)
                for param in model.parameters():
                    param.data += delta_dict[param]
            else:
                delta_dict = {param: torch.zeros_like(param) for param in model.parameters()}

            prev_delta_dict = {param: delta_dict[param].clone() for param in delta_dict}
            worst_model_dict = copy.deepcopy(model.state_dict())

            prev_worst_loss, worst_loss = init_loss, init_loss
            worst_err = init_err
            step_size, prev_step_size = 2 * rho * step_size_mult, 2 * rho * step_size_mult
            prev_cp = 0
            num_of_updates = 0

            for i in range(n_iters):

                delta_dict, prev_delta_dict = weight_ascent_step_momentum(
                    model, x, y, loss_f, orig_param_dict, delta_dict, prev_delta_dict,
                    step_size, rho, momentum=0.75, layer_name_pattern=layer_name_pattern,
                    no_grad_norm=no_grad_norm, verbose=False, adaptive=adaptive,
                    norm=norm, **gradient_step_kwargs)

                with torch.no_grad():
                    curr_loss, curr_err = get_loss_and_err(model, loss_f, x, y)
                    delta_norm_total = torch.cat(
                        [delta_param.flatten() for delta_param in delta_dict.values()]
                    ).norm().item()

                    if curr_loss > worst_loss:
                        worst_loss = curr_loss
                        worst_err = curr_err
                        worst_model_dict = copy.deepcopy(model.state_dict())
                        worst_delta_norm = delta_norm_total
                        num_of_updates += 1

                    # Step-size adaptation
                    if i in w:
                        cond1 = num_of_updates < (min_update_ratio * (i - prev_cp))
                        cond2 = (prev_step_size == step_size) and (prev_worst_loss == worst_loss)
                        prev_step_size, prev_worst_loss, prev_cp = step_size, worst_loss, i
                        num_of_updates = 0

                        if cond1 or cond2:
                            step_size *= step_size_scaler
                            model.load_state_dict(worst_model_dict)

                if verbose:
                    str_to_log = (
                        f"[batch={i_batch + 1} iter={i + 1}] "
                        f"Sharpness: obj={curr_loss - init_loss:.4f}, "
                        f"err={curr_err - init_err:.2%}, "
                        f"delta_norm={delta_norm_total:.5f} (step={step_size:.5f})"
                    )
                    print(str_to_log)
                    output += str_to_log + '\n'

            # Keep the best values over restarts.
            if worst_loss > worst_loss_over_restarts:
                worst_loss_over_restarts = worst_loss
                worst_err_over_restarts = worst_err
                worst_delta_norm_over_restarts = worst_delta_norm

            # ✅ Reload the unperturbed model for the next restart or batch.
            model.load_state_dict(orig_model_state_dict)

            if verbose:
                print('')

        # Accumulate over batches.
        n_batches += 1
        avg_loss += worst_loss_over_restarts
        avg_err += worst_err_over_restarts
        delta_norm = max(delta_norm, worst_delta_norm_over_restarts)

        if verbose:
            print('')

    vals = (
        (avg_loss - avg_init_loss) / n_batches,
        (avg_err - avg_init_err) / n_batches,
        delta_norm,
    )
    if return_output:
        vals += (output,)

    return vals


def _limited_batches(loader, max_batches):
    """
    Yield at most `max_batches` batches from a DataLoader.
    This is the main knob to make sharpness computation tractable.
    """
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        yield batch


def compute_sharpness_for_rhos(
        model,
        train_loader,
        loss_fn,
        rho_values,
        n_iters=10,
        max_batches=8,
        layer_name_pattern="all",
):
    """
    Helper: returns list of sharpness values (one per rho).

    Speed optimizations:
      - only uses the first `max_batches` batches of train_loader
      - passes `layer_name_pattern` down to gradient steps to optionally
        restrict to a subset of layers
    """
    sharpness_vals = []
    for rho in rho_values:
        sharp, _, _ = eval_APGD_sharpness(
            model,
            _limited_batches(train_loader, max_batches),
            loss_f=loss_fn,
            train_err=None,
            train_loss=None,
            rho=rho,
            n_iters=n_iters,
            adaptive=True,
            norm='linf',
            n_restarts=1,
            layer_name_pattern=layer_name_pattern,
        )
        sharpness_vals.append(sharp)
    return sharpness_vals


# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP ViT-B/32 compression & APGD sharpness across ratios/checkpoints"
    )
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Directory with CLIP checkpoints (.pt)")
    parser.add_argument("--method", type=str, default="fold",
                        choices=["fold", "mag-l1", "mag-l2"])
    parser.add_argument("--imagenet_root", type=str, default="../data",
                        help="Path to ImageNet root")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sharp-iters", type=int, default=5,
                        help="PGD iterations per batch for sharpness (reduced for speed)")
    parser.add_argument("--sharp-max-batches", type=int, default=8,
                        help="Max number of train batches to use for sharpness")
    parser.add_argument("--sharp-layer-pattern", type=str, default="all",
                        help="Regex / pattern for layer names in weight_ascent_step_momentum "
                             "(e.g. 'visual.transformer.resblocks.11' to restrict to last block)")
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
    rho_values = [0.0001, 0.0005, 0.001]

    # Load preprocess from first checkpoint
    _, preprocess = load_clip_vit_model(num_classes, ckpt_paths[0], device)

    # Prepare datasets
    train_dataset = ImageNet(root=args.imagenet_root, split="train", transform=preprocess)
    val_dataset = ImageNet(root=args.imagenet_root, split="val", transform=preprocess)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=8, pin_memory=True
    )

    # Map pruning methods
    pruner_map = {
        "fold":   lambda m, r: CLIPViT_ModelFolding(m, compression_ratio=r),
        "mag-l1": lambda m, r: CLIPViT_MagnitudePruning(m, compression_ratio=r, p=1),
        "mag-l2": lambda m, r: CLIPViT_MagnitudePruning(m, compression_ratio=r, p=2),
    }

    loss_fn = nn.CrossEntropyLoss()

    for i, ckpt_path in enumerate(ckpt_paths):
        model_name = os.path.basename(ckpt_path)

        for ratio in prune_ratios:
            # Load model fresh per ratio
            model, _ = load_clip_vit_model(num_classes, ckpt_path, device)
            orig_params = count_parameters(model)

            # Baseline evaluation
            if ratio == 0.0:
                acc = test(model, val_loader, device)
                if acc < 10.0:  # sanity check threshold for ImageNet
                    print(f"Skipping model {model_name} (baseline acc={acc:.2f} < 10)")
                    break
                print(f"\n[MODEL] {i + 1}/{len(ckpt_paths)} {model_name}")
                log_line(ratio, "BASE", params=orig_params, acc=f"{acc:.2f}")
                continue

            # --------------------------------------------------------
            # Apply compression
            # --------------------------------------------------------
            pruner = pruner_map[args.method](model, ratio)
            model = pruner.apply().to(device)
            pruned_params = count_parameters(model)

            # Test after pruning
            acc_pruned = test(model, val_loader, device)
            log_line(ratio, "PRUNE", params=pruned_params, acc=f"{acc_pruned:.2f}")

            # --------------------------------------------------------
            # Sharpness (APGD ℓ∞, multiple ρ)
            # Use logits wrapper so model(x) -> 1000-d logits,
            # and parameters() only returns visual + head.
            # --------------------------------------------------------
            sharp_model = CLIPLogitsWrapper(model).to(device)

            sharp_vals = compute_sharpness_for_rhos(
                sharp_model,
                train_loader,
                loss_fn,
                rho_values=rho_values,
                n_iters=args.sharp_iters,
                max_batches=args.sharp_max_batches,
                layer_name_pattern=args.sharp_layer_pattern,
            )
            sharp_str = "|".join(f"{s:.4f}" for s in sharp_vals)
            print(
                f"[SHARPNESS] RATIO={ratio:.1f} METHOD={args.method} "
                f"batches={args.sharp_max_batches} iters={args.sharp_iters} Sharpness={sharp_str}"
            )


if __name__ == "__main__":
    main()
