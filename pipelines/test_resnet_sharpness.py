import os
import sys
import glob
import argparse
import random
import copy
import math
from functools import partial
from itertools import islice  # <--- added

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.resnet import ResNet18
from compression.fold import ResNet18_ModelFolding
from compression.mag_prune import ResNet18_MagnitudePruning
from utils.eval_utils import test, count_parameters
from utils.tune_utils import repair_bn

# ------------------------------------------------------------------------
# Import / define APGD helpers (adapt to your project structure if needed)
# ------------------------------------------------------------------------
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
    """Unified concise logging (same style as PreActResNet18 script)."""
    parts = [f"[BASE] Ratio={ratio:.1f}", f"Eval={event}"]
    if "Params" in kwargs:
        parts.insert(1, f"Params={kwargs.pop('Params')}")
    if "RESULT" in kwargs:
        # pass already formatted RESULT string
        parts.append(f"[RESULT] {kwargs.pop('RESULT')}")
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


def get_dataloaders(data_root="../data", batch_size=128, num_workers=8):
    train_ds = datasets.CIFAR10(data_root, train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(data_root, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# --------------------------------------------------------
# Model
# --------------------------------------------------------
def load_resnet18_model(num_classes, checkpoint_path, device):
    model = ResNet18(num_classes=num_classes).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    # Handle checkpoints potentially saved with key 'last'
    if isinstance(state_dict, dict) and 'last' in state_dict:
        model.load_state_dict(state_dict['last'])
    else:
        model.load_state_dict(state_dict)
    return model


# --------------------------------------------------------
# APGD ℓ∞-sharpness (same implementation as in PreActResNet18 script)
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
            output = m(x)
            loss = loss_fn(output, y)
            err = (output.max(1)[1] != y).float().mean()
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
                        [delta_param.flatten() for delta_param in delta_dict.values()]).norm().item()

                    if curr_loss > worst_loss:
                        worst_loss = curr_loss
                        worst_err = curr_err
                        worst_model_dict = copy.deepcopy(model.state_dict())
                        worst_delta_norm = delta_norm_total
                        num_of_updates += 1

                    if i in w:
                        cond1 = num_of_updates < (min_update_ratio * (i - prev_cp))
                        cond2 = (prev_step_size == step_size) and (prev_worst_loss == worst_loss)
                        prev_step_size, prev_worst_loss, prev_cp = step_size, worst_loss, i
                        num_of_updates = 0

                        if cond1 or cond2:
                            step_size *= step_size_scaler
                            model.load_state_dict(worst_model_dict)

                str_to_log = (
                    f"[batch={i_batch + 1} iter={i + 1}] "
                    f"Sharpness: obj={curr_loss - init_loss:.4f}, "
                    f"err={curr_err - init_err:.2%}, "
                    f"delta_norm={delta_norm_total:.5f} (step={step_size:.5f})"
                )
                if verbose:
                    print(str_to_log)
                output += str_to_log + '\n'

            # Keep the best values over restarts.
            if worst_loss > worst_loss_over_restarts:
                worst_loss_over_restarts = worst_loss
                worst_err_over_restarts = worst_err
                worst_delta_norm_over_restarts = worst_delta_norm

            # Reload the unperturbed model for the next restart or batch.
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


def compute_sharpness_for_rhos(
        model,
        train_loader,
        loss_fn,
        rho_values,
        n_iters=10,
        max_batches=20,  # <--- NEW: limit number of batches for sharpness
):
    """Helper: returns list of sharpness values (one per rho), using only a
    subset of batches for efficiency.
    """
    sharpness_vals = []
    for rho in rho_values:
        # Take only first `max_batches` batches from the train_loader
        batches = islice(train_loader, max_batches)
        sharp, _, _ = eval_APGD_sharpness(
            model, batches, loss_fn,
            train_err=None, train_loss=None,
            rho=rho, n_iters=n_iters,
            adaptive=True, norm='linf', n_restarts=1
        )
        sharpness_vals.append(sharp)
    return sharpness_vals


# --------------------------------------------------------
# Main
# --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ResNet18 compression & APGD sharpness across ratios/checkpoints"
    )
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--method", type=str, default="fold",
                        choices=["fold", "mag-l1", "mag-l2"])
    parser.add_argument("--sharp-iters", type=int, default=10,
                        help="Number of APGD iterations per batch for sharpness")
    parser.add_argument("--data-root", type=str, default="../data")
    parser.add_argument("--max-sharp-batches", type=int, default=20,
                        help="Max number of train batches used for sharpness")
    args = parser.parse_args()

    fix_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    train_loader, test_loader = get_dataloaders(data_root=args.data_root)

    # Checkpoints
    ckpt_paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pth")))
    if len(ckpt_paths) == 0:
        print(f"No checkpoints found in {args.ckpt_dir}")
        return

    # Compression & sharpness setup
    compression_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    prune_ratios = [0.0] + compression_ratios  # include baseline
    rho_values = [0.0001, 0.0005, 0.001]

    pruner_map = {
        "fold":   lambda m, r: ResNet18_ModelFolding(m, compression_ratio=r),
        "mag-l1": lambda m, r: ResNet18_MagnitudePruning(m, compression_ratio=r, p=1),
        "mag-l2": lambda m, r: ResNet18_MagnitudePruning(m, compression_ratio=r, p=2),
    }

    loss_fn = nn.CrossEntropyLoss()

    for i, ckpt_path in enumerate(ckpt_paths):
        model_name = os.path.basename(ckpt_path)
        print(f"\n{model_name} ========")

        # We reload per ratio so each compression starts from a clean model
        for ratio in prune_ratios:
            # Load fresh model per ratio
            model = load_resnet18_model(10, ckpt_path, device)
            orig_params = count_parameters(model)

            # Baseline (only for ratio=0.0)
            if ratio == 0.0:
                acc = test(model, test_loader, device)
                # skip terrible models
                if acc < 50.0:
                    print(f"Skipping model {model_name} (baseline acc={acc:.2f} < 50)")
                    break

                print(f"[MODEL] {i + 1}/{len(ckpt_paths)} {model_name}")
                log_line(
                    ratio,
                    "BeforePruning",
                    Params=orig_params,
                    RESULT=f"TestAcc={acc:.2f}"
                )
                continue

            # --------------------------------------------------------
            # Apply compression (fold / mag-l1 / mag-l2)
            # --------------------------------------------------------
            pruner = pruner_map[args.method](model, ratio)
            model = pruner.apply().to(device)
            pruned_params = count_parameters(model)

            # Eval immediately after pruning/folding
            acc_pruned = test(model, test_loader, device)
            log_line(
                ratio,
                "AfterPruning",
                Params=pruned_params,
                RESULT=f"TestAcc={acc_pruned:.2f}"
            )

            # --------------------------------------------------------
            # Repair BN
            # --------------------------------------------------------
            repair_bn(model, train_loader)
            acc_repair = test(model, test_loader, device)
            log_line(
                ratio,
                "AfterRepair",
                Params=pruned_params,
                RESULT=f"TestAcc={acc_repair:.2f}"
            )

            # --------------------------------------------------------
            # Sharpness (APGD ℓ∞, multiple ρ), using only a subset of batches
            # --------------------------------------------------------
            sharp_vals = compute_sharpness_for_rhos(
                model,
                train_loader,
                loss_fn,
                rho_values=rho_values,
                n_iters=args.sharp_iters,
                max_batches=args.max_sharp_batches,
            )
            sharp_str = "|".join(f"{s:.4f}" for s in sharp_vals)
            print(f"[SHARPNESS] Ratio={ratio:.1f} Method={args.method} Sharpness={sharp_str}")


if __name__ == "__main__":
    main()
