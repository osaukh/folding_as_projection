import torch
import torch.nn as nn
from collections import defaultdict

from compression.base_resnet import BaseResNetCompression

import torch
import torch.nn as nn
from compression.base_resnet import BaseResNetCompression

class ResNet18_WandaPruning(BaseResNetCompression):
    """
    ResNet-18 structured channel pruning with Wanda scores.
    - Same layer-by-layer flow and ratios as your magnitude version.
    - Only the scores change: score_out[j] = sum_i ( ||W_{j,i,...}||_1 * s_in[i] ),
      where s_in[i] is the L2 norm of input activation channel i (from calibration).
    """

    def __init__(self, model, min_channels=1, compression_ratio=0.5):
        super().__init__(model, min_channels, compression_ratio)
        self.act_scales = {}   # layer_name -> 1D tensor [Cin] of L2 norms
        self._hooks = []

    # -------- Calibration (collect per-input-channel L2 norms) --------
    def _register_activation_hooks(self):
        self._act_sum_sq = {}

        def make_hook(name):
            def hook(mod, inputs):
                x = inputs[0]
                with torch.no_grad():
                    if isinstance(mod, nn.Conv2d):
                        # x: [B, Cin, H, W] -> per-Cin sum of squares
                        ss = (x.float() ** 2).sum(dim=(0, 2, 3)).detach().cpu()
                    elif isinstance(mod, nn.Linear):
                        # x: [B, Cin]
                        ss = (x.float() ** 2).sum(dim=0).detach().cpu()
                    else:
                        return
                    self._act_sum_sq[name] = self._act_sum_sq.get(name, 0)
                    self._act_sum_sq[name] = self._act_sum_sq[name] + ss
            return hook

        for lname, layer in self.model_layers.items():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                self._hooks.append(layer.register_forward_pre_hook(make_hook(lname)))

    def _clear_hooks(self):
        for h in self._hooks:
            try: h.remove()
            except: pass
        self._hooks = []

    @torch.no_grad()
    def run_calibration(self, dataloader, device, num_batches=10):
        self.model.eval().to(device)
        self._register_activation_hooks()
        seen = 0
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            _ = self.model(x.to(device))
            seen += 1
            if seen >= num_batches:
                break
        self._clear_hooks()
        # finalize L2 norms (sqrt of sum of squares)
        self.act_scales = {name: ss.sqrt() for name, ss in self._act_sum_sq.items()}

    # ---------------- Wanda scoring helpers ----------------
    @staticmethod
    def _wanda_scores_conv(weight: torch.Tensor, s_in: torch.Tensor):
        # weight: [Cout, Cin, kH, kW], s_in: [Cin]
        w_abs_sum = weight.abs().view(weight.size(0), weight.size(1), -1).sum(dim=2)  # [Cout, Cin]
        return torch.matmul(w_abs_sum, s_in.to(weight.device))                         # [Cout]

    @staticmethod
    def _wanda_scores_linear(weight: torch.Tensor, s_in: torch.Tensor):
        return torch.matmul(weight.abs(), s_in.to(weight.device))                      # [Cout]

    @staticmethod
    def _slice_weight_in(weight: torch.Tensor, in_keep):
        if in_keep is None:
            return weight
        idx = in_keep.to(weight.device)
        if weight.ndim == 4:  # Conv2d
            return weight.index_select(1, idx)
        else:                 # Linear
            return weight.index_select(1, idx)

    def _get_s_in(self, layer_name, in_keep=None, fallback_weight=None):
        """
        Get per-input-channel L2 norms for a layer (CPU tensor).
        If absent (no calibration), fall back to a weight-based proxy.
        """
        if layer_name in self.act_scales:
            s = self.act_scales[layer_name]
            if in_keep is not None:
                s = s[in_keep.cpu()]
            return s.clamp_min(1e-8)
        # Fallback proxy from weights: average |W| over Cout & kernel dims per Cin
        w = fallback_weight
        if w is None:
            raise RuntimeError(f"No activation stats or fallback weights for '{layer_name}'. "
                               "Run calibration first.")
        if w.ndim == 4:
            proxy = w.abs().view(w.size(0), w.size(1), -1).sum(dim=2).mean(dim=0)  # [Cin]
        else:
            proxy = w.abs().mean(dim=0)                                            # [Cin]
        if in_keep is not None:
            proxy = proxy[in_keep.to(proxy.device)]
        return proxy.detach().cpu().clamp_min(1e-8)

    # ---------------- Apply (same flow as your magnitude code) ----------------
    @torch.no_grad()
    def apply(self):
        # ----- Initial conv + BN (Wanda scores on conv1 OUT channels) -----
        conv1 = self.model_layers['conv1']
        k1 = max(int(conv1.out_channels * self.keep_ratio), self.min_channels)
        s_in_conv1 = self._get_s_in('conv1', fallback_weight=conv1.weight)
        scores1 = self._wanda_scores_conv(conv1.weight, s_in_conv1)
        keep = self._get_keep_indices(scores1, k1)
        self._rebuild_conv('conv1', out_keep=keep)
        self._adjust_bn('bn1', keep)
        prev_keep = keep  # indices in ORIGINAL conv1 basis, as in your code

        # ----- Residual blocks (unchanged mapping logic; only scores swapped) -----
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            blocks = self.model_layers[layer]
            for i, block in enumerate(blocks):
                prefix = f"{layer}.{i}"
                block_module = self.model_layers[prefix]

                # conv1 pruning (Wanda on OUT channels; inputs subset by in_keep)
                in_keep = prev_keep if i == 0 else keep2
                conv1 = block.conv1
                s_in1 = self._get_s_in(f"{prefix}.conv1", in_keep=in_keep, fallback_weight=conv1.weight)
                w1 = self._slice_weight_in(conv1.weight, in_keep)
                assert w1.size(1) == s_in1.numel(), f"Cin mismatch @ {prefix}.conv1"
                conv1_scores = self._wanda_scores_conv(w1, s_in1)
                conv1_k = max(int(conv1.out_channels * self.keep_ratio), self.min_channels)
                keep1 = self._get_keep_indices(conv1_scores, conv1_k)

                self._rebuild_conv(f"{prefix}.conv1", in_keep=in_keep, out_keep=keep1)
                self._adjust_bn(f"{prefix}.bn1", keep1)

                # conv2 pruning (Wanda on OUT channels; inputs subset by keep1)
                conv2 = block.conv2
                s_in2 = self._get_s_in(f"{prefix}.conv2", in_keep=keep1, fallback_weight=conv2.weight)
                w2 = self._slice_weight_in(conv2.weight, keep1)
                assert w2.size(1) == s_in2.numel(), f"Cin mismatch @ {prefix}.conv2"
                conv2_scores = self._wanda_scores_conv(w2, s_in2)
                conv2_k = max(int(conv2.out_channels * self.keep_ratio), self.min_channels)

                if hasattr(block_module, 'downsample') and isinstance(block_module.downsample, nn.Sequential):
                    downsample_conv_name = f"{prefix}.downsample.0"
                    if downsample_conv_name in self.model_layers:
                        keep2 = self._get_keep_indices(conv2_scores, conv2_k)
                        self._rebuild_conv(downsample_conv_name, in_keep=prev_keep, out_keep=keep2)
                        self._adjust_bn(f"{prefix}.downsample.1", keep2)
                    else:
                        keep2 = in_keep
                else:
                    keep2 = in_keep  # identity: match skip width & order, as in your code

                self._rebuild_conv(f"{prefix}.conv2", in_keep=keep1, out_keep=keep2)
                self._adjust_bn(f"{prefix}.bn2", keep2)

                prev_keep = keep2

        # ----- Final FC (prune inputs only; outputs = classes unchanged) -----
        self._prune_linear("fc", prev_keep)
        return self.model

