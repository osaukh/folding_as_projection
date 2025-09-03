import torch
import torch.nn as nn

from compression.base_resnet import BaseResNetCompression
from compression.base_preact_resnet import BasePreActResNetCompression
from compression.base_vit import BaseViTCompression


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


class PreActResNet18_WandaPruning(BasePreActResNetCompression):
    """
    Structured channel pruning for PreActResNet-18 using Wanda scores.
    Keeps your original per-layer compression ratio and block mapping logic:
      - Identity blocks: keep2 = in_keep
      - Downsample blocks: keep2 from Wanda scores; remap shortcut to keep2
      - BN alignment follows PreAct ordering (bn1 with input, bn2 with conv1 output)
    """

    def __init__(self, model, min_channels=1, compression_ratio=0.5):
        super().__init__(model, min_channels, compression_ratio)
        self.act_scales = {}   # layer_name -> 1D CPU tensor [Cin] of L2 norms
        self._hooks = []

    # -------------------- Calibration --------------------
    def _register_activation_hooks(self):
        self._act_sum_sq = {}

        def make_hook(name):
            def hook(mod, inputs):
                x = inputs[0]
                with torch.no_grad():
                    if isinstance(mod, nn.Conv2d):
                        ss = (x.float() ** 2).sum(dim=(0, 2, 3)).detach().cpu()  # [Cin]
                    elif isinstance(mod, nn.Linear):
                        ss = (x.float() ** 2).sum(dim=0).detach().cpu()          # [Cin]
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
    def run_calibration(self, dataloader, device, num_batches=50):
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
        # finalize per-input L2 norms (sqrt of sum of squares), clamp to avoid zeros
        self.act_scales = {name: ss.sqrt().clamp_min(1e-8) for name, ss in self._act_sum_sq.items()}

    # -------------------- Wanda helpers --------------------
    @staticmethod
    def _wanda_scores_conv(weight: torch.Tensor, s_in: torch.Tensor):
        # weight: [Cout, Cin, kH, kW]; s_in: [Cin] (device-agnostic)
        w_abs_sum = weight.abs().view(weight.size(0), weight.size(1), -1).sum(dim=2)  # [Cout, Cin]
        return torch.matmul(w_abs_sum, s_in.to(weight.device))                         # [Cout]

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
        Get per-input-channel L2 norms for a layer (CPU tensor). If absent, fall back
        to a weight-derived proxy (avg |W| over Cout & kernel dims per Cin).
        """
        if layer_name in self.act_scales:
            s = self.act_scales[layer_name]
            if in_keep is not None:
                s = s[in_keep.cpu()]
            return s.clamp_min(1e-8)

        # Fallback proxy (no calibration)
        if fallback_weight is None:
            raise RuntimeError(f"No activation stats or fallback weights for '{layer_name}'. Run calibration first.")
        w = fallback_weight
        if w.ndim == 4:
            proxy = w.abs().view(w.size(0), w.size(1), -1).sum(dim=2).mean(dim=0)  # [Cin]
        else:
            proxy = w.abs().mean(dim=0)                                            # [Cin]
        if in_keep is not None:
            proxy = proxy[in_keep.to(proxy.device)]
        return proxy.detach().cpu().clamp_min(1e-8)

    # -------------------- Apply (mirrors your magnitude flow) --------------------
    @torch.no_grad()
    def apply(self):
        # --- Initial conv (no BN at root in PreAct) ---
        conv1 = self.model_layers['conv1']
        k1 = max(int(conv1.out_channels * self.keep_ratio), self.min_channels)
        s_in_conv1 = self._get_s_in('conv1', fallback_weight=conv1.weight)
        scores1 = self._wanda_scores_conv(conv1.weight, s_in_conv1)
        keep = self._get_keep_indices(scores1, k1)
        self._rebuild_conv('conv1', out_keep=keep)
        prev_keep = keep

        # --- Residual blocks ---
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            blocks = self.model_layers[layer]
            for i, block in enumerate(blocks):
                prefix = f"{layer}.{i}"

                # conv1 (PreAct order: bn1 -> relu -> conv1)
                in_keep = prev_keep
                s_in1 = self._get_s_in(f"{prefix}.conv1", in_keep=in_keep, fallback_weight=block.conv1.weight)
                w1 = self._slice_weight_in(block.conv1.weight, in_keep)
                assert w1.size(1) == s_in1.numel(), f"Cin mismatch @ {prefix}.conv1"
                conv1_scores = self._wanda_scores_conv(w1, s_in1)
                conv1_k = max(int(block.conv1.out_channels * self.keep_ratio), self.min_channels)
                keep1 = self._get_keep_indices(conv1_scores, conv1_k)

                self._rebuild_conv(f"{prefix}.conv1", in_keep=in_keep, out_keep=keep1)
                # bn1 sits BEFORE conv1, so it matches the block INPUT (in_keep)
                self._adjust_bn(f"{prefix}.bn1", in_keep)

                # conv2 (PreAct order: bn2 -> relu -> conv2)
                s_in2 = self._get_s_in(f"{prefix}.conv2", in_keep=keep1, fallback_weight=block.conv2.weight)
                w2 = self._slice_weight_in(block.conv2.weight, keep1)
                assert w2.size(1) == s_in2.numel(), f"Cin mismatch @ {prefix}.conv2"
                conv2_scores = self._wanda_scores_conv(w2, s_in2)
                conv2_k = max(int(block.conv2.out_channels * self.keep_ratio), self.min_channels)

                # shortcut mapping (if present, match downsample to conv2 OUT keep2)
                if hasattr(block, 'shortcut') and isinstance(block.shortcut, nn.Sequential):
                    keep2 = self._get_keep_indices(conv2_scores, conv2_k)
                else:
                    # identity: the add happens in this block; width/order must equal input
                    keep2 = in_keep

                self._rebuild_conv(f"{prefix}.conv2", in_keep=keep1, out_keep=keep2)
                # bn2 sits BEFORE conv2, so it matches conv1 output (keep1)
                self._adjust_bn(f"{prefix}.bn2", keep1)

                if hasattr(block, 'shortcut') and isinstance(block.shortcut, nn.Sequential):
                    self._rebuild_conv(f"{prefix}.shortcut.0", in_keep=in_keep, out_keep=keep2)

                prev_keep = keep2

        # final BN (PreAct tail BN) and classifier
        self._adjust_bn("bn", prev_keep)
        self._prune_linear("linear", prev_keep)
        return self.model




class ViT_WandaPruning(BaseViTCompression):
    """
    Wanda pruning for SimpleViT MLP (c_fc + c_proj):
    - Score c_fc *rows* (hidden units) by |W_fc| weighted with input-channel L2 norms.
    - Keep top-k rows in c_fc; select the same columns in c_proj.
    - Per-layer keep ratio identical to the magnitude version.
    """

    def __init__(self, model, min_channels=1, compression_ratio=0.5):
        super().__init__(model, min_channels, compression_ratio)
        self.act_scales = {}   # module_name -> 1D CPU tensor [in_dim] with L2 norms
        self._hooks = []
        # optional: map of module names to modules (for robust hook naming)
        self._named_modules = dict(self.model.named_modules())

    # -------------------- Calibration --------------------
    def _register_activation_hooks(self):
        """
        Collect ∑ x^2 per input feature for every Linear we might prune.
        Works with inputs shaped [B, N, C] or [*, C].
        """
        self._sum_sq = {}

        def make_hook(name):
            def hook(mod, inputs):
                x = inputs[0]
                with torch.no_grad():
                    # flatten all but last dim -> [T, C]
                    x2 = x.float().reshape(-1, x.shape[-1])
                    ss = (x2 ** 2).sum(dim=0).detach().cpu()     # [C]
                    if name in self._sum_sq:
                        self._sum_sq[name] += ss
                    else:
                        self._sum_sq[name] = ss
            return hook

        # Attach to every Linear; we’ll only use entries for c_fc modules in compress()
        for name, mod in self._named_modules.items():
            if isinstance(mod, nn.Linear):
                self._hooks.append(mod.register_forward_pre_hook(make_hook(name)))

    def _clear_hooks(self):
        for h in self._hooks:
            try: h.remove()
            except: pass
        self._hooks = []

    @torch.no_grad()
    def run_calibration(self, dataloader, device, num_batches=50):
        """
        Run a few batches to estimate input-channel L2 norms for each Linear.
        Use a *clean/no-aug* loader if possible.
        """
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

        # L2 = sqrt(sum of squares); clamp for numerical stability
        self.act_scales = {name: ss.sqrt().clamp_min(1e-8) for name, ss in self._sum_sq.items()}

    # -------------------- Wanda scoring --------------------
    @staticmethod
    def _wanda_row_scores(W_fc: torch.Tensor, s_in: torch.Tensor):
        """
        W_fc: [hidden_dim, in_dim], s_in: [in_dim] -> scores per hidden unit (rows).
        """
        # |W| @ s_in  (broadcasted matmul)
        return torch.matmul(W_fc.abs(), s_in.to(W_fc.device))

    # -------------------- Compress one MLP pair --------------------
    def compress_function(self, axes, params):
        """
        Compress weights for SimpleViT MLP (c_fc + c_proj) using Wanda.
        axes: [module_fc_name, module_proj_name]
        params: dict of tensors; keys like '<name>.weight' / '.bias'
        """
        compressed = {}
        merge_sizes = {}

        module_fc   = axes[0]  # nn.Linear(in_dim -> hidden_dim)
        module_proj = axes[1]  # nn.Linear(hidden_dim -> out_dim)  (usually out_dim == in_dim)

        # --- Extract weights ---
        W_fc   = params[module_fc + '.weight']          # [H, C]
        W_proj = params[module_proj + '.weight']        # [O, H]

        H, C = W_fc.shape
        keep_units = max(int(H * self.keep_ratio), self.min_channels)

        # --- Get s_in for c_fc (input features of c_fc) ---
        # Try calibrated value keyed by *module name*, else fall back to weight proxy
        if module_fc in self.act_scales and self.act_scales[module_fc].numel() == C:
            s_in = self.act_scales[module_fc]
        else:
            # fallback proxy per input dim: avg |W| over rows
            s_in = W_fc.abs().mean(dim=0).detach().cpu()
        s_in = s_in.clamp_min(1e-8)

        # --- Wanda row scores for c_fc rows (hidden units) ---
        scores = self._wanda_row_scores(W_fc, s_in)     # [H]
        topk = torch.topk(scores, keep_units, largest=True).indices.sort()[0].to(W_fc.device)

        # --- Apply selection ---
        new_fc   = W_fc[topk, :]         # rows kept
        new_proj = W_proj[:, topk]       # match columns

        compressed[module_fc + '.weight']  = new_fc
        compressed[module_proj + '.weight'] = new_proj

        # biases
        if module_fc + '.bias' in params and params[module_fc + '.bias'] is not None:
            compressed[module_fc + '.bias'] = params[module_fc + '.bias'][topk]
        if module_proj + '.bias' in params and params[module_proj + '.bias'] is not None:
            compressed[module_proj + '.bias'] = params[module_proj + '.bias']

        # sizes
        merge_sizes[module_fc]   = new_fc.shape[0]
        merge_sizes[module_proj] = new_proj.shape[1]

        return compressed, merge_sizes
