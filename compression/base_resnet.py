import torch
import torch.nn as nn
from collections import defaultdict
from models.resnet import get_axis_to_perm_ResNet18, get_module_by_name_ResNet, axes2perm_to_perm2axes

class BaseResNetCompression:
    """
    Base class for ResNet compression (folding or pruning).
    Unified apply() pipeline:
      - Iterate groups via axis_to_perm
      - Collect weights & biases
      - Call self.compress_or_prune() to get compressed params
      - Rebuild and replace modules
    """

    def __init__(self, model, min_channels=1, compression_ratio=0.5):
        self.model = model
        self.min_channels = min_channels
        self.keep_ratio = 1.0 - compression_ratio
        self.device = next(model.parameters()).device
        self._rename_layers(self.model)
        self.model_layers = dict(self.model.named_modules())

    # --- Setup ---
    def _rename_layers(self, model):
        """Attach flat module dict for fast lookup"""
        model._module_dict = {name: module for name, module in model.named_modules()}

    def _replace(self, name, new_layer):
        """Replace module by name in nested model"""
        parts = name.split('.')
        mod = self.model
        for part in parts[:-1]:
            mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
        last = parts[-1]
        if last.isdigit():
            mod[int(last)] = new_layer
        else:
            setattr(mod, last, new_layer)

    # --- Abstract method ---
    def apply(self):
        raise NotImplementedError


    # ---- Helpers: Conv, BN, FC folding ----
    def _fold_bn_params(self, original_bn, cluster_labels, n_clusters):
        """Fuse BatchNorm params by averaging clusters"""
        device = original_bn.weight.device
        new_bn = nn.BatchNorm2d(n_clusters).to(device)

        for pname in ['weight', 'bias']:
            original = getattr(original_bn, pname).data
            fused = torch.stack([
                original[cluster_labels == k].mean() if (cluster_labels == k).any() else torch.tensor(0., device=device)
                for k in range(n_clusters)
            ])
            setattr(new_bn, pname, nn.Parameter(fused))

        for pname in ['running_mean', 'running_var']:
            original = getattr(original_bn, pname).data
            fused = torch.stack([
                original[cluster_labels == k].mean() if (cluster_labels == k).any() else torch.tensor(0., device=device)
                for k in range(n_clusters)
            ])
            getattr(new_bn, pname).data.copy_(fused)

        return new_bn

    def _copy_bn(self, old_bn, param_dict):
        new_bn = nn.BatchNorm2d(param_dict['weight'].shape[0]).to(self.device)
        for pname in ['weight', 'bias', 'running_mean', 'running_var']:
            if pname in param_dict:
                getattr(new_bn, pname).data.copy_(param_dict[pname])
        return new_bn

    # --- Module rebuild ---
    def _rebuild_module(self, old_module, param_dict, cluster_labels=None, n_clusters=None):
        # Conv2d
        if isinstance(old_module, nn.Conv2d):
            w = param_dict['weight']
            new_conv = nn.Conv2d(
                in_channels=w.shape[1], out_channels=w.shape[0],
                kernel_size=old_module.kernel_size, stride=old_module.stride,
                padding=old_module.padding, dilation=old_module.dilation,
                groups=old_module.groups, bias='bias' in param_dict,
                padding_mode=old_module.padding_mode
            ).to(self.device)
            new_conv.weight.data.copy_(w)
            if 'bias' in param_dict:
                new_conv.bias.data.copy_(param_dict['bias'])
            return new_conv

        # BatchNorm2d
        elif isinstance(old_module, nn.BatchNorm2d):
            return self._fold_bn_params(old_module, cluster_labels, n_clusters) \
                if cluster_labels is not None else self._copy_bn(old_module, param_dict)

        # Linear
        elif isinstance(old_module, nn.Linear):
            w = param_dict['weight']
            new_fc = nn.Linear(w.shape[1], w.shape[0]).to(self.device)
            new_fc.weight.data.copy_(w)
            if 'bias' in param_dict:
                new_fc.bias.data.copy_(param_dict['bias'])
            return new_fc

        return old_module

    # ---- Helpers: Conv, BN, FC pruning ----
    def _get_keep_indices(self, scores, k):
        return torch.argsort(scores, descending=True)[:k]

    def _prune_linear(self, name, keep):
        layer = self.model_layers[name]
        assert isinstance(layer, nn.Linear)
        new_linear = nn.Linear(len(keep), layer.out_features).to(self.device)
        new_linear.weight = nn.Parameter(layer.weight[:, keep].clone())
        if layer.bias is not None:
            new_linear.bias = nn.Parameter(layer.bias.detach().clone())
        self._replace(name, new_linear)
        self.model_layers[name] = new_linear

    def _rebuild_conv(self, name, in_keep=None, out_keep=None):
        if name not in self.model_layers:
            return
        layer = self.model_layers[name]
        assert isinstance(layer, nn.Conv2d)
        weight = layer.weight.detach()
        orig_out, orig_in = weight.shape[:2]
        out_idx = out_keep if out_keep is not None else torch.arange(orig_out, device=weight.device)
        in_idx = in_keep if in_keep is not None else torch.arange(orig_in, device=weight.device)
        new_weight = weight.index_select(0, out_idx).index_select(1, in_idx).clone()
        new_conv = nn.Conv2d(
            in_channels=len(in_idx),
            out_channels=len(out_idx),
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode
        ).to(self.device)
        new_conv.weight = nn.Parameter(new_weight)
        if layer.bias is not None:
            new_conv.bias = nn.Parameter(layer.bias[out_idx].clone())
        self._replace(name, new_conv)
        self.model_layers[name] = new_conv

    def _adjust_bn(self, name, keep):
        bn = self.model_layers[name]
        assert isinstance(bn, nn.BatchNorm2d)
        new_bn = nn.BatchNorm2d(len(keep)).to(self.device)
        new_bn.weight = nn.Parameter(bn.weight[keep].detach().clone())
        new_bn.bias = nn.Parameter(bn.bias[keep].detach().clone())
        new_bn.running_mean = bn.running_mean[keep].detach().clone()
        new_bn.running_var = bn.running_var[keep].detach().clone()
        self._replace(name, new_bn)
        self.model_layers[name] = new_bn



