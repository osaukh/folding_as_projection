import torch
import torch.nn as nn
from collections import defaultdict

from models.preact_resnet import get_axis_to_perm_PreActResNet18, get_module_by_name_PreActResNet18
from utils.weight_clustering import axes2perm_to_perm2axes, WeightClustering


class BasePreActResNetCompression:
    def __init__(self, model, min_channels=1, compression_ratio=0.5):
        self.model = model
        self.min_channels = min_channels
        self.keep_ratio = 1.0 - compression_ratio
        self.device = next(model.parameters()).device
        self._rename_layers(self.model)

    # --- Setup ---
    def _rename_layers(self, model):
        for name, module in model.named_modules():
            if not hasattr(model, '_module_dict'):
                model._module_dict = {}
            model._module_dict[name] = module

    # --- Abstract method ---
    def apply(self):
        raise NotImplementedError

    # ---- Helpers: Conv, BN, FC folding ----
    def _fold_bn_params(self, original_bn, cluster_labels, n_clusters):
        device = original_bn.weight.device
        new_bn = nn.BatchNorm2d(n_clusters).to(device)

        for param_name in ['weight', 'bias']:
            original = getattr(original_bn, param_name).data
            fused = torch.zeros(n_clusters, device=device)
            for k in range(n_clusters):
                mask = cluster_labels == k
                if mask.sum() > 0:
                    fused[k] = original[mask].mean()
            setattr(new_bn, param_name, nn.Parameter(fused))

        for stat_name in ['running_mean', 'running_var']:
            original = getattr(original_bn, stat_name).data
            fused = torch.zeros(n_clusters, device=device)
            for k in range(n_clusters):
                mask = cluster_labels == k
                if mask.sum() > 0:
                    fused[k] = original[mask].mean()
            getattr(new_bn, stat_name).data.copy_(fused)

        return new_bn

    def _rebuild_module(self, name, old_module, param_dict, cluster_labels=None, n_clusters=None):
        if isinstance(old_module, nn.Conv2d):
            new_weight = param_dict.get('weight')
            new_out_channels = new_weight.shape[0]
            new_in_channels = new_weight.shape[1]
            new_conv = nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=new_out_channels,
                kernel_size=old_module.kernel_size,
                stride=old_module.stride,
                padding=old_module.padding,
                dilation=old_module.dilation,
                groups=old_module.groups,
                bias='bias' in param_dict,
                padding_mode=old_module.padding_mode
            ).to(self.device)
            new_conv.weight.data = param_dict['weight'].clone()
            if 'bias' in param_dict:
                new_conv.bias.data = param_dict['bias'].clone()
            return new_conv

        elif isinstance(old_module, nn.BatchNorm2d):
            if cluster_labels is not None and n_clusters is not None:
                return self._fold_bn_params(old_module, cluster_labels, n_clusters)
            else:
                new_num_features = param_dict['weight'].shape[0]
                new_bn = nn.BatchNorm2d(new_num_features).to(self.device)
                for pname in ['weight', 'bias', 'running_mean', 'running_var']:
                    if pname in param_dict:
                        getattr(new_bn, pname).data = param_dict[pname].clone()
                return new_bn

        elif isinstance(old_module, nn.Linear):
            new_weight = param_dict.get('weight')
            new_out_features = new_weight.shape[0]
            new_in_features = new_weight.shape[1]
            new_fc = nn.Linear(new_in_features, new_out_features).to(self.device)
            new_fc.weight.data = param_dict['weight'].clone()
            if 'bias' in param_dict:
                new_fc.bias.data = param_dict['bias'].clone()
            return new_fc

        return old_module
