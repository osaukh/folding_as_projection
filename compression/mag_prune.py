import torch
import torch.nn as nn

from models.resnet import get_module_by_name_ResNet, get_axis_to_perm_ResNet18
from models.preact_resnet import get_module_by_name_PreActResNet18, get_axis_to_perm_PreActResNet18

from compression.base_clip_vit import BaseCLIPViTCompression
from compression.base_resnet import BaseResNetCompression
from compression.base_preact_resnet import BasePreActResNetCompression
from utils.weight_clustering import axes2perm_to_perm2axes


class ResNet18_MagnitudePruning(BaseResNetCompression):
    def __init__(self, model, min_channels=1, compression_ratio=0.5, p=2):
        super().__init__(model, min_channels, compression_ratio)
        self.p = p
        self.model_layers = dict(self.model.named_modules())

    def apply(self):
        # Initial conv + BN
        conv1 = self.model_layers['conv1']
        k1 = max(int(conv1.out_channels * self.keep_ratio), self.min_channels)
        scores1 = torch.norm(conv1.weight.view(conv1.out_channels, -1), p=self.p, dim=1)
        keep = self._get_keep_indices(scores1, k1)
        self._rebuild_conv('conv1', out_keep=keep)
        self._adjust_bn('bn1', keep)
        prev_keep = keep

        # Residual blocks
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            blocks = self.model_layers[layer]
            for i, block in enumerate(blocks):
                prefix = f"{layer}.{i}"
                block_module = self.model_layers[prefix]

                # conv1 pruning
                conv1 = block.conv1
                conv1_scores = torch.norm(conv1.weight.view(conv1.out_channels, -1), p=self.p, dim=1)
                conv1_k = max(int(len(conv1_scores) * self.keep_ratio), self.min_channels)
                keep1 = self._get_keep_indices(conv1_scores, conv1_k)
                in_keep = prev_keep if i == 0 else keep2

                self._rebuild_conv(f"{prefix}.conv1", in_keep=in_keep, out_keep=keep1)
                self._adjust_bn(f"{prefix}.bn1", keep1)

                # conv2 pruning
                conv2 = block.conv2
                conv2_scores = torch.norm(conv2.weight.view(conv2.out_channels, -1), p=self.p, dim=1)
                conv2_k = max(int(len(conv2_scores) * self.keep_ratio), self.min_channels)

                if hasattr(block_module, 'downsample') and isinstance(block_module.downsample, nn.Sequential):
                    downsample_conv_name = f"{prefix}.downsample.0"
                    if downsample_conv_name in self.model_layers:
                        keep2 = self._get_keep_indices(conv2_scores, conv2_k)
                        self._rebuild_conv(downsample_conv_name, in_keep=prev_keep, out_keep=keep2)
                        self._adjust_bn(f"{prefix}.downsample.1", keep2)
                    else:
                        keep2 = in_keep
                else:
                    keep2 = in_keep

                self._rebuild_conv(f"{prefix}.conv2", in_keep=keep1, out_keep=keep2)
                self._adjust_bn(f"{prefix}.bn2", keep2)

                prev_keep = keep2

        # Final FC
        self._prune_linear("fc", prev_keep)
        return self.model




# --- Magnitude pruning for CLIP ViT ---
class CLIPViT_MagnitudePruning(BaseCLIPViTCompression):
    def __init__(self, model, min_channels=1, compression_ratio=0.5, p=2):
        super().__init__(model, min_channels, compression_ratio)
        self.p = p

    def compress_function(self, axes, params):
        """
            Perform magnitude-based channel pruning for CLIP ViT MLP (c_fc + c_proj):
            - Rank c_fc output channels by magnitude (L2 norm)
            - Keep top-k channels according to keep_ratio
            - Apply same selection to c_proj input channels
            """
        compressed = {}
        merge_sizes = {}

        # --- Unpack modules ---
        module_fc, _ = axes[0]  # c_fc (output channels)
        module_proj, _ = axes[1]  # c_proj (input channels)

        # --- Extract weights ---
        W_fc = params[module_fc]  # [hidden_dim, in_dim]
        W_proj = params[module_proj]  # [out_dim, hidden_dim]

        # --- Compute per-channel L2 norm ---
        norms = torch.norm(W_fc.view(W_fc.shape[0], -1), dim=1, p=self.p)  # [hidden_dim]

        # --- Determine number of channels to keep ---
        n_channels = W_fc.shape[0]
        k = max(int(n_channels * self.keep_ratio), self.min_channels)  # keep ratio
        topk_indices = torch.topk(norms, k=k, largest=True).indices

        # --- Sort indices for consistency ---
        topk_indices, _ = torch.sort(topk_indices)

        # --- Select pruned weights ---
        new_fc = W_fc[topk_indices, :]
        new_proj = W_proj[:, topk_indices]

        compressed[module_fc + '.weight'] = new_fc
        compressed[module_proj + '.weight'] = new_proj

        # --- Bias pruning
        if module_fc + '.bias' in params and params[module_fc + '.bias'] is not None:
            compressed[module_fc + '.bias'] = params[module_fc + '.bias'][topk_indices]
        if module_proj + '.bias' in params and params[module_proj + '.bias'] is not None:
            compressed[module_proj + '.bias'] = params[module_proj + '.bias']

        # --- Track new sizes
        merge_sizes[module_fc] = new_fc.shape[0]
        merge_sizes[module_proj] = new_proj.shape[1]

        return compressed, merge_sizes




class PreActResNet18_MagnitudePruning(BasePreActResNetCompression):
    def __init__(self, model, min_channels=1, compression_ratio=0.5, p=2):
        super().__init__(model, min_channels, compression_ratio)
        self.p = p

    def apply(self):
        # --- Initial conv (no BN at root in PreActResNet) ---
        conv1 = self.model.conv1
        k1 = max(int(conv1.out_channels * self.keep_ratio), self.min_channels)
        scores1 = torch.norm(conv1.weight.view(conv1.out_channels, -1), p=self.p, dim=1)
        keep = self._get_keep_indices(scores1, k1)
        self._rebuild_conv('conv1', out_keep=keep)
        prev_keep = keep

        # --- Residual blocks ---
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            blocks = getattr(self.model, layer)
            for i, block in enumerate(blocks):
                prefix = f"{layer}.{i}"

                # conv1
                conv1_scores = torch.norm(block.conv1.weight.view(block.conv1.out_channels, -1), p=self.p, dim=1)
                conv1_k = max(int(len(conv1_scores) * self.keep_ratio), self.min_channels)
                keep1 = self._get_keep_indices(conv1_scores, conv1_k)
                in_keep = prev_keep if i == 0 else keep2

                self._rebuild_conv(f"{prefix}.conv1", in_keep=in_keep, out_keep=keep1)
                self._adjust_bn(f"{prefix}.bn1", keep1)

                # conv2
                conv2_scores = torch.norm(block.conv2.weight.view(block.conv2.out_channels, -1), p=self.p, dim=1)
                conv2_k = max(int(len(conv2_scores) * self.keep_ratio), self.min_channels)

                # Handle shortcut if present
                if hasattr(block, "shortcut") and isinstance(block.shortcut, nn.Sequential) and len(block.shortcut) > 0:
                    keep2 = self._get_keep_indices(conv2_scores, conv2_k)
                    self._rebuild_conv(f"{prefix}.shortcut.0", in_keep=prev_keep, out_keep=keep2)
                    self._adjust_bn(f"{prefix}.shortcut.1", keep2)
                else:
                    keep2 = in_keep

                self._rebuild_conv(f"{prefix}.conv2", in_keep=keep1, out_keep=keep2)
                self._adjust_bn(f"{prefix}.bn2", keep2)

                prev_keep = keep2

        # --- Final linear ---
        self._prune_linear("linear", prev_keep)
        return self.model




