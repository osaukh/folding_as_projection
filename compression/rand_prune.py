import torch
import torch.nn as nn
from compression.base_resnet import BaseResNetCompression
from compression.base_preact_resnet import BasePreActResNetCompression
from compression.base_clip_vit import BaseCLIPViTCompression

class ResNet18_RandomPruning(BaseResNetCompression):
    def __init__(self, model, min_channels=1, compression_ratio=0.5):
        super().__init__(model, min_channels, compression_ratio)
        self.model_layers = dict(self.model.named_modules())

    def _get_random_indices(self, n, k):
        """Select k random channel indices (no sorting)."""
        perm = torch.randperm(n, device=self.device)
        return perm[:k]

    def apply(self):
        # Initial conv + BN
        conv1 = self.model_layers['conv1']
        k1 = max(int(conv1.out_channels * self.keep_ratio), self.min_channels)
        keep = self._get_random_indices(conv1.out_channels, k1)
        self._rebuild_conv('conv1', out_keep=keep)
        self._adjust_bn('bn1', keep)
        prev_keep = keep

        # Residual blocks
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            blocks = self.model_layers[layer]
            for i, block in enumerate(blocks):
                prefix = f"{layer}.{i}"
                block_module = self.model_layers[prefix]

                # conv1 random pruning
                conv1 = block.conv1
                k1 = max(int(conv1.out_channels * self.keep_ratio), self.min_channels)
                keep1 = self._get_random_indices(conv1.out_channels, k1)
                in_keep = prev_keep if i == 0 else keep2

                self._rebuild_conv(f"{prefix}.conv1", in_keep=in_keep, out_keep=keep1)
                self._adjust_bn(f"{prefix}.bn1", keep1)

                # conv2 random pruning
                conv2 = block.conv2
                k2 = max(int(conv2.out_channels * self.keep_ratio), self.min_channels)

                if hasattr(block_module, 'downsample') and isinstance(block_module.downsample, nn.Sequential):
                    downsample_conv_name = f"{prefix}.downsample.0"
                    if downsample_conv_name in self.model_layers:
                        keep2 = self._get_random_indices(conv2.out_channels, k2)
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




class CLIPViT_RandomPruning(BaseCLIPViTCompression):
    def compress_function(self, axes, params):
        """
        Random pruning for CLIP ViT:
        - Randomly keep a subset of channels proportional to keep_ratio.
        - Apply same subset to c_proj input channels.
        """
        compressed, merge_sizes = {}, {}

        # Module names
        module_fc, _ = axes[0]     # c_fc
        module_proj, _ = axes[1]   # c_proj

        # Extract weights
        W_fc = params[module_fc]      # [hidden_dim, in_dim]
        W_proj = params[module_proj]  # [out_dim, hidden_dim]
        device = W_fc.device

        # Determine number of channels to keep
        n_channels = W_fc.shape[0]
        n_keep = max(int(n_channels * self.keep_ratio), self.min_channels)

        # Random selection of channels
        perm = torch.randperm(n_channels, device=device)
        keep_indices = perm[:n_keep]

        # Apply pruning to c_fc (output) and c_proj (input)
        new_fc = W_fc[keep_indices, :]           # reduced rows
        new_proj = W_proj[:, keep_indices]       # reduced columns

        # Assign compressed weights
        compressed[module_fc + '.weight'] = new_fc
        compressed[module_proj + '.weight'] = new_proj

        # Bias pruning
        if module_fc + '.bias' in params and params[module_fc + '.bias'] is not None:
            compressed[module_fc + '.bias'] = params[module_fc + '.bias'][keep_indices]
        if module_proj + '.bias' in params and params[module_proj + '.bias'] is not None:
            compressed[module_proj + '.bias'] = params[module_proj + '.bias']  # unchanged

        # Track new sizes
        merge_sizes[module_fc] = new_fc.shape[0]
        merge_sizes[module_proj] = new_proj.shape[1]

        return compressed, merge_sizes


class PreActResNet18_RandomPruning(BasePreActResNetCompression):
    def __init__(self, model, min_channels=1, compression_ratio=0.5):
        super().__init__(model, min_channels, compression_ratio)
        self.model_layers = dict(self.model.named_modules())

    def _get_random_indices(self, n, k):
        """Randomly select k channel indices."""
        return torch.randperm(n, device=self.device)[:k]

    def apply(self):
        # --- Initial conv (no BN at root in PreActResNet) ---
        conv1 = self.model_layers['conv1']
        k1 = max(int(conv1.out_channels * self.keep_ratio), self.min_channels)
        keep = self._get_random_indices(conv1.out_channels, k1)
        self._rebuild_conv('conv1', out_keep=keep)
        prev_keep = keep

        # Residual blocks
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            blocks = self.model_layers[layer]

            for i, block in enumerate(blocks):
                prefix = f"{layer}.{i}"

                # --- conv1 ---
                conv1_k = max(int(block.conv1.out_channels * self.keep_ratio), self.min_channels)
                keep1 = self._get_random_indices(block.conv1.out_channels, conv1_k)
                in_keep = prev_keep

                self._rebuild_conv(f"{prefix}.conv1", in_keep=in_keep, out_keep=keep1)
                self._adjust_bn(f"{prefix}.bn1", in_keep)

                # --- conv2 ---
                conv2_k = max(int(block.conv2.out_channels * self.keep_ratio), self.min_channels)
                # If shortcut exists, conv2 output must match shortcut output
                if hasattr(block, 'shortcut') and isinstance(block.shortcut, nn.Sequential):
                    keep2 = self._get_random_indices(block.conv2.out_channels, conv2_k)
                else:
                    keep2 = in_keep

                self._rebuild_conv(f"{prefix}.conv2", in_keep=keep1, out_keep=keep2)
                self._adjust_bn(f"{prefix}.bn2", keep1)

                # --- shortcut (downsample) ---
                if hasattr(block, 'shortcut') and isinstance(block.shortcut, nn.Sequential):
                    self._rebuild_conv(f"{prefix}.shortcut.0", in_keep=in_keep, out_keep=keep2)

                prev_keep = keep2

        # Final BN and FC
        self._adjust_bn("bn", prev_keep)
        self._prune_linear("linear", prev_keep)
        return self.model
