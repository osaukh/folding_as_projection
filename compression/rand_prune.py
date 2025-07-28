import torch
import torch.nn as nn
from compression.base_resnet import BaseResNet18Compression


class ResNet18_RandomPruning(BaseResNet18Compression):
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
