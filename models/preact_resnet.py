import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Pre-activation Residual Block
# ---------------------------------------------------------
class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        # FIX: use Sequential to match checkpoint key naming (shortcut.0.weight)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.shortcut = nn.Sequential()  # still sequential, but empty

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if len(self.shortcut) > 0 else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


# ---------------------------------------------------------
# Pre-activation ResNet
# ---------------------------------------------------------
class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        # --- FIX: Global Average Pooling to get 512 features ---
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

# ---------------------------------------------------------
# Factory for PreActResNet18
# ---------------------------------------------------------
def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)

# ---------------------------------------------------------
# Axis-to-permutation mapping (for compression)
# ---------------------------------------------------------
def get_axis_to_perm_PreActResNet18(override: bool = False):
    """
    Axis-to-permutation mapping for PreActResNet18.
    - Conv layers: permute output channels (axis=0)
    - Linear layer: permute input channels (axis=1)
    - Shortcut convolutions are excluded (handled as identity or separately if needed)
    """

    axis_to_perm = {}
    perm_id = 0

    # ---- Initial conv (conv1) ----
    # conv1 defines first feature channels (output permute)
    axis_to_perm["conv1"] = [perm_id, None]
    perm_id += 1

    # ---- Residual stages ----
    # 4 stages: layer1..layer4, each with 2 blocks: 0 and 1
    # Each block: conv1 and conv2
    # conv1 (output) → next conv2 (input)
    # conv2 (output) → next block/stage input
    stages = ["layer1", "layer2", "layer3", "layer4"]

    for stage_name in stages:
        for block_idx in range(2):  # two blocks per stage
            block_prefix = f"{stage_name}.{block_idx}"

            # conv1: new permutation group
            axis_to_perm[f"{block_prefix}.conv1"] = [perm_id, None]
            perm_id += 1

            # conv2: new permutation group
            axis_to_perm[f"{block_prefix}.conv2"] = [perm_id, None]
            perm_id += 1

    # ---- Final linear layer ----
    # Linear input aligns with last conv2 (layer4.1.conv2) output
    axis_to_perm["linear"] = [None, perm_id - 1]

    return axis_to_perm







def get_module_by_name_PreActResNet18(model, name: str):
    module = model
    for part in name.split('.'):
        if part.isdigit():
            idx = int(part)
            children = list(module.children())
            if idx >= len(children):
                # Return module early for empty shortcut
                return module
            module = children[idx]
        else:
            if not hasattr(module, part):
                raise AttributeError(f"{name} not found (missing {part})")
            module = getattr(module, part)
    return module





