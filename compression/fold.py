import torch
import torch.nn as nn
from collections import defaultdict

from models.resnet import get_module_by_name_ResNet, get_axis_to_perm_ResNet18
from models.preact_resnet import get_module_by_name_PreActResNet18, get_axis_to_perm_PreActResNet18
from utils.weight_clustering import WeightClustering, _log_cluster_stats, concat_weights, axes2perm_to_perm2axes, \
    merge_channel_clustering, NopMerge

from compression.base_clip_vit import BaseCLIPViTCompression
from compression.base_resnet import BaseResNetCompression
from compression.base_preact_resnet import BasePreActResNetCompression


class ResNet18_ModelFolding(BaseResNetCompression):
    def compress_function(self, axes, params):
        """
        Folding logic: perform clustering and merge weights.
        """
        # Use hkmeans clustering
        from utils.weight_clustering import merge_channel_clustering

        n_channels = params[axes[0][0]].shape[axes[0][1]]
        n_clusters = max(int(n_channels * self.keep_ratio), 1)

        # Flatten and cluster
        weight = concat_weights({0: axes}, params, 0, n_channels)
        clusterer = WeightClustering(n_clusters=n_clusters, n_features=n_channels,
                                     method="hkmeans", normalize=False, use_pca=True)
        labels = clusterer(weight).to(self.device).long()

        # Log cluster stats
        _log_cluster_stats(weight, labels, axes[0][0])

        # Convert to merge matrix
        merge_matrix = torch.zeros((n_clusters, n_channels), device=self.device, dtype=torch.float32)
        merge_matrix.scatter_(0, labels.unsqueeze(0), 1.0)
        merge_matrix /= merge_matrix.sum(dim=1, keepdim=True).clamp(min=1)

        # Merge weights
        from utils.weight_clustering import NopMerge
        compressed_params = merge_channel_clustering({0: axes}, params, 0, merge_matrix, custom_merger=NopMerge())

        return compressed_params, {'cluster_labels': labels}

    def apply(self):
        print(f"[INFO] Starting {self.__class__.__name__}...")
        axis_to_perm = get_axis_to_perm_ResNet18(override=False)
        perm_to_axes = axes2perm_to_perm2axes(axis_to_perm)

        for perm_id, axes in perm_to_axes.items():
            # --- Collect raw params ---
            raw_params = {}
            for module_name, axis in axes:
                module = get_module_by_name_ResNet(self.model, module_name)
                weight = module.weight.data if hasattr(module, 'weight') else module.data
                raw_params[module_name] = weight

            # --- Call specific compression/pruning logic ---
            compressed_params, meta = self.compress_function(axes, raw_params)

            # --- Rebuild modules ---
            param_groups = defaultdict(dict)
            for full_name, tensor in compressed_params.items():
                module_name, pname = full_name.rsplit('.', 1)
                param_groups[module_name][pname] = tensor

            for module_name, param_dict in param_groups.items():
                module = get_module_by_name_ResNet(self.model, module_name)
                cluster_labels = meta.get('cluster_labels') if meta else None
                n_clusters = param_dict['weight'].shape[0]
                new_module = self._rebuild_module(module, param_dict, cluster_labels, n_clusters)
                self._replace(module_name, new_module)

        return self.model



class CLIPViT_ModelFolding(BaseCLIPViTCompression):
    # --- Folding for CLIP ViT (Representative Selection) ---
    def compress_function(self, axes, params):
        """
            Compress weights for CLIP ViT MLP (c_fc + c_proj) using representative selection:
            - Cluster c_fc output channels
            - Select representative indices per cluster (first occurrence)
            - Apply same selection to c_proj input channels
            """
        compressed = {}
        merge_sizes = {}

        # Unpack module names
        module_fc, _ = axes[0]
        module_proj, _ = axes[1]

        # Extract weights
        W_fc = params[module_fc]  # [hidden_dim, in_dim]
        W_proj = params[module_proj]  # [out_dim, hidden_dim]

        # --- Clustering (HKMeans, no PCA, no normalization) ---
        n_channels = W_fc.shape[0]
        n_clusters = max(int(n_channels * self.keep_ratio), self.min_channels)

        clusterer = WeightClustering(n_clusters=n_clusters, method="hkmeans", use_pca=False, normalize=False)
        labels = clusterer(W_fc).to(W_fc.device).long()

        _log_cluster_stats(W_fc, labels, module_fc)

        # Representative indices (first element in each cluster)
        unique_labels = torch.unique(labels, sorted=True)
        rep_indices = torch.stack([
            torch.nonzero(labels == lbl, as_tuple=True)[0][0] for lbl in unique_labels
        ]).to(W_fc.device)

        # Select channels
        new_fc = W_fc[rep_indices, :]
        new_proj = W_proj[:, rep_indices]

        compressed[module_fc + '.weight'] = new_fc
        compressed[module_proj + '.weight'] = new_proj

        # Biases
        if module_fc + '.bias' in params and params[module_fc + '.bias'] is not None:
            compressed[module_fc + '.bias'] = params[module_fc + '.bias'][rep_indices]
        if module_proj + '.bias' in params and params[module_proj + '.bias'] is not None:
            compressed[module_proj + '.bias'] = params[module_proj + '.bias']

        # Track new sizes
        merge_sizes[module_fc] = new_fc.shape[0]
        merge_sizes[module_proj] = new_proj.shape[1]

        return compressed, merge_sizes




class PreActResNet18_ModelFolding(BasePreActResNetCompression):
    def compress_function(self, axes, params):
        from utils.weight_clustering import merge_channel_clustering, NopMerge

        compressed_params = {}
        meta = {}

        # Filter axes: skip modules not in params (e.g., empty shortcuts)
        valid_axes = [(mn, ax) for mn, ax in axes if mn in params]

        # Group by channel count
        size_groups = defaultdict(list)
        for module_name, axis in valid_axes:
            n_channels = params[module_name].shape[axis]
            size_groups[n_channels].append((module_name, axis))

        # Process each group
        for n_channels, grouped_axes in size_groups.items():
            n_clusters = max(int(n_channels * self.keep_ratio), 1)

            # Build features for clustering
            features = []
            for module_name, axis in grouped_axes:
                weight = params[module_name]
                weight_t = weight.transpose(0, axis).contiguous()
                reshaped = weight_t.view(weight_t.shape[0], -1)
                features.append(reshaped)

            all_features = torch.cat(features, dim=1)

            # Cluster
            clusterer = WeightClustering(
                n_clusters=n_clusters,
                n_features=all_features.shape[1],
                method="hkmeans",
                normalize=False,
                use_pca=True
            )
            labels = clusterer(all_features).to(self.device).long()

            _log_cluster_stats(all_features, labels, grouped_axes[0][0])

            # Merge weights
            merge_matrix = torch.zeros((n_clusters, n_channels), device=self.device)
            merge_matrix.scatter_(0, labels.unsqueeze(0), 1.0)
            merge_matrix /= merge_matrix.sum(dim=1, keepdim=True).clamp(min=1)

            compressed_group = merge_channel_clustering(
                {0: grouped_axes}, params, 0, merge_matrix, custom_merger=NopMerge()
            )
            compressed_params.update(compressed_group)

            # Store cluster labels for BN folding
            for mn, _ in grouped_axes:
                meta[mn] = {"cluster_labels": labels}

        return compressed_params, meta

    def apply(self):
        """
        Apply model folding (channel clustering + centroid merging) to PreActResNet18.
        Handles conv layers and folds bn2 (output BN) while leaving bn1 untouched.
        Adjusts the final linear layer to match last conv output.
        """
        print(f"[INFO] Starting {self.__class__.__name__}...")

        # --- Build mapping of permutation groups ---
        axis_to_perm = get_axis_to_perm_PreActResNet18(override=False)
        perm_to_axes = axes2perm_to_perm2axes(axis_to_perm)

        # --- Process each permutation group ---
        for perm_id, axes in perm_to_axes.items():
            raw_params = {}

            # Gather weights for this group
            for module_name, axis in axes:
                module = get_module_by_name_PreActResNet18(self.model, module_name)

                # Skip identity shortcuts (no weight attribute)
                if not hasattr(module, "weight"):
                    continue

                raw_params[module_name] = module.weight.data

            # Skip if group has no trainable weights
            if not raw_params:
                continue

            # --- Perform clustering + merge ---
            compressed_params, meta = self.compress_function(axes, raw_params)

            # Rebuild modules with compressed weights
            param_groups = defaultdict(dict)
            for full_name, tensor in compressed_params.items():
                if "." in full_name:
                    module_name, param_name = full_name.rsplit(".", 1)
                else:
                    module_name, param_name = full_name, "weight"
                param_groups[module_name][param_name] = tensor

            for module_name, param_dict in param_groups.items():
                module = get_module_by_name_PreActResNet18(self.model, module_name)
                new_module = self._rebuild_module(module_name, module, param_dict)

                # Replace module in model
                parent_name = ".".join(module_name.split(".")[:-1])
                attr_name = module_name.split(".")[-1]
                if parent_name:
                    parent = get_module_by_name_PreActResNet18(self.model, parent_name)
                    setattr(parent, attr_name, new_module)
                else:
                    setattr(self.model, attr_name, new_module)

                # --- Handle BatchNorm folding ---
                # Only fold bn2 (output BN of conv1) to match reduced output channels
                if "conv1" in module_name:
                    bn_name = module_name.replace("conv1", "bn2")
                else:
                    bn_name = None

                if bn_name:
                    try:
                        bn_module = get_module_by_name_PreActResNet18(self.model, bn_name)
                    except AttributeError:
                        bn_module = None

                    if bn_module is not None and "weight" in param_dict:
                        labels = meta.get(module_name, {}).get("cluster_labels", None)
                        if labels is not None:
                            n_clusters = param_dict["weight"].shape[0]
                            folded_bn = self._fold_bn_params(bn_module, labels, n_clusters)

                            # Replace BN in model
                            parent_name_bn = ".".join(bn_name.split(".")[:-1])
                            attr_name_bn = bn_name.split(".")[-1]
                            if parent_name_bn:
                                parent_bn = get_module_by_name_PreActResNet18(self.model, parent_name_bn)
                                setattr(parent_bn, attr_name_bn, folded_bn)
                            else:
                                setattr(self.model, attr_name_bn, folded_bn)

        # --- Adjust final linear layer to match last conv output ---
        last_conv = get_module_by_name_PreActResNet18(self.model, "layer4.1.conv2")
        new_in_features = last_conv.out_channels
        old_linear = self.model.linear
        new_linear = nn.Linear(new_in_features, old_linear.out_features).to(self.device)

        # Copy weights partially if dimensions differ
        min_dim = min(old_linear.weight.shape[1], new_in_features)
        new_linear.weight.data[:, :min_dim] = old_linear.weight.data[:, :min_dim]
        if old_linear.bias is not None:
            new_linear.bias.data = old_linear.bias.data.clone()

        self.model.linear = new_linear

        print("Model folding complete.")
        return self.model




