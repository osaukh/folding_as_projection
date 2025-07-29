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
        """
        Folding logic: perform clustering and merge weights.
        """
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
        axis_to_perm = get_axis_to_perm_PreActResNet18(override=False)
        perm_to_axes = axes2perm_to_perm2axes(axis_to_perm)

        for perm_id, axes in perm_to_axes.items():
            # --- Collect weights ---
            features = []
            raw_params = {}
            module_offsets = {}
            offset = 0

            for module_name, axis in axes:
                module = get_module_by_name_PreActResNet18(self.model, module_name)
                weight = module.weight.data if hasattr(module, 'weight') else module.data
                raw_params[module_name] = weight
                weight = weight.transpose(0, axis).contiguous()
                n_channels = weight.shape[0]
                reshaped = weight.view(n_channels, -1)
                features.append(reshaped)
                module_offsets[module_name] = (offset, offset + n_channels)
                offset += n_channels

            # --- Cluster and fold ---
            all_features = torch.cat(features, dim=1)
            n_channels = all_features.shape[0]
            n_clusters = max(int(n_channels * self.keep_ratio), self.min_channels)

            compressed_params, merge_sizes = self.compress_function(axes, raw_params)

            # --- Rebuild modules ---
            param_groups = defaultdict(dict)
            for full_name, tensor in compressed_params.items():
                module_name, param_name = full_name.rsplit('.', 1)
                param_groups[module_name][param_name] = tensor

            for module_name, param_dict in param_groups.items():
                module = get_module_by_name_PreActResNet18(self.model, module_name)

                # Determine if this module should have BN folded
                cluster_labels = None
                if module_name in module_offsets:
                    start, end = module_offsets[module_name]
                    cluster_labels = merge_sizes.get('cluster_labels') if merge_sizes else None

                # Rebuild Conv/Linear/BN
                new_module = self._rebuild_module(
                    module_name,
                    module,
                    param_dict,
                    cluster_labels=cluster_labels,
                    n_clusters=n_clusters
                )

                # Replace in parent
                parent_name = '.'.join(module_name.split('.')[:-1])
                attr_name = module_name.split('.')[-1]
                if parent_name:
                    parent = get_module_by_name_PreActResNet18(self.model, parent_name)
                    setattr(parent, attr_name, new_module)
                else:
                    setattr(self.model, attr_name, new_module)

        print("Model folding complete (with BN folding).")
        return self.model





