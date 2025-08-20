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
from compression.base_vit import BaseViTCompression


class ResNet18_ModelFolding(BaseResNetCompression):
    def compress_function(self, axes, params):
        """
        Folding logic: perform clustering and merge weights.
        """
        # Always cluster on output channels
        n_channels = params[axes[0][0]].shape[0]
        n_clusters = max(int(n_channels * self.keep_ratio), 1)

        # Flatten weights across layers (output dim)
        weight = concat_weights({0: axes}, params, 0, n_channels)

        # Cluster
        clusterer = WeightClustering(n_clusters=n_clusters, n_features=n_channels,
                                     method="hkmeans", normalize=False, use_pca=True)
        labels = clusterer(weight).to(self.device).long()

        _log_cluster_stats(weight, labels, axes[0][0])

        # Build merge matrix
        merge_matrix = torch.zeros((n_clusters, n_channels), device=self.device)
        merge_matrix.scatter_(0, labels.unsqueeze(0), 1.0)
        cluster_counts = merge_matrix.sum(dim=1, keepdim=True)
        merge_matrix /= cluster_counts.clamp(min=1)  # avoid div by zero

        # Merge params
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
    def compress_function(self, axes, params):
        """
        Compress weights for CLIP ViT MLP (c_fc + c_proj) using cluster means:
          - Normalize W_fc (per feature) for fair clustering
          - Cluster c_fc output channels on the normalized matrix
          - Use the mean vector of each cluster (in normalized space), then de-normalize
          - For c_proj, sum columns corresponding to clustered members
          - For c_fc bias, average within each cluster
        """
        import torch

        compressed = {}
        merge_sizes = {}

        # Unpack module names (axes entries are tuples, second element unused here)
        module_fc, _ = axes[0]
        module_proj, _ = axes[1]

        # Extract weights
        W_fc = params[module_fc]  # [hidden_dim, in_dim]
        W_proj = params[module_proj]  # [out_dim, hidden_dim]

        device = W_fc.device
        dtype = W_fc.dtype

        # --- Determine number of clusters ---
        n_channels = W_fc.shape[0]
        n_clusters = max(int(n_channels * self.keep_ratio), self.min_channels)
        n_clusters = min(n_clusters, n_channels)

        # --- Column-wise normalization (z-score) before clustering ---
        eps = torch.finfo(dtype).eps
        col_mean = W_fc.mean(dim=0, keepdim=True)  # [1, in_dim]
        col_std = W_fc.std(dim=0, unbiased=False, keepdim=True) + eps  # [1, in_dim]
        W_fc_norm = (W_fc - col_mean) / col_std  # [hidden_dim, in_dim]

        # --- Clustering on normalized matrix (no extra normalization inside) ---
        clusterer = WeightClustering(
            n_clusters=n_clusters,
            method="hkmeans",
            use_pca=False,
            normalize=False  # already normalized above
        )
        labels = clusterer(W_fc_norm).to(device).long()

        _log_cluster_stats(W_fc_norm, labels, module_fc)

        # --- Ensure labels cover all (possibly fewer) clusters ---
        unique_labels = torch.unique(labels, sorted=True)
        if unique_labels.numel() < n_clusters:
            remap = {int(lbl): i for i, lbl in enumerate(unique_labels.tolist())}
            labels = torch.tensor([remap[int(l.item())] for l in labels],
                                  device=device, dtype=torch.long)
            n_clusters = unique_labels.numel()

        # --- Build de-normalized cluster means for W_fc, and sum columns for W_proj ---
        cluster_means = []
        proj_cols = []
        for cid in range(n_clusters):
            members = (labels == cid).nonzero(as_tuple=True)[0]  # row indices in this cluster

            # Mean in normalized space, then de-normalize
            mean_norm = W_fc_norm[members, :].mean(dim=0, keepdim=False)  # [in_dim]
            mean_vec = mean_norm * col_std.squeeze(0) + col_mean.squeeze(0)
            cluster_means.append(mean_vec)

            # Sum corresponding columns in W_proj (downstream combination of clustered units)
            proj_sum = W_proj[:, members].sum(dim=1, keepdim=True)  # [out_dim, 1]
            proj_cols.append(proj_sum)

        new_fc = torch.stack(cluster_means, dim=0)  # [n_clusters, in_dim]
        new_proj = torch.cat(proj_cols, dim=1)  # [out_dim, n_clusters]

        compressed[module_fc + '.weight'] = new_fc.to(device=device, dtype=dtype)
        compressed[module_proj + '.weight'] = new_proj.to(device=device, dtype=dtype)

        # --- Biases ---
        # c_fc bias: average within each cluster
        if module_fc + '.bias' in params and params[module_fc + '.bias'] is not None:
            b_fc = params[module_fc + '.bias']
            new_b = []
            for cid in range(n_clusters):
                members = (labels == cid).nonzero(as_tuple=True)[0]
                new_b.append(b_fc[members].mean())
            compressed[module_fc + '.bias'] = torch.stack(new_b, dim=0).to(device=device, dtype=dtype)

        # c_proj bias unchanged
        if module_proj + '.bias' in params and params[module_proj + '.bias'] is not None:
            compressed[module_proj + '.bias'] = params[module_proj + '.bias']

        # --- Track new sizes ---
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




class ViT_ModelFolding(BaseViTCompression):
    def __init__(self, model, min_channels=1, compression_ratio=0.5):
        super().__init__(model, min_channels, compression_ratio)

    def compress_function(self, axes, params):
        compressed = {}
        merge_sizes = {}

        # --- Unpack module names ---
        module_fc = axes[0]
        module_proj = axes[1]

        # --- Extract weights ---
        W_fc = params[module_fc + '.weight']  # [hidden_dim, in_dim]
        W_proj = params[module_proj + '.weight']  # [out_dim, hidden_dim]

        device = W_fc.device
        dtype = W_fc.dtype

        # --- Determine number of clusters ---
        n_channels = W_fc.shape[0]
        n_clusters = max(int(n_channels * self.keep_ratio), self.min_channels)
        n_clusters = min(n_clusters, n_channels)

        # --- Column-wise normalization (z-score) for clustering ---
        #     We normalize features (columns) so each dimension contributes comparably.
        eps = torch.finfo(dtype).eps
        col_mean = W_fc.mean(dim=0, keepdim=True)  # [1, in_dim]
        col_std = W_fc.std(dim=0, unbiased=False, keepdim=True) + eps  # [1, in_dim]
        W_fc_norm = (W_fc - col_mean) / col_std  # [hidden_dim, in_dim]

        # --- Clustering on normalized matrix (disable internal normalization) ---
        clusterer = WeightClustering(
            n_clusters=n_clusters,
            method="hkmeans",
            use_pca=False,
            normalize=False  # we already normalized
        )
        labels = clusterer(W_fc_norm).to(device).long()

        _log_cluster_stats(W_fc_norm, labels, module_fc)

        # --- Ensure labels cover all clusters (HKMeans can be sparse in edge cases) ---
        unique_labels = torch.unique(labels, sorted=True)
        if unique_labels.numel() < n_clusters:
            # Collapse to the actually used cluster ids in a stable order
            # (Optional: you could re-run clustering with a smaller k if desired.)
            remap = {int(lbl): i for i, lbl in enumerate(unique_labels.tolist())}
            labels = torch.tensor([remap[int(l.item())] for l in labels], device=device, dtype=torch.long)
            n_clusters = unique_labels.numel()

        # --- Build cluster means in normalized space, then de-normalize ---
        #     mean_norm: [n_clusters, in_dim]; mean_vec = mean_norm * std + mean
        cluster_means = []
        proj_cols = []
        for cid in range(n_clusters):
            members = (labels == cid).nonzero(as_tuple=True)[0]  # indices of rows in this cluster
            # Normalized-space mean vector
            mean_norm = W_fc_norm[members, :].mean(dim=0, keepdim=False)  # [in_dim]
            # De-normalize to original scale
            mean_vec = mean_norm * col_std.squeeze(0) + col_mean.squeeze(0)  # [in_dim]
            cluster_means.append(mean_vec)

            # For the projection layer, sum columns corresponding to the members
            # (since the new hidden unit approximates the mean of members)
            proj_sum = W_proj[:, members].sum(dim=1, keepdim=True)  # [out_dim, 1]
            proj_cols.append(proj_sum)

        new_fc = torch.stack(cluster_means, dim=0)  # [n_clusters, in_dim]
        new_proj = torch.cat(proj_cols, dim=1)  # [out_dim, n_clusters]

        compressed[module_fc + '.weight'] = new_fc.to(device=device, dtype=dtype)
        compressed[module_proj + '.weight'] = new_proj.to(device=device, dtype=dtype)

        # --- Handle biases ---
        if module_fc + '.bias' in params and params[module_fc + '.bias'] is not None:
            b_fc = params[module_fc + '.bias']
            new_b = []
            for cid in range(n_clusters):
                members = (labels == cid).nonzero(as_tuple=True)[0]
                new_b.append(b_fc[members].mean())
            new_b = torch.stack(new_b, dim=0)  # [n_clusters]
            compressed[module_fc + '.bias'] = new_b.to(device=device, dtype=dtype)

        if module_proj + '.bias' in params and params[module_proj + '.bias'] is not None:
            # Next-layer bias remains unchanged
            compressed[module_proj + '.bias'] = params[module_proj + '.bias']

        # --- Track new sizes ---
        merge_sizes[module_fc] = new_fc.shape[0]
        merge_sizes[module_proj] = new_proj.shape[1]

        return compressed, merge_sizes

