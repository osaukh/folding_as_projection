import torch
from collections import defaultdict
from models.resnet import get_axis_to_perm_ResNet18, get_module_by_name_ResNet
from utils.weight_clustering import axes2perm_to_perm2axes, merge_channel_clustering, NopMerge, _log_cluster_stats

from compression.fold import ResNet18_ModelFolding, PreActResNet18_ModelFolding
from compression.base_clip_vit import BaseCLIPViTCompression


class ResNet18_RandomFolding(ResNet18_ModelFolding):
    def compress_function(self, axes, params):
        """
        Random folding with balanced clusters:
        each channel assigned exactly once, clusters non-empty.
        """
        n_channels = params[axes[0][0]].shape[axes[0][1]]

        # Number of clusters (respects keep_ratio, at least 1)
        n_clusters = max(int(n_channels * self.keep_ratio), 1)

        # Random clustering requirements:
        # - Every cluster is non-empty (fills one element per cluster first).
        # - Each channel is assigned to exactly one cluster.
        # Start with one element per cluster to guarantee non-emptiness
        base = torch.arange(n_clusters, device=self.device)
        # Fill remaining channels randomly
        extra = torch.randint(0, n_clusters, (n_channels - n_clusters,), device=self.device) \
            if n_channels > n_clusters else torch.tensor([], device=self.device, dtype=torch.long)
        # Combine and shuffle
        labels = torch.cat([base, extra])
        labels = labels[torch.randperm(n_channels, device=self.device)]

        # Log cluster stats
        _log_cluster_stats(params[axes[0][0]], labels, axes[0][0])

        # Merge matrix (rows=clusters, cols=channels)
        merge_matrix = torch.zeros((n_clusters, n_channels), device=self.device)
        merge_matrix.scatter_(0, labels.unsqueeze(0), 1.0)
        merge_matrix /= merge_matrix.sum(dim=1, keepdim=True).clamp(min=1)

        # Merge weights consistently across group
        compressed_params = merge_channel_clustering({0: axes}, params, 0, merge_matrix, custom_merger=NopMerge())

        return compressed_params, {'cluster_labels': labels}



class CLIPViT_RandomFolding(BaseCLIPViTCompression):
    def compress_function(self, axes, params):
        """
        Random folding for CLIP ViT:
        - Random balanced cluster assignment for c_fc output channels
        - Compute centroids per cluster
        - Apply centroid-averaged channels to c_proj input
        """
        compressed, merge_sizes = {}, {}

        # Unpack module names
        module_fc, _ = axes[0]     # c_fc
        module_proj, _ = axes[1]   # c_proj

        # Extract weights
        W_fc = params[module_fc]       # [hidden_dim, in_dim]
        W_proj = params[module_proj]   # [out_dim, hidden_dim]
        device = W_fc.device

        # Determine number of clusters
        n_channels = W_fc.shape[0]
        n_clusters = max(int(n_channels * self.keep_ratio), self.min_channels)

        # --- Balanced random clustering ---
        base = torch.arange(n_clusters, device=device)
        extra = torch.randint(0, n_clusters, (n_channels - n_clusters,), device=device) if n_channels > n_clusters else torch.tensor([], device=device, dtype=torch.long)
        labels = torch.cat([base, extra])[torch.randperm(n_channels, device=device)]

        _log_cluster_stats(W_fc, labels, module_fc)

        # --- Compute centroids for each cluster ---
        centroids = []
        for k in range(n_clusters):
            cluster_indices = (labels == k).nonzero(as_tuple=True)[0]
            cluster_mean = W_fc[cluster_indices].mean(dim=0, keepdim=True)
            centroids.append(cluster_mean)
        new_fc = torch.cat(centroids, dim=0)  # [n_clusters, in_dim]

        # Apply same cluster averaging to c_proj input (transpose logic)
        new_proj = []
        for k in range(n_clusters):
            cluster_indices = (labels == k).nonzero(as_tuple=True)[0]
            cluster_mean = W_proj[:, cluster_indices].mean(dim=1, keepdim=True)
            new_proj.append(cluster_mean)
        new_proj = torch.cat(new_proj, dim=1)  # [out_dim, n_clusters]

        # Assign compressed weights
        compressed[module_fc + '.weight'] = new_fc
        compressed[module_proj + '.weight'] = new_proj

        # Bias handling
        if module_fc + '.bias' in params and params[module_fc + '.bias'] is not None:
            fc_bias = []
            for k in range(n_clusters):
                cluster_indices = (labels == k).nonzero(as_tuple=True)[0]
                cluster_mean = params[module_fc + '.bias'][cluster_indices].mean().unsqueeze(0)
                fc_bias.append(cluster_mean)
            compressed[module_fc + '.bias'] = torch.cat(fc_bias, dim=0)

        if module_proj + '.bias' in params and params[module_proj + '.bias'] is not None:
            compressed[module_proj + '.bias'] = params[module_proj + '.bias']  # unchanged

        # Track new sizes
        merge_sizes[module_fc] = new_fc.shape[0]
        merge_sizes[module_proj] = new_proj.shape[1]

        return compressed, merge_sizes


class PreActResNet18_RandomFolding(PreActResNet18_ModelFolding):
    def compress_function(self, axes, params):
        """
        Random folding with balanced clusters:
        each channel assigned exactly once, clusters non-empty.
        """
        n_channels = params[axes[0][0]].shape[axes[0][1]]

        # Number of clusters (respects keep_ratio, at least 1)
        n_clusters = max(int(n_channels * self.keep_ratio), 1)

        # Random clustering requirements:
        # - Every cluster is non-empty (fills one element per cluster first).
        # - Each channel is assigned to exactly one cluster.
        # Start with one element per cluster to guarantee non-emptiness
        base = torch.arange(n_clusters, device=self.device)
        # Fill remaining channels randomly
        extra = torch.randint(0, n_clusters, (n_channels - n_clusters,), device=self.device) \
            if n_channels > n_clusters else torch.tensor([], device=self.device, dtype=torch.long)
        # Combine and shuffle
        labels = torch.cat([base, extra])
        labels = labels[torch.randperm(n_channels, device=self.device)]

        # Log cluster stats
        _log_cluster_stats(params[axes[0][0]], labels, axes[0][0])

        # Merge matrix (rows=clusters, cols=channels)
        merge_matrix = torch.zeros((n_clusters, n_channels), device=self.device)
        merge_matrix.scatter_(0, labels.unsqueeze(0), 1.0)
        merge_matrix /= merge_matrix.sum(dim=1, keepdim=True).clamp(min=1)

        # Merge weights consistently across group
        compressed_params = merge_channel_clustering({0: axes}, params, 0, merge_matrix, custom_merger=NopMerge())

        return compressed_params, {'cluster_labels': labels}