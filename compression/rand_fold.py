import torch
from collections import defaultdict
from models.resnet18 import get_axis_to_perm_ResNet18, get_module_by_name_ResNet18
from utils.weight_clustering import axes2perm_to_perm2axes, merge_channel_clustering, NopMerge, _log_cluster_stats
from compression.fold import ResNet18_ModelFolding

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



