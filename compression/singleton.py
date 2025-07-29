import torch
from utils.weight_clustering import merge_channel_clustering, NopMerge, _log_cluster_stats

from compression.fold import ResNet18_ModelFolding, PreActResNet18_ModelFolding
from compression.base_clip_vit import BaseCLIPViTCompression


class ResNet18_Singleton(ResNet18_ModelFolding):
    def compress_function(self, axes, params):
        """
        Singleton folding:
        - Top (k-1) channels (highest L2 norm) form singleton clusters.
        - Remaining channels grouped into one cluster.
        """
        # Number of channels and clusters
        n_channels = params[axes[0][0]].shape[axes[0][1]]
        n_clusters = max(int(n_channels * self.keep_ratio), 1)

        # Compute per-channel L2 norm (flatten along remaining dimensions)
        weight = params[axes[0][0]]
        magnitudes = torch.norm(weight.view(n_channels, -1), p=2, dim=1)

        # Get indices of top (k-1) channels
        if n_clusters > 1:
            topk = torch.topk(magnitudes, n_clusters - 1, largest=True).indices
        else:
            topk = torch.tensor([], device=self.device, dtype=torch.long)

        # Build labels: top-k are unique, others share one cluster
        labels = torch.full((n_channels,), n_clusters - 1, device=self.device, dtype=torch.long)
        for idx, ch in enumerate(topk):
            labels[ch] = idx

        # Log cluster stats
        _log_cluster_stats(weight, labels, axes[0][0])

        # Build merge matrix
        merge_matrix = torch.zeros((n_clusters, n_channels), device=self.device)
        merge_matrix.scatter_(0, labels.unsqueeze(0), 1.0)
        merge_matrix /= merge_matrix.sum(dim=1, keepdim=True).clamp(min=1)

        # Merge weights (folding)
        compressed_params = merge_channel_clustering({0: axes}, params, 0, merge_matrix, custom_merger=NopMerge())

        return compressed_params, {'cluster_labels': labels}


class CLIPViT_Singleton(BaseCLIPViTCompression):
    def compress_function(self, axes, params):
        """
        Singleton folding for CLIP ViT:
        - Top (k-1) channels form singleton clusters
        - Remaining channels merged into the last cluster
        - Centroids used for merging
        """
        compressed, merge_sizes = {}, {}

        module_fc, _ = axes[0]     # c_fc
        module_proj, _ = axes[1]   # c_proj

        # Extract weights
        W_fc = params[module_fc]       # [hidden_dim, in_dim]
        W_proj = params[module_proj]   # [out_dim, hidden_dim]
        device = W_fc.device

        # Determine number of clusters
        n_channels = W_fc.shape[0]
        n_clusters = max(int(n_channels * self.keep_ratio), 1)

        # --- Magnitude ranking ---
        mags = torch.norm(W_fc, p=2, dim=1)  # L2 norm per channel
        sorted_idx = torch.argsort(mags, descending=True)

        # Top (k-1) are singletons, rest go to last cluster
        if n_clusters > 1:
            singleton_idx = sorted_idx[:n_clusters - 1]
            remainder_idx = sorted_idx[n_clusters - 1:]
        else:
            singleton_idx = torch.tensor([], dtype=torch.long, device=device)
            remainder_idx = sorted_idx

        # Build labels
        labels = torch.empty(n_channels, dtype=torch.long, device=device)
        # Assign singletons to clusters 0..k-2
        for i, idx in enumerate(singleton_idx):
            labels[idx] = i
        # Assign remainder to cluster k-1
        labels[remainder_idx] = n_clusters - 1

        _log_cluster_stats(W_fc, labels, module_fc)

        # --- Compute centroids for clusters ---
        centroids = []
        for k in range(n_clusters):
            cluster_indices = (labels == k).nonzero(as_tuple=True)[0]
            cluster_mean = W_fc[cluster_indices].mean(dim=0, keepdim=True)
            centroids.append(cluster_mean)
        new_fc = torch.cat(centroids, dim=0)  # [n_clusters, in_dim]

        # Apply same clustering to c_proj (input side)
        new_proj = []
        for k in range(n_clusters):
            cluster_indices = (labels == k).nonzero(as_tuple=True)[0]
            cluster_mean = W_proj[:, cluster_indices].mean(dim=1, keepdim=True)
            new_proj.append(cluster_mean)
        new_proj = torch.cat(new_proj, dim=1)  # [out_dim, n_clusters]

        # Assign compressed weights
        compressed[module_fc + '.weight'] = new_fc
        compressed[module_proj + '.weight'] = new_proj

        # Bias folding
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


class PreActResNet18_Singleton(PreActResNet18_ModelFolding):
    def compress_function(self, axes, params):
        """
        Singleton folding:
        - Top (k-1) channels (highest L2 norm) form singleton clusters.
        - Remaining channels grouped into one cluster (averaged).
        BN handling:
        - Singletons keep their BN stats directly.
        - Merged cluster averages its BN stats.
        """
        n_channels = params[axes[0][0]].shape[axes[0][1]]
        n_clusters = max(int(n_channels * self.keep_ratio), 1)

        # Compute L2 norm for importance
        weight = params[axes[0][0]]
        magnitudes = torch.norm(weight.contiguous().view(n_channels, -1), p=2, dim=1)

        # Select top-(n_clusters-1) for singleton clusters
        if n_clusters > 1:
            topk = torch.topk(magnitudes, n_clusters - 1, largest=True).indices
        else:
            topk = torch.tensor([], device=self.device, dtype=torch.long)

        # Initialize all to merged cluster (last cluster ID)
        labels = torch.full((n_channels,), n_clusters - 1, device=self.device, dtype=torch.long)
        for idx, ch in enumerate(topk):
            labels[ch] = idx  # Assign singleton IDs

        # Debug cluster stats
        _log_cluster_stats(weight, labels, axes[0][0])

        # Build merge matrix
        merge_matrix = torch.zeros((n_clusters, n_channels), device=self.device)
        merge_matrix.scatter_(0, labels.unsqueeze(0), 1.0)
        merge_matrix /= merge_matrix.sum(dim=1, keepdim=True).clamp(min=1)

        # Validate singleton count
        unique_labels = torch.unique(labels)
        singleton_rows = unique_labels[unique_labels != (n_clusters - 1)]
        assert len(singleton_rows) <= (n_clusters - 1), \
            f"Expected ≤ {n_clusters-1} singletons, got {len(singleton_rows)}"

        # Merge weights (folding)
        compressed_params = merge_channel_clustering({0: axes}, params, 0, merge_matrix, custom_merger=NopMerge())

        # --- Special BN handling ---
        for name in list(compressed_params.keys()):
            # Process only 1D BN parameters (weight, bias, running stats)
            if compressed_params[name].ndim != 1:
                continue
            if name not in params:  # Safety check
                continue

            orig_tensor = params[name]
            if orig_tensor.ndim != 1:
                continue

            new_tensor = torch.zeros(n_clusters, device=self.device)
            for k in range(n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    if k == n_clusters - 1:
                        # Merged cluster: average stats
                        new_tensor[k] = orig_tensor[mask].mean()
                    else:
                        # Singleton cluster: copy exact value
                        new_tensor[k] = orig_tensor[mask][0]

            compressed_params[name] = new_tensor

        return compressed_params, {'cluster_labels': labels}


