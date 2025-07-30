import copy
import warnings

import torch
import itertools
import numpy as np
from collections import defaultdict

from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch, DBSCAN
from sklearn.linear_model._cd_fast import ConvergenceWarning
from hkmeans import HKMeans


class NopMerge:
    def requires(self, arg):
        return False


def axes2perm_to_perm2axes(axes_to_perm):
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return perm_to_axes


class WeightClustering:
    """
    Unified clustering for folding/pruning:
      - Supports HKMeans (default) and multiple sklearn clusterers
      - Optional PCA (default True) and normalization (default False)
      - Can return either cluster labels or binary merge matrix
    """

    def __init__(self, n_clusters: int, n_features: int = None,
        method: str = "hkmeans",       # Options: "hkmeans", "kmeans", "agglomerative", "spectral", "birch", "dbscan"
        normalize: bool = False, use_pca: bool = False, return_matrix: bool = False):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.method = method.lower()
        self.normalize = normalize
        self.use_pca = use_pca
        self.return_matrix = return_matrix

    def _get_clustering_algorithm(self):
        """Return clustering object based on selected method."""
        if self.method == "hkmeans":
            return HKMeans(n_clusters=self.n_clusters, random_state=42, n_init=1, n_jobs=-1,
                           max_iter=10, verbose=False)
        if self.method == "kmeans":
            return KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10, max_iter=300)
        if self.method == "agglomerative":
            return AgglomerativeClustering(n_clusters=self.n_clusters)
        if self.method == "spectral":
            return SpectralClustering(n_clusters=self.n_clusters, assign_labels='kmeans')
        if self.method == "birch":
            return Birch(n_clusters=self.n_clusters)
        if self.method == "dbscan":
            return DBSCAN()
        raise ValueError(f"Unsupported clustering method: {self.method}")

    def __call__(self, tensor: torch.Tensor):
        """
        Cluster input tensor and return either labels or merge matrix.
        Merge matrix is always [clusters, channels].
        """
        # --- Flatten tensor to [channels, features] ---
        X = tensor.view(tensor.shape[0], -1).cpu().numpy()

        # --- Normalize if enabled ---
        if self.normalize:
            X = preprocessing.RobustScaler().fit_transform(X)

        # Replace NaNs/Infs before PCA
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # --- PCA if enabled ---
        if self.use_pca:
            n_samples, n_features = X.shape
            X = PCA(n_components=min(n_samples, n_features)).fit_transform(X)

        # --- Clustering ---
        clustering = self._get_clustering_algorithm()
        # Some clustering methods (e.g., HKMeans) may raise warnings during convergence.
        # E.g., when finding fewer clusters than requested.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clustering.fit(X)
        labels = clustering.labels_

        # Handle fewer-than-expected clusters (KMeans collapse)
        unique_labels = np.unique(labels)
        if len(unique_labels) < self.n_clusters:
            # Remap labels to be contiguous 0..num_clusters-1
            label_map = {old: new for new, old in enumerate(unique_labels)}
            labels = np.array([label_map[l] for l in labels])

        labels_torch = torch.tensor(labels, dtype=torch.int64, device=tensor.device)

        # Return either labels or merge matrix
        if not self.return_matrix:
            return labels_torch

        # Build merge matrix (clusters x channels)
        num_clusters = labels_torch.max().item() + 1
        num_channels = labels_torch.shape[0]
        merge = torch.zeros((num_clusters, num_channels), device=tensor.device, dtype=torch.float32)
        merge.scatter_(0, labels_torch.unsqueeze(0), 1.0)
        merge /= merge.sum(dim=1, keepdim=True).clamp(min=1)

        return merge


# def compress_weight_clustering(perm_to_axes, params, max_ratio=0.5, custom_merger=None):
#     if custom_merger is None:
#         custom_merger = NopMerge()
#
#     new_merge_sizes = {}
#
#     for p_name, axes in perm_to_axes.items():
#         # Original channel count
#         n_channels = params[axes[0][0]].shape[axes[0][1]]
#         n_clusters = max(int(n_channels * max_ratio), 1)
#
#         # Flatten weights for clustering
#         weight = concat_weights(perm_to_axes, params, p_name, n_channels)
#
#         # Cluster → binary merge matrix or labels
#         clusterer = WeightClustering(n_clusters=n_clusters, n_features=n_channels, method="hkmeans",
#                                      normalize=False, use_pca=True)
#         merge = clusterer(weight)
#
#         # *** Ensure same device as params ***
#         param_device = params[axes[0][0]].device
#         merge = merge.to(param_device)
#
#         # Convert labels → binary merge matrix
#         if merge.ndim == 1:
#             labels = merge.to(param_device).long()
#             num_clusters = labels.max().item() + 1
#
#             # *** Log cluster stats here ***
#             _log_cluster_stats(weight, labels, p_name)
#
#             merge_matrix = torch.zeros((num_clusters, labels.shape[0]),
#                                        device=param_device,
#                                        dtype=torch.float32)
#             merge_matrix.scatter_(0, labels.unsqueeze(0), 1.0)
#             merge_matrix /= merge_matrix.sum(dim=1, keepdim=True).clamp(min=1)
#             merge = merge_matrix
#         else:
#             merge = merge.to(torch.float32)
#
#         # Apply folding
#         prm = merge_channel_clustering(perm_to_axes, params, p_name, merge, custom_merger)
#
#         # Update params
#         for wk, axis in axes:
#             params[wk] = prm[wk]
#
#         # Record new size
#         new_merge_sizes[p_name] = params[axes[0][0]].shape[axes[0][1]]
#
#     return params, new_merge_sizes


def get_average_correlation(merge, w):
    avg_corr = np.zeros(merge.shape[0])
    for i in range(avg_corr.shape[0]):
        wh = np.where(merge.cpu().numpy()[i, :] > 0)[0]
        pairs = list(itertools.product(wh, wh))

        assert len(wh) >= 1

        if len(wh) <= 1:
            continue

        pairs = [pair for pair in pairs if pair[0] != pair[1]]
        cnt = 0
        for pair in pairs:
            cnt += 1
            m = pair[0]
            n = pair[1]
            a = w[m, :].flatten()
            b = w[n, :].flatten()
            avg_corr[i] += (a.T @ b) / np.sqrt((a.T @ a) * (b.T @ b))

        avg_corr[i] /= cnt

    avg_corr = torch.tensor(avg_corr).to("cuda").float()

    return avg_corr


def concat_weights(perm_to_axes, params, p_name, n, approx_repair=False):
    A = None
    for wk, axis in perm_to_axes[p_name]:
        w_a = params[wk]

        if "running" in wk or "identity_transform" in wk:
            continue

        if approx_repair is True:
            if axis == 1:
                continue

            if len(w_a.shape) < 2:
                continue

        w_a = torch.movedim(w_a, axis, 0).reshape((n, -1))

        if A is None:
            A = w_a
        else:
            A = torch.hstack((A, w_a))

    return A.cpu()


def merge_channel_clustering(perm_to_axes, params, p_name, merge, custom_merger=None):
    params = copy.deepcopy(params)
    for wk, axis in perm_to_axes[p_name]:
        assert axis in (0, 1)
        if axis == 0:
            p = params[wk].detach().clone()
            if len(p.shape) == 1:
                params[wk] = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ p
            else:
                sh = p.shape
                merge_f = merge.float()
                merged = (torch.diag(1.0 / torch.diag(merge_f @ merge_f.T))) @ merge_f @ p.reshape(sh[0], -1)

                merged = merged.reshape(merge.shape[0], *sh[1:])
                params[wk] = merged
        else:
            p = params[wk].detach().clone()

            if custom_merger.requires(wk):
                params[wk] = custom_merger.merge(wk, p, merge)
            else:
                if len(p.shape) == 2:
                    p = p.permute(1, 0)
                else:
                    p = p.permute(1, 0, 2, 3)

                sh = p.shape
                merged = merge @ p.clone().detach().reshape(sh[0], -1)
                merged = merged.reshape(merge.shape[0], *sh[1:])

                if len(p.shape) == 2:
                    merged = merged.reshape(merge.shape[0], *sh[1:]).permute(1, 0)
                else:
                    merged = merged.reshape(merge.shape[0], *sh[1:]).permute(1, 0, 2, 3)

                params[wk] = merged

    return params


def merge_channel_clustering_approx_repair(perm_to_axes, params, p_name, merge, custom_merger=None):
    params = copy.deepcopy(params)
    n = params[perm_to_axes[p_name][0][0]].shape[perm_to_axes[p_name][0][1]]
    w = concat_weights(perm_to_axes, params, p_name, n)

    avg_corr = get_average_correlation(merge, w.detach().cpu().numpy())

    for wk, axis in perm_to_axes[p_name]:
        if axis == 0:
            p = params[wk].detach().clone()
            if len(p.shape) == 1:
                if "running_mean" in wk:
                    params[wk] = torch.zeros(merge.shape[0])
                    continue

                if "running_var" in wk:
                    params[wk] = torch.ones(merge.shape[0])
                    continue

                params[wk] = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ p

                if "weight" in wk:
                    n_c = torch.sum(merge, axis=1)
                    params[wk] = params[wk] * torch.sqrt(n_c / (1 + (n_c - 1) * avg_corr))
            else:
                sh = p.shape
                merged = (torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge @ p.clone().detach().reshape(sh[0], -1)
                merged = merged.reshape(merge.shape[0], *sh[1:])
                params[wk] = merged
        else:
            p = params[wk].detach().clone()

            if custom_merger.requires(wk):
                params[wk] = custom_merger.merge(wk, p, merge)
            else:
                if len(p.shape) == 2:
                    p = p.permute(1, 0)
                else:
                    p = p.permute(1, 0, 2, 3)

                sh = p.shape
                merged = merge @ p.clone().detach().reshape(sh[0], -1)
                merged = merged.reshape(merge.shape[0], *sh[1:])

                if len(p.shape) == 2:
                    merged = merged.reshape(merge.shape[0], *sh[1:]).permute(1, 0)
                else:
                    merged = merged.reshape(merge.shape[0], *sh[1:]).permute(1, 0, 2, 3)

                params[wk] = merged

    return params



def _log_cluster_stats(W_fc, labels, module_name):
    """
    Compute and log intra-/inter-cluster distance stats (mean L2 norm).
    """
    # Ensure same device
    labels = labels.to(W_fc.device)

    unique_labels = torch.unique(labels)
    intra_dists = []
    centroids = []

    # Compute cluster centroids
    for lbl in unique_labels:
        members = (labels == lbl).nonzero(as_tuple=True)[0].to(W_fc.device)
        cluster_vecs = W_fc[members]
        centroid = cluster_vecs.mean(dim=0)
        centroids.append(centroid)
        intra_dists.append(
            torch.norm(cluster_vecs - centroid, dim=1).mean().item()
        )

    centroids = torch.stack(centroids)
    inter_dist = torch.cdist(centroids, centroids).mean().item()

    print(f"[CLUSTERS] {module_name}: Clusters={len(unique_labels)}, "
          f"Intra={sum(intra_dists)/len(intra_dists):.4f}, Inter={inter_dist:.4f}")



def merge_channel_align(perm_to_axes, params, p_name, merge, unmerge, custom_merger=None):
    params = copy.deepcopy(params)
    for wk, axis in perm_to_axes[p_name]:
        assert axis in (0, 1)

        if axis == 0:
            p = params[wk].detach().clone()

            if len(p.shape) == 1:
                params[wk] = merge @ p
            else:
                sh = p.shape

                merged = merge @ p.clone().detach().reshape(sh[0], -1)
                merged = merged.reshape(merge.shape[0], *sh[1:])
                params[wk] = merged
        else:
            p = params[wk].detach().clone()

            if custom_merger.requires(wk):
                params[wk] = custom_merger.merge(wk, p, merge)
            else:
                if len(p.shape) == 2:
                    p = p.permute(1, 0)
                else:
                    p = p.permute(1, 0, 2, 3)

                sh = p.shape
                merged = (unmerge) @ p.clone().detach().reshape(sh[0], -1)
                merged = merged.reshape((unmerge).shape[0], *sh[1:])

                if len(p.shape) == 2:
                    merged = merged.reshape((unmerge).shape[0], *sh[1:]).permute(1, 0)
                else:
                    merged = merged.reshape((unmerge).shape[0], *sh[1:]).permute(1, 0, 2, 3)

                params[wk] = merged

    return params


def align_weight_clustering(perm_to_axes, axes_to_perm, params_a, params_b, regularizer=1.0, custom_merger=None):
    merge_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}
    merges = dict()
    true_merges = dict()
    unmerges = dict()

    params_a_f = copy.deepcopy(params_a)
    params_b_f = copy.deepcopy(params_b)
    params = copy.deepcopy(params_a)

    if custom_merger is None:
        custom_merger = NopMerge()

    for p_name in merge_sizes.keys():
        print('Compressing block: "' + p_name + '"')
        n = params_a[perm_to_axes[p_name][0][0]].shape[perm_to_axes[p_name][0][1]]

        distance_a = concat_weights(perm_to_axes, params_a, p_name, n)
        distance_b = concat_weights(perm_to_axes, params_b, p_name, n)
        distance = torch.vstack((regularizer * distance_a, (1.0 / regularizer) * distance_b))

        merger = WeightClustering(int(distance.shape[0] * 0.5), distance.shape[0])
        merge, __dict__ = merger(distance)
        merge = merge.cpu()

        true_merges[p_name] = ((torch.diag(1.0 / torch.diag(merge @ merge.T))) @ merge).chunk(2, dim=1)
        merges[p_name] = merge.chunk(2, dim=1)
        unmerges[p_name] = (merge).chunk(2, dim=1)

    for idx, p_name in enumerate(merges.keys()):
        print('Merging block: "' + p_name + '"')
        merge = true_merges[p_name]
        unmerge = unmerges[p_name]

        # Ensure merge/unmerge matrices are on the same device as the target parameter
        device = params_b_f[p_name].device if p_name in params_b_f else next(iter(params_b_f.values())).device
        merge = (merge[0].to(device), merge[1].to(device))
        unmerge = (unmerge[0].to(device), unmerge[1].to(device))

        params_b_f = merge_channel_align(perm_to_axes, params_b_f, p_name, merge[1], unmerge[1],
                                         custom_merger=custom_merger)
        params_a_f = merge_channel_align(perm_to_axes, params_a_f, p_name, merge[0], unmerge[0],
                                         custom_merger=custom_merger)

    for wk in params.keys():
        params[wk] = (params_a_f[wk] + params_b_f[wk])

    new_merge_sizes = {p: params[axes[0][0]].shape[axes[0][1]] for p, axes in perm_to_axes.items()}

    return params, new_merge_sizes
