"""Functions to infer a rough channel map from ephys data, used for microwire brush arrays"""

import numpy as np
from sklearn.manifold import Isomap


def infer_map(signal, knn_k=5, scale_method="max", scale_params=None):
    """Use Isomap algorithm to infer 2D channel map from correlations within signal array

    args:
        signal (np.ndarray): 2D numpy array of shape (n_samples, n_channels)
        knn_k (int): number of nearest neighbors for isomap knn graph
        scale_method (str): fed to scale_map, see that function for details
        scale_params (dict): fed to scale_map, see that function for details

    returns:
        X (np.ndarray): channel map coordinates, of shape (n_channels, 2)
    """
    # default scaling
    if scale_method == "max" and not scale_params:
        scale_params = {"max_dist": 350}

    C = signal_to_corr(signal)
    D = corr_to_dist(C)
    X = dist_to_coords(D, knn_k)
    X = scale_map(X, scale_method, scale_params)
    X = rotate_pc1_vertical(X)
    return X


def signal_to_corr(signal):
    """Generate correlations from signal"""
    C = np.corrcoef(signal, rowvar=False)
    return C


def corr_to_dist(C):
    """Convert pairwise channel correlation matrix C to distance matrix D"""
    D = 1 - C
    return D


def dist_to_coords(D, knn_k):
    """Use Isomap to infer channel coordinates from locally valid, approximate distances"""
    isomap = Isomap(
        n_neighbors=knn_k, n_components=2, metric="precomputed", eigen_solver="dense"
    )
    X = isomap.fit_transform(D)
    return X


def scale_map(X, scale_method, scale_params):
    """Scale inferred channel map (scale all distances by a linear factor)

    args:
        X (np.ndarray): (N x 2) map of inferred channel coordinates
        scale_method (str): method for determining linear factor
        scale_params (dict): free parameters for scale method

    returns:
        X (np.ndarray): scaled coordinates

    See helper functions (scale_by_*) for scale method details
    """
    if scale_method == "radius":
        return scale_by_radius(X, scale_params["n_chan"], scale_params["r"])
    elif scale_method == "max":
        return scale_by_max(X, scale_params["max_dist"])
    else:
        raise NotImplementedError(f"{scale_method} is not a valid scale method")


def scale_by_radius(X, n_chan, r):
    from sklearn.neighbors import kneighbors_graph

    """Scale the inferred coordinates (X) so that no more than n_chan channels lie within a circle of radius r (approx.)"""
    G = kneighbors_graph(X, n_chan, mode="distance", include_self=True).toarray()
    # get smallest distance to the n_chan-th nearest neighbor
    min_dist = G.max(axis=1).min()
    # scale s.t. the smallest distance to the n_chan-th nearest neighbor is r
    X_out = X / min_dist * r
    return X_out


def scale_by_max(X, max_dist):
    """Scale the inferred coordinates (X) so that the maximum distance between any two points is max_dist"""
    from scipy.spatial.distance import pdist

    X = X / pdist(X).max() * max_dist
    return X


def rotate_pc1_vertical(X):
    """Helper function used to ensure similar/identical channel maps are visibly similar/identical

    Note that rotation does not alter how the channel map is processed by kilosort, which cares only about distances between channels.
    """
    from sklearn.decomposition import PCA

    pc1 = PCA(1).fit(X).components_[0]
    theta = np.arctan(pc1[1] / pc1[0]) - np.pi / 2

    def R(x):
        return np.array([[np.cos(x), np.sin(x)], [-np.sin(x), np.cos(x)]])

    X_rot = np.matmul(R(theta), X.T).T
    return X_rot
