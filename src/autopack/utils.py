import numpy as np
from scipy.spatial.distance import cdist


def grid_idxs_to_coords(grid_coords: np.ndarray, grid_idxs: np.ndarray) -> np.ndarray:
    """
    From a grid of coordinates and a list of grid indexes, return the
    coordinates of the grid indexes.

    Assumes cost field-like coordinates, i.e. shape (x, y, z, 3).
    """
    return grid_coords[grid_idxs[:, 0], grid_idxs[:, 1], grid_idxs[:, 2]]


def path_length(coords: np.ndarray) -> float:
    """
    Return the length of a path given by an array of N-dimensional
    coordinates, where the first array dimension corresponds to each point in
    the path. I.e. shape (n_points, dim_0, ..., dim_n).
    """
    dist = cdist(coords[:-1, ...], coords[1:, ...])
    return dist[np.diag_indices_from(dist)].sum()
