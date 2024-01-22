import numpy as np
import xarray as xr
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


def farthest_point_sampling(
    points, num_points, max_farthest_distance=None, min_points=1, seed=None
):
    """
    Selects a subset of points using farthest point sampling. Terminates
    when the number of points is `num_points` or (optionally) when the
    distance to the farthest point is equal to or less than
    `max_farthest_distance`. In both cases, the minimum number of points
    is `min_points`.
    """

    # Ensure the number of points to select is not greater than the total number of points
    num_points = np.clip(num_points, min_points, len(points))
    assert min_points <= len(points), "Not enough points to satisfy min_points"

    # Initialize an empty list to store selected points
    selected_idxs = []

    # Randomly choose the first point
    rand = np.random.default_rng(seed=seed)
    first_point_idx = rand.choice(len(points))
    first_point = points[first_point_idx]
    selected_idxs.append(first_point_idx)

    # Calculate distances from the selected point to all other points
    distances = cdist(first_point[np.newaxis, :], points)

    # Iteratively select the farthest point
    for _ in range(num_points - 1):
        # Find the point with the maximum minimum distance
        min_distances = np.min(distances, axis=0)
        farthest_point_idx = np.argmax(min_distances)
        farthest_distance = min_distances[farthest_point_idx]

        # Update the distances array
        distances = np.minimum(
            distances, cdist(points[farthest_point_idx][np.newaxis, :], points)
        )

        # Add the farthest point to the selected points
        selected_idxs.append(farthest_point_idx)

        if (
            max_farthest_distance
            and farthest_distance <= max_farthest_distance
            and len(selected_idxs) >= min_points
        ):
            break

    return points[selected_idxs]


def normalize(data: xr.DataArray | xr.Dataset, dim=None):
    """
    Normalizes a DataArray or Dataset along the specified dimension(s)
    to the range [0, 1].
    """
    data_min = data.min(dim=dim)
    data_max = data.max(dim=dim)

    return (data - data_min) / (data_max - data_min)
