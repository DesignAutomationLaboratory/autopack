from typing import Literal

import numpy as np
import torch
import xarray as xr
from botorch.utils.multi_objective.pareto import is_non_dominated

from .data_model import DatasetVariableAttrs


def functions_of_interest(dataset: xr.Dataset) -> xr.Dataset:
    """
    Returns a dataset with only the functions of interest (objectives
    and constraints) of the input dataset as data variables.
    """

    data_vars = [
        name
        for name, var in dataset.data_vars.items()
        if var.attrs.get("objective", False) or var.attrs.get("constraint", False)
    ]

    return dataset[data_vars]


def score_multipliers(dataset: xr.Dataset) -> xr.Dataset:
    """
    Returns a dataset with the score multipliers for all data variables
    in the input dataset, that can be used to scale the data variables
    according to their score direction.
    """

    return dataset.assign(
        {
            name: var.attrs.get("score_direction", -1)
            for name, var in dataset.data_vars.items()
        }
    )


def only_dims(dataset: xr.Dataset, dims: list[str]) -> xr.Dataset:
    """
    Returns a dataset with only the specified `dims`. See
    `xarray.Dataset.drop_dims` for details on the `errors` argument.
    """
    return dataset.drop_dims(set(dataset.dims) - set(dims))


def only_dtypes(dataset: xr.Dataset, dtypes: list[type | np.dtype]) -> xr.Dataset:
    """
    Returns a dataset with the data variables that have a dtype in
    `dtypes`.
    """

    vars = [
        name
        for name, var_dtype in dataset.dtypes.items()
        if any(np.issubdtype(var_dtype, dtype) for dtype in dtypes)
    ]

    return dataset[vars]


def non_dominated_mask(dataset: xr.Dataset) -> xr.DataArray:
    """
    Returns a boolean mask of the non-dominated points in the dataset,
    in a DataArray. Considers constraints as objectives, meaning that it
    may return non-feasible points if they constitute viable trade-offs.
    """
    # objective_ds = dataset.filter_by_attrs(objective=True)
    # constraint_ds = dataset.filter_by_attrs(constraint=True)
    # voi_ds = xr.merge([objective_ds, constraint_ds])
    foi_ds = functions_of_interest(dataset)

    # Scale the functions of interest by their score direction
    scaler_ds = score_multipliers(foi_ds)
    scaled_darray = (foi_ds * scaler_ds).to_dataarray()

    # BoTorch's is_non_dominated assumes maximization and requires a
    # tensor as input
    mask = is_non_dominated(
        torch.tensor(scaled_darray.values.T, device="cpu"),
        deduplicate=False,
    )

    mask_darray = xr.DataArray(
        mask,
        dims=["solution"],
        coords=foi_ds.coords,
        attrs={
            "title": "Optimal",
            "description": "Whether the point is non-dominated (Pareto-optimal), considering all objectives and constraints",
        },
    )

    return DatasetVariableAttrs.validate_dataarray(mask_darray)
