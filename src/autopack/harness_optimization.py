import warnings

import numpy as np
import pandas as pd
import xarray as xr
from linear_operator.utils.warnings import NumericalWarning

from . import data_model
from .data_model import CostField, IPSInstance, ProblemSetup
from .ips_communication.ips_commands import route_harnesses
from .optimization import OptimizationMeta, OptimizationProblem, minimize
from .utils import path_length


def normalize_costs(costs: np.ndarray, scale=(0, 1)) -> np.ndarray:
    """
    Min-max normalize costs to `scale` (default [0, 1]). If min ==
    max, all costs are set to 0. Infeasible nodes are always kept as
    is.
    """

    finite_mask = np.isfinite(costs)
    min_value = np.amin(costs[finite_mask])
    max_value = np.amax(costs[finite_mask])
    if min_value == max_value:
        unit_costs = np.zeros_like(costs)
    else:
        unit_costs = (costs - min_value) / (max_value - min_value)
    scaled_costs = unit_costs * (scale[1] - scale[0]) + scale[0]
    # inf * 0 = nan, so we must set inf again where it was before
    scaled_costs[~finite_mask] = np.inf
    return scaled_costs


def combine_cost_fields(cost_fields, weights, output_scale=(1, 10)):
    assert len(cost_fields) == len(weights), "Must give one weight per cost field"
    coords = cost_fields[0].coordinates

    all_costs = np.stack(
        [normalize_costs(cf.costs, scale=(0, 1)) for cf in cost_fields]
    )
    combined_costs = np.sum(all_costs * weights[:, None, None, None], axis=0)
    combined_scaled_costs = normalize_costs(combined_costs, output_scale)

    return CostField(name="Combined", coordinates=coords, costs=combined_scaled_costs)


def harness_volume(harness: data_model.Harness) -> float:
    # Use the first segments to see what kind of coordinates we have
    first_segment = harness.harness_segments[0]

    if len(first_segment.smooth_coords) > 0:
        segment_lengths = np.array(
            [path_length(seg.smooth_coords) for seg in harness.harness_segments]
        )
    elif len(first_segment.presmooth_coords) > 0:
        segment_lengths = np.array(
            [path_length(seg.presmooth_coords) for seg in harness.harness_segments]
        )
    else:
        segment_lengths = np.array(
            [path_length(seg.discrete_coords) for seg in harness.harness_segments]
        )

    segment_radii = np.array([seg.radius for seg in harness.harness_segments])

    return np.sum(segment_lengths * segment_radii**2 * np.pi)


def design_point_ds(
    ips: IPSInstance,
    problem_setup: ProblemSetup,
    meta: OptimizationMeta,
    x: np.ndarray,
    iter_in_batch: int,
) -> xr.Dataset:
    """
    Evaluates a design point and returns a dataset with the results.
    """
    case_id = f"{meta.batch}.{iter_in_batch}"
    cost_field_ids = [cf.name for cf in problem_setup.cost_fields]

    combined_cost_field = combine_cost_fields(
        cost_fields=problem_setup.cost_fields,
        weights=x[:-1],
        output_scale=(1, 10),
    )

    harness_solutions = route_harnesses(
        ips=ips,
        harness_setup=problem_setup.harness_setup,
        cost_field=combined_cost_field,
        bundling_weight=x[-1],
        harness_id=case_id,
    )

    num_ips_solutions = len(harness_solutions)

    bundle_total_costs = np.array(
        [
            [
                evaluate_harness(harness, cost_field)
                for cost_field in problem_setup.cost_fields
            ]
            for harness in harness_solutions
        ]
    )
    bundle_costs = bundle_total_costs[:, :, 0]
    total_costs = bundle_total_costs[:, :, 1]

    ds = xr.Dataset(
        {
            "meta.timestamp": xr.DataArray(
                np.tile(pd.Timestamp.utcnow(), num_ips_solutions),
                dims=["solution"],
            ),
            "meta.category": xr.DataArray(
                np.tile(meta.category, num_ips_solutions),
                dims=["solution"],
            ),
            "meta.batch_idx": xr.DataArray(
                np.tile(meta.batch, num_ips_solutions),
                dims=["solution"],
            ),
            "meta.iter_idx": xr.DataArray(
                np.tile(iter_in_batch, num_ips_solutions),
                dims=["solution"],
            ),
            "meta.ips_idx": xr.DataArray(
                range(num_ips_solutions),
                dims=["solution"],
            ),
            "cost_field_weight": xr.DataArray(
                np.tile(x[:-1], (num_ips_solutions, 1)),
                dims=["solution", "cost_field"],
            ),
            "bundling_weight": xr.DataArray(
                np.tile(x[-1], num_ips_solutions),
                dims=["solution"],
            ),
            "bundling_cost": xr.DataArray(
                bundle_costs,
                dims=["solution", "cost_field"],
            ),
            "total_cost": xr.DataArray(
                total_costs,
                dims=["solution", "cost_field"],
            ),
            "num_estimated_clips": xr.DataArray(
                [h.numb_of_clips for h in harness_solutions],
                dims=["solution"],
            ),
            "harness_volume": xr.DataArray(
                [harness_volume(h) for h in harness_solutions],
                dims=["solution"],
            ),
            # FIXME: this doesn't work with zarr
            "harness": xr.DataArray(
                np.array(harness_solutions, dtype=object),
                dims=["solution"],
            ),
        }
    )
    ds = ds.assign_coords(
        solution=[h.name for h in harness_solutions],
        cost_field=cost_field_ids,
    )
    return ds


def batch_voi(
    batch_ds: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes the dataset for a full batch and returns the variables of
    interest (xs, objs, cons)
    """
    objs = batch_ds[
        ["bundling_cost", "total_cost", "num_estimated_clips"]
    ].to_stacked_array(new_dim="obj", sample_dims=["solution"], name="objectives")
    cons = np.empty((objs.shape[0], 0), dtype=float)

    # Index the xs by the solution from obj, so we get the corresponding
    # x for each solution
    xs = (
        batch_ds[["cost_field_weight", "bundling_weight"]]
        .to_stacked_array(
            new_dim="desvar", sample_dims=["solution"], name="design_variables"
        )
        .sel(solution=objs.solution)
    )

    return (
        xs.values,
        objs.values,
        cons,
    )


def problem_from_setup(problem_setup, ips_instance) -> OptimizationProblem:
    num_cost_fields = len(problem_setup.cost_fields)
    num_dims = num_cost_fields + 1
    num_objectives = 2 * num_cost_fields + 1
    weights_bounds = np.array([[0.001, 1.0]] * num_cost_fields)
    bundling_weight_bounds = np.array([[0.05, 0.9]])
    bounds = np.array([*weights_bounds, *bundling_weight_bounds])

    batch_datasets = []

    def batch_analysis_func(xs: np.ndarray, meta: OptimizationMeta):
        this_batch_dataset = xr.concat(
            (
                design_point_ds(
                    ips=ips_instance,
                    problem_setup=problem_setup,
                    meta=meta,
                    x=x,
                    iter_in_batch=i,
                )
                for i, x in enumerate(xs)
            ),
            dim="solution",
        )
        batch_datasets.append(this_batch_dataset)

        return batch_voi(this_batch_dataset)

    return OptimizationProblem(
        func=batch_analysis_func,
        bounds=bounds,
        num_objectives=num_objectives,
        num_constraints=0,
        state={"batch_datasets": batch_datasets},
    )


def global_optimize_harness(
    ips_instance: data_model.IPSInstance,
    problem_setup: data_model.ProblemSetup,
    init_samples: int = 8,
    batches: int = 4,
    batch_size: int = 4,
) -> xr.Dataset:
    if problem_setup.harness_setup.allow_infeasible_topology and batches > 1:
        raise ValueError(
            "Cannot optimize harnesses with infeasible topology. Set allow_infeasible_topology to False in the harness setup or run with batches=0 to disable optimization."
        )

    problem = problem_from_setup(problem_setup, ips_instance)

    with warnings.catch_warnings(category=NumericalWarning, action="ignore"):
        minimize(
            problem=problem,
            batches=batches,
            batch_size=batch_size,
            init_samples=init_samples,
            seed=0,
        )

    dataset = xr.concat(problem.state["batch_datasets"], dim="solution")
    dataset.attrs["problem_setup"] = problem_setup
    dataset.attrs["init_samples"] = init_samples
    dataset.attrs["batches"] = batches
    dataset.attrs["batch_size"] = batch_size
    dataset.attrs["ips_version"] = ips_instance.version

    return dataset
