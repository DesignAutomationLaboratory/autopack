import itertools
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from . import data_model, logger
from .cost_fields import combine_cost_fields
from .data_model import CostField, Harness, HarnessSegment, HarnessSetup, ProblemSetup
from .ips_communication.ips_class import IPSError, IPSInstance
from .optimization import OptimizationMeta, OptimizationProblem, minimize
from .utils import consecutive_distance, grid_idxs_to_coords, path_length


def route_harnesses(
    ips: IPSInstance,
    harness_setup: HarnessSetup,
    cost_field: CostField,
    bundling_weight: float,
    harness_id: str,
) -> list[Harness]:
    assert not np.isnan(cost_field.costs).any(), "Cost field contains NaNs"
    # IPS seems to go haywire when given negative costs (as in suddenly
    # chewing up all memory)
    assert not np.any(cost_field.costs < 0), "Cost field contains negative values"

    response = ips.call(
        "autopack.routeHarnesses",
        harness_setup,
        cost_field.costs,
        bundling_weight,
        harness_id,
    )

    def gen_harness_segments(segment_dict):
        for segment in segment_dict:
            discrete_nodes = np.array(segment["discreteNodes"])
            discrete_coords = grid_idxs_to_coords(
                grid_coords=cost_field.coordinates, grid_idxs=discrete_nodes
            )

            yield HarnessSegment(
                radius=segment["radius"],
                cables=segment["cables"],
                discrete_nodes=discrete_nodes,
                discrete_coords=discrete_coords,
                # Make sure that these are always arrays with the last dim=3, even if empty
                presmooth_coords=np.array(segment["presmoothCoords"]).reshape(-1, 3),
                smooth_coords=np.array(segment["smoothCoords"]).reshape(-1, 3),
                clip_coords=np.array(segment["clipPositions"]).reshape(-1, 3),
            )

    solutions = [
        Harness(
            name=solution["name"],
            topology_feasible=solution["topologyFeasible"],
            harness_segments=list(gen_harness_segments(solution["segments"])),
            cable_segment_order=solution["cableSegmentOrder"],
            num_branch_points=solution["numBranchPoints"],
            bundling_weight=solution["objectiveWeightBundling"],
            bundling_objective=solution["solutionObjectiveBundling"],
            length_objective=solution["solutionObjectiveLength"],
            length_total=solution["lengthTotal"],
            length_in_collision=solution["lengthInCollision"],
        )
        for solution in response
    ]
    return solutions


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


def smooth_cost(harness: data_model.Harness, cost_field: CostField) -> float:
    """
    Returns the cost of the harness through the cost field, calculated
    as the integral of the cost field along the smooth path of the
    harness.
    """
    segment_costs = []
    for segment in harness.harness_segments:
        coords = segment.smooth_coords
        costs = cost_field.interpolate(coords)
        cum_distance_along_path = np.concatenate(
            [[0], consecutive_distance(coords)]
        ).cumsum()
        segment_costs.append(np.trapz(costs, cum_distance_along_path))

    return np.sum(segment_costs)


def clip_costs(
    harnesses: list[data_model.Harness],
    cost_fields: list[CostField],
    cost_field_dim_name: str,
):
    for harness in harnesses:
        all_clip_coords = harness.all_clip_coords
        values = np.array(
            [cost_field.interpolate(all_clip_coords) for cost_field in cost_fields]
        )
        yield xr.DataArray(
            values.T,
            dims=["clip", cost_field_dim_name],
            coords={
                "clip": range(len(all_clip_coords)),
                cost_field_dim_name: [cf.name for cf in cost_fields],
            },
        )


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

    ips_cost_field = problem_setup.ref_cost_field
    ergo_cost_fields = [cf for cf in problem_setup.cost_fields if cf.ergo]

    if ergo_cost_fields:
        _clip_ergo_values = list(
            clip_costs(
                harness_solutions,
                ergo_cost_fields,
                cost_field_dim_name="ergo_standard",
            )
        )
        clip_ergo_values = xr.concat(_clip_ergo_values, dim="solution")
        mean_clip_ergo_value = clip_ergo_values.sel(ergo_standard="REBA").mean("clip")
        mean_clip_ergo_value.attrs["objective"] = True

        ergo_variables = {
            "clip_ergo_values": clip_ergo_values,
            "mean_clip_ergo_value": mean_clip_ergo_value,
        }
    else:
        ergo_variables = {}

    num_clips = np.array(
        [
            (
                np.sum([len(seg.clip_coords) for seg in harness.harness_segments])
                if harness.topology_feasible
                else -1
            )
            for harness in harness_solutions
        ]
    )

    ds = xr.Dataset(
        data_vars={
            "cost_field_weight": xr.DataArray(
                np.tile(x[:-1], (num_ips_solutions, 1)),
                dims=["solution", "cost_field"],
            ),
            "bundling_weight": xr.DataArray(
                np.tile(x[-1], num_ips_solutions),
                dims=["solution"],
            ),
            "num_clips": xr.DataArray(
                num_clips,
                dims=["solution"],
                attrs={
                    "title": "Number of clips",
                    "objective": True,
                },
            ),
            "volume": xr.DataArray(
                [harness_volume(h) for h in harness_solutions],
                dims=["solution"],
                attrs={
                    "title": "Volume",
                    "description": "Total volume of the harness.",
                    "units": "m^3",
                    "objective": True,
                },
            ),
            "collision_length": xr.DataArray(
                [h.length_in_collision for h in harness_solutions],
                dims=["solution"],
                attrs={
                    "title": "Collision length",
                    "description": "Total length of the harness that is in collision or violates the clearance constraint.",
                    "units": "m",
                    "constraint": True,
                },
            ),
            "collision_ratio": xr.DataArray(
                [h.length_in_collision / h.length_total for h in harness_solutions],
                dims=["solution"],
                attrs={
                    "title": "Collision ratio",
                    "description": "Ratio of the total length of the harness that is in collision or violates the clearance constraint.",
                    "units": None,
                },
            ),
            "geometry_penalty": xr.DataArray(
                [
                    smooth_cost(harness=h, cost_field=ips_cost_field)
                    for h in harness_solutions
                ],
                dims=["solution"],
                attrs={
                    "title": "Geometry penalty",
                    "description": "Integral of the cost field along the smooth path of the harness.",
                    "units": None,
                    "objective": True,
                },
            ),
            **ergo_variables,
            # FIXME: this doesn't work with zarr
            "harness": xr.DataArray(
                np.array(harness_solutions, dtype=object),
                dims=["solution"],
                attrs={
                    "title": "Harness object",
                },
            ),
        },
        coords={
            "timestamp": xr.DataArray(
                np.tile(pd.Timestamp.utcnow(), num_ips_solutions),
                dims=["solution"],
            ),
            "category": xr.DataArray(
                np.tile(meta.category, num_ips_solutions),
                dims=["solution"],
            ),
            "batch_idx": xr.DataArray(
                np.tile(meta.batch, num_ips_solutions),
                dims=["solution"],
            ),
            "iter_idx": xr.DataArray(
                np.tile(iter_in_batch, num_ips_solutions),
                dims=["solution"],
            ),
            "ips_idx": xr.DataArray(
                range(num_ips_solutions),
                dims=["solution"],
            ),
        },
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

    # Get all objectives. NOTE that they are not scaled
    objs = batch_ds.filter_by_attrs(objective=True).to_stacked_array(
        new_dim="obj", sample_dims=["solution"], name="objectives"
    )

    cons = batch_ds.filter_by_attrs(constraint=True).to_stacked_array(
        new_dim="con", sample_dims=["solution"], name="constraints"
    )

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
        cons.values,
    )


def build_optimization_problem(
    ips: IPSInstance, problem_setup: ProblemSetup
) -> OptimizationProblem:
    ergo_available = problem_setup.ergo_settings is not None
    num_cost_fields = len(problem_setup.cost_fields)
    num_objectives = 4 if ergo_available else 3
    weights_bounds = np.array([[0.001, 1.0]] * num_cost_fields)
    bundling_weight_bounds = np.array([[0.05, 0.9]])
    bounds = np.array([*weights_bounds, *bundling_weight_bounds])

    batch_datasets = []

    def batch_analysis_func(xs: np.ndarray, meta: OptimizationMeta):
        this_batch_dataset = xr.concat(
            (
                design_point_ds(
                    ips=ips,
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
        num_constraints=1,
        state={"batch_datasets": batch_datasets},
    )
