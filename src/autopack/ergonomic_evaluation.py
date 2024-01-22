import numpy as np
import xarray as xr
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist

from autopack import logger
from autopack.data_model import Cable, CostField, Geometry, HarnessSetup, ProblemSetup
from autopack.ips_communication.ips_class import IPSInstance
from autopack.ips_communication.ips_commands import (
    add_point_cloud,
)
from autopack.utils import farthest_point_sampling

ERGO_STANDARD_BOUNDS = xr.DataArray(
    [[1, 15], [1, 7]],
    coords={
        "ergo_standard": ["REBA", "RULA"],
        "bound": ["min", "max"],
    },
)


def create_ergonomic_cost_field(
    ips: IPSInstance,
    harness_setup: HarnessSetup,
    ref_cost_field: CostField,
    # Max allowed gripping tolerance [m]
    max_grip_diff=0.01,
    # Multiplies the grid resolution to get the max distance between samples [-]
    relative_sample_resolution=6,
    use_rbpp=True,
    update_screen=False,
    keep_generated_objects=False,
):
    # Only evaluate the standards we can handle
    ergo_standards = ERGO_STANDARD_BOUNDS.ergo_standard.values

    logger.info(f"Creating ergonomy cost fields: {', '.join(ergo_standards)}")
    ref_costs = ref_cost_field.costs
    ref_coords = ref_cost_field.coordinates
    ref_coords_flat = ref_coords.reshape(-1, 3)
    # No point in evaluating infeasible points
    feasible_mask = np.invert(np.isposinf(ref_costs)).reshape(-1)

    families: list[dict[str, str]] = ips.call("autopack.getAllManikinFamilies")
    logger.info(f"Found {len(families)} manikin families")
    if not families:
        logger.warning(
            "No manikin families found in scene, ergonomy cost fields will be empty"
        )
        return []

    geometries_to_consider = [
        geo.name for geo in harness_setup.geometries if geo.assembly
    ]

    # Max ratio of grid points to evaluate
    max_sampling_ratio = 1 / relative_sample_resolution**3
    max_samples = round(ref_coords_flat.shape[0] * max_sampling_ratio)
    max_farthest_distance = harness_setup.grid_resolution * relative_sample_resolution
    logger.info(
        f"Picking at most {max_samples} points spaced {max_farthest_distance} meters apart, out of {ref_coords_flat.shape[0]}, using farthest point sampling"
    )
    eval_coords = farthest_point_sampling(
        points=ref_coords_flat,
        num_points=max_samples,
        # The RBF interpolator requires at least 4 points
        min_points=4,
        max_farthest_distance=max_farthest_distance,
        seed=0,  # For deterministic behavior
    )
    num_coords = eval_coords.shape[0]

    add_point_cloud(
        ips=ips,
        coords=eval_coords,
        name="Autopack ergo evaluation points",
        replace_existing=True,
        visible=False,
    )

    family_datasets = []
    for family in families:
        family_id = family["id"]
        family_name = family["name"]
        family_manikin_names = family["manikinNames"]
        logger.info(f"Evaluating {num_coords} points with {family_name}")
        ergo_eval = ips.call(
            "autopack.evalErgo",
            geometries_to_consider,
            family_id,
            eval_coords,
            max_grip_diff,
            ergo_standards,
            use_rbpp,
            update_screen,
            keep_generated_objects,
        )

        _bad_grip_mask = np.vectorize(lambda msg: "Grip was not satisfied" in msg)(
            np.array(ergo_eval["errorMsgs"], dtype=object)
        )

        family_ds = xr.Dataset(
            data_vars={
                "ergo_values": xr.DataArray(
                    ergo_eval["ergoValues"],
                    dims=["coord", "hand", "ergo_standard", "manikin"],
                ),
                "reachable": xr.DataArray(
                    np.invert(_bad_grip_mask),
                    dims=["coord", "hand"],
                ),
            },
            coords={
                # "coord": range(num_coords),
                "hand": ["left", "right"],
                "ergo_standard": ergo_standards,
                "manikin": family_manikin_names,
            },
        ).assign_coords(family=family_name)
        family_datasets.append(family_ds)

    ds = xr.concat(family_datasets, dim="family")
    aggregated_ergo_values = (
        ds.ergo_values
        # The worst of all manikins within each family
        .max("manikin")
        # Where we can't reach, assign the worst possible value for each
        # ergo standard
        .where(
            cond=ds.reachable,
            # Assigns where the condition is False
            other=ERGO_STANDARD_BOUNDS.sel(bound="max", drop=True),
        )
        # Use the best family and hand available
        .min(["family", "hand"])
    )
    num_unreachable_coords = (
        np.invert(ds.reachable.any(["family", "hand"])).sum().item()
    )
    logger.notice(
        f"{num_unreachable_coords} points are unreachable for all manikin families"
    )

    cost_fields = []
    for ergo_std in ergo_standards:
        logger.info(f"Interpolating {ergo_std} cost field")
        interpolator = RBFInterpolator(
            eval_coords, aggregated_ergo_values.sel(ergo_standard=ergo_std).values
        )
        predicted_costs = interpolator(ref_coords_flat)

        ergo_std_bounds = ERGO_STANDARD_BOUNDS.sel(ergo_standard=ergo_std).values
        predicted_costs = np.clip(
            predicted_costs, ergo_std_bounds[0], ergo_std_bounds[1]
        )

        # Infeasible points have not been evaluated, so we know nothing
        # about them
        predicted_costs[~feasible_mask] = np.inf

        cost_fields.append(
            CostField(
                name=ergo_std,
                coordinates=ref_coords,
                costs=predicted_costs.reshape(ref_costs.shape),
                interpolator=interpolator,
            )
        )

    return cost_fields
