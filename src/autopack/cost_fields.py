import numpy as np
import xarray as xr
from matplotlib import cm, colors
from scipy.interpolate import RBFInterpolator

from autopack import logger
from autopack.data_model import CostField, ErgoSettings, HarnessSetup
from autopack.ips import IPSInstance
from autopack.utils import add_point_cloud, farthest_point_sampling

ERGO_STANDARD_BOUNDS = xr.DataArray(
    [[1, 15], [1, 7]],
    coords={
        "ergo_standard": ["REBA", "RULA"],
        "bound": ["min", "max"],
    },
)


def create_ips_cost_field(ips, harness_setup):
    response = ips.call("autopack.getCostField", harness_setup)

    coords = np.array(response["coords"])
    costs = np.array(response["costs"])

    return CostField(name="IPS", coordinates=coords, costs=costs)


def create_ergo_cost_fields(
    ips: IPSInstance,
    harness_setup: HarnessSetup,
    ergo_settings: ErgoSettings,
    ref_cost_field: CostField,
    update_screen=False,
    keep_generated_objects=False,
):
    # Only evaluate the standards we can handle
    ergo_standards = ERGO_STANDARD_BOUNDS.ergo_standard.values

    sample_ratio = ergo_settings.sample_ratio

    logger.info(
        f"Creating ergonomy cost fields {', '.join(ergo_standards)} with a sampling ratio of {sample_ratio:.3%} along each axis"
    )
    ref_costs = ref_cost_field.costs
    ref_coords = ref_cost_field.coordinates
    ref_coords_flat = ref_coords.reshape(-1, 3)
    # No point in evaluating infeasible points
    feasible_mask = np.isfinite(ref_costs).reshape(-1)
    feasible_coords = ref_coords_flat[feasible_mask]

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

    # 4 is a hard requirement for the RBF interpolator, but that is not
    # viable in 3D space
    min_samples = ergo_settings.min_samples
    # Accomodate for a size*sample_ratio number of points along each axis
    max_samples = np.ceil(ref_cost_field.size * sample_ratio).prod().astype(int)
    # ...and scale the resolution by the sample ratio to get point distance
    max_farthest_distance = ref_cost_field.resolution / sample_ratio
    logger.info(
        f"Picking {min_samples} to {max_samples} points, nominally spaced {max_farthest_distance:.2f} meters apart, out of {feasible_coords.shape[0]}, using farthest point sampling"
    )
    eval_coords = farthest_point_sampling(
        points=feasible_coords,
        num_points=max_samples,
        min_points=min_samples,
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
            ergo_settings.grip_tolerance,
            ergo_standards,
            ergo_settings.use_rbpp,
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
                ergo=True,
            )
        )

    return cost_fields


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


def cost_field_vis(ips: IPSInstance, cost_field, visible=True):
    coords = cost_field.coordinates.reshape(-1, 3)
    costs = cost_field.costs.reshape(-1)
    finite_mask = np.isfinite(costs)

    norm = colors.Normalize()
    norm.autoscale(costs[finite_mask])
    norm_costs = norm(costs)

    cmap = cm.get_cmap("viridis")
    cmap.set_over("red")
    # Gets the colors and drops the alpha channel
    point_colors = cmap(norm_costs)[:, :-1]

    add_point_cloud(
        ips=ips,
        coords=coords,
        colors=point_colors,
        parent_name="Autopack cost fields",
        name=cost_field.name,
        replace_existing=True,
        visible=visible,
    )
