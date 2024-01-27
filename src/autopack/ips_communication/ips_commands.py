import os
from typing import Optional

import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np

from autopack.data_model import (
    CostField,
    Harness,
    HarnessSegment,
    HarnessSetup,
    IPSInstance,
)

from ..utils import grid_idxs_to_coords


def create_costfield(ips, harness_setup):
    response = ips.call("autopack.getCostField", harness_setup)

    coords = np.array(response["coords"])
    costs = np.array(response["costs"])

    return CostField(name="IPS", coordinates=coords, costs=costs)


def load_scene(ips: IPSInstance, scene_file_path: os.PathLike, clear=False):
    """
    Load the scene file at `scene_file_path`. Optionally clears the
    scene before loading if `clear` is True.
    """
    assert os.path.isabs(scene_file_path), "Scene file path must be absolute"
    assert os.path.exists(
        scene_file_path
    ), f"Scene file does not exist at {scene_file_path}"
    if clear:
        ips.call("autopack.clearScene")
    return ips.call("autopack.loadAndFitScene", scene_file_path)


def save_scene(ips: IPSInstance, scene_file_path: os.PathLike):
    assert os.path.isabs(scene_file_path), "Scene file path must be absolute"
    return ips.call("autopack.saveScene", scene_file_path)


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


def add_point_cloud(
    ips,
    coords,
    colors=None,
    parent_name=None,
    name=None,
    replace_existing=False,
    visible=True,
):
    if colors is None:
        colors = np.ones_like(coords) * np.array([[0, 0, 1]])
    ips.call(
        "autopack.createColoredPointCloud",
        np.hstack([coords, colors]),
        parent_name,
        name,
        replace_existing,
        visible,
        return_result=False,
    )


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
