from os import PathLike
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
from . import lua_commands


def create_costfield(ips_instance, harness_setup):
    response = ips_instance.call("autopack.getCostField", harness_setup)

    coords = np.array(response["coords"])
    costs = np.array(response["costs"])

    ips_field = CostField(name="ips", coordinates=coords, costs=costs)
    length_field = CostField(
        name="length",
        coordinates=coords,
        costs=np.ones_like(costs),
    )
    return ips_field, length_field


def load_scene(ips_instance, scene_file_path: PathLike, clear=False):
    if clear:
        ips_instance.call("autopack.clearScene")
    return ips_instance.call("autopack.loadAndFitScene", str(scene_file_path))


def save_scene(ips, scene_file_path: PathLike):
    return ips.call("autopack.saveScene", str(scene_file_path))


def route_harnesses(
    ips: IPSInstance,
    harness_setup: HarnessSetup,
    cost_field: CostField,
    bundling_weight: float,
    harness_id: str,
) -> list[Harness]:
    assert not np.isnan(cost_field.costs).any(), "Cost field contains NaNs"

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
                presmooth_coords=np.array(segment["presmoothCoords"]),
                smooth_coords=np.array(segment["smoothCoords"]),
                clip_coords=np.array(segment["clipPositions"]),
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


def get_stl_meshes(ips_instance):
    command = lua_commands.get_stl_meshes()
    print(command)
    str_meshes = ips_instance.eval(command)
    print(str_meshes)
