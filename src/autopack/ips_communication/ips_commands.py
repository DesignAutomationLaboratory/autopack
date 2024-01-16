from os import PathLike
from typing import Optional

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
    solutions_to_capture: Optional[list[int]] = None,
    smooth_solutions: bool = False,
    build_discrete_solutions: bool = False,
    build_presmooth_solutions: bool = False,
    build_smooth_solutions: bool = False,
    build_cable_simulations: bool = False,
) -> list[Harness]:
    assert not np.isnan(cost_field.costs).any(), "Cost field contains NaNs"

    response = ips.call(
        "autopack.routeHarnesses",
        harness_setup,
        cost_field.costs,
        bundling_weight,
        harness_id,
        solutions_to_capture or [],
        smooth_solutions,
        build_discrete_solutions,
        build_presmooth_solutions,
        build_smooth_solutions,
        build_cable_simulations,
    )

    def gen_harness_segments(segment_dict):
        for segment in segment_dict:
            discrete_nodes = np.array(segment["discreteNodes"])
            discrete_coords = grid_idxs_to_coords(
                grid_coords=cost_field.coordinates, grid_idxs=discrete_nodes
            )
            _smooth_coords = segment.get("smoothCoords", None)
            smooth_coords = (
                np.array(_smooth_coords) if _smooth_coords is not None else None
            )
            _clip_coords = segment.get("clipPositions", None)
            clip_coords = np.array(_clip_coords) if _clip_coords is not None else None

            yield HarnessSegment(
                radius=segment["radius"],
                cables=segment["cables"],
                discrete_nodes=discrete_nodes,
                discrete_coords=discrete_coords,
                presmooth_coords=np.array(segment["presmoothCoords"]),
                smooth_coords=smooth_coords,
                clip_coords=clip_coords,
            )

    solutions = [
        Harness(
            name=solution["name"],
            harness_segments=list(gen_harness_segments(solution["segments"])),
            numb_of_clips=solution["estimatedNumClips"],
            num_branch_points=solution["numBranchPoints"],
            bundling_weight=solution["objectiveWeightBundling"],
            bundling_objective=solution["solutionObjectiveBundling"],
            length_objective=solution["solutionObjectiveLength"],
        )
        for solution in response
    ]
    return solutions


def check_distance_of_points(
    ips_instance, harness_setup, coords, min_geometry_dist, max_geometry_dist
):
    geo_names = [geo.name for geo in harness_setup.geometries if geo.assembly]
    coord_distances_to_geo = np.array(
        ips_instance.call("autopack.coordDistancesToGeo", coords, geo_names, True)
    )

    return np.logical_and(
        coord_distances_to_geo >= min_geometry_dist,
        coord_distances_to_geo <= max_geometry_dist,
    )


def cost_field_vis(ips_instance, cost_field):
    command = lua_commands.add_cost_field_vis(cost_field)
    ips_instance.eval(command)


def get_stl_meshes(ips_instance):
    command = lua_commands.get_stl_meshes()
    print(command)
    str_meshes = ips_instance.eval(command)
    print(str_meshes)
