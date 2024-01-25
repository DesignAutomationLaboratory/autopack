import numpy as np
import pytest

from autopack.data_model import Cable, Geometry, HarnessSetup
from autopack.ips_communication.ips_commands import route_harnesses
from autopack.workflows import build_problem


@pytest.mark.parametrize("feasible_scene", [True, False])
@pytest.mark.parametrize("allow_infeasible_topology", [True, False])
def test_topology_feasibility(
    feasible_scene,
    allow_infeasible_topology,
    test_scenes_path,
    ips_instance,
):
    if feasible_scene:
        scene_path = test_scenes_path / "topology_feasible.ips"
    else:
        scene_path = test_scenes_path / "topology_infeasible.ips"
    harness_setup = HarnessSetup(
        scene_path=str(scene_path.resolve()),
        geometries=[
            Geometry(
                name="Static Geometry",
                clearance=1.0,
                preference="Near",
                clipable=True,
                assembly=True,
            ),
        ],
        cables=[
            Cable(
                start_node=f"start_{n}",
                end_node=f"end_{n}",
                # Thick enough to collide in the gap
                cable_type="ID_Rubber_template",
            )
            for n in range(1, 4)
        ],
        # The feasible scene needs a fine enough grid to find a solution
        grid_resolution=0.02 if feasible_scene else 0.1,
        allow_infeasible_topology=allow_infeasible_topology,
    )

    prob_setup = build_problem(
        ips=ips_instance,
        harness_setup=harness_setup,
    )

    harnesses = route_harnesses(
        ips=ips_instance,
        harness_setup=harness_setup,
        cost_field=prob_setup.cost_fields[0],
        bundling_weight=0.7,
        harness_id="test",
    )

    if allow_infeasible_topology:
        expected_feasibility = feasible_scene
        assert len(harnesses) >= 1
    else:
        # If we don't allow infeasible topologies, we should get no solutions
        expected_feasibility = True
        if feasible_scene:
            assert len(harnesses) >= 1
        else:
            assert len(harnesses) == 0

    for harness in harnesses:
        assert harness.topology_feasible is expected_feasibility
        assert len(harness.cable_segment_order) == len(harness_setup.cables)
        if feasible_scene:
            assert harness.length_in_collision > 0
            assert harness.length_in_collision < harness.length_total
        else:
            assert np.isnan(harness.length_in_collision)
            assert np.isnan(harness.length_total)

        for segment in harness.harness_segments:
            assert len(segment.discrete_nodes) > 0
            assert len(segment.discrete_coords) > 0
            if feasible_scene:
                assert len(segment.smooth_coords) > 0
                assert len(segment.presmooth_coords) > 0
            else:
                assert len(segment.smooth_coords) == 0
                assert len(segment.presmooth_coords) == 0
                assert len(segment.clip_coords) == 0
