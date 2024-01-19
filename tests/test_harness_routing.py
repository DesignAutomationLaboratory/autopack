import numpy as np
import pytest

from autopack.data_model import Cable, Geometry, HarnessSetup
from autopack.default_commands import create_default_prob_setup
from autopack.ips_communication.ips_commands import create_costfield, route_harnesses


@pytest.mark.parametrize("feasible", [True, False])
def test_topology_feasibility(
    feasible,
    test_scenes_path,
    ips_instance,
):
    if feasible:
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
        grid_resolution=0.02 if feasible else 0.1,
    )

    prob_setup = create_default_prob_setup(
        ips_instance=ips_instance,
        harness_setup=harness_setup,
        create_imma=False,
    )

    harnesses = route_harnesses(
        ips=ips_instance,
        harness_setup=harness_setup,
        cost_field=prob_setup.cost_fields[0],
        bundling_weight=0.7,
        harness_id="test",
    )

    for harness in harnesses:
        assert harness.topology_feasible is feasible
        assert len(harness.cable_segment_order) == len(harness_setup.cables)
        if feasible:
            assert harness.length_in_collision > 0
            assert harness.length_in_collision < harness.length_total
        else:
            assert np.isnan(harness.length_in_collision)
            assert np.isnan(harness.length_total)

        for segment in harness.harness_segments:
            assert len(segment.discrete_nodes) > 0
            assert len(segment.discrete_coords) > 0
            if feasible:
                assert len(segment.smooth_coords) > 0
                assert len(segment.presmooth_coords) > 0
            else:
                assert len(segment.smooth_coords) == 0
                assert len(segment.presmooth_coords) == 0
                assert len(segment.clip_coords) == 0
