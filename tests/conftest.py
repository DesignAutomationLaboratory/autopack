import os
import pathlib

import pytest


@pytest.fixture
def test_scenes_path():
    return pathlib.Path(__file__).parent / "scenes"


@pytest.fixture
def in_ci():
    # See https://docs.github.com/en/actions/learn-github-actions/variables
    return os.getenv("GITHUB_ACTIONS", "false") == "true"


@pytest.fixture
def skip_in_ci(in_ci):
    if in_ci:
        pytest.skip("Skipping in CI")
    return


@pytest.fixture
def ips_instance(skip_in_ci):
    from autopack.ips_communication.ips_class import IPSInstance

    ips = IPSInstance()
    ips.start()
    yield ips
    ips.kill()


@pytest.fixture
def simple_plate_harness_setup(test_scenes_path):
    from autopack import data_model

    return data_model.HarnessSetup(
        scene_path=test_scenes_path / "simple_plate.ips",
        geometries=[
            data_model.Geometry(
                name="part1",
                clearance=5.0,
                preference="Near",
                clipable=True,
                assembly=True,
            ),
            data_model.Geometry(
                name="part2",
                clearance=5.0,
                preference="Near",
                clipable=True,
                assembly=True,
            ),
            data_model.Geometry(
                name="part3",
                clearance=5.0,
                preference="Near",
                clipable=True,
                assembly=True,
            ),
        ],
        cables=[
            data_model.Cable(
                start_node="Cable1_start",
                end_node="Cable1_end",
                cable_type="ABS_sensor_cable",
            ),
            data_model.Cable(
                start_node="Cable2_start",
                end_node="Cable2_end",
                cable_type="ABS_sensor_cable",
            ),
            data_model.Cable(
                start_node="Cable3_start",
                end_node="Cable3_end",
                cable_type="ABS_sensor_cable",
            ),
        ],
        # Coarse grid resolution to speed up tests
        grid_size=10,
    )
