import pathlib

from autopack.data_model import HarnessSetup


def test_harness_setup_from_json(test_scenes_path):
    harness_setup = HarnessSetup.from_json_file(
        test_scenes_path / "simple_plate.harness_setup.json"
    )

    assert isinstance(harness_setup, HarnessSetup)
    # Assert that our relative lookup works
    assert harness_setup.scene_path == pathlib.Path(
        test_scenes_path / "simple_plate.ips"
    )
    assert len(harness_setup.geometries) == 3
    assert len(harness_setup.cables) == 3
