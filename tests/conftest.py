import os
import pathlib

import pytest


@pytest.fixture(autouse=True)
def mock_dirs(tmpdir, monkeypatch):
    """
    Makes sure that USER_DIR and derived paths are mocked to temporary
    directories during tests. This must happen before any tests import
    them.
    """
    import autopack
    from autopack.gui import main_gui

    tmpdir = pathlib.Path(tmpdir)

    user_dir = tmpdir / "mock_user_dir"
    sessions_dir = tmpdir / "mock_sessions_dir"
    gui_settings_path = user_dir / "gui-settings.json"

    with monkeypatch.context() as m:
        # FIXME: it is really ugly to have to mock these in two places.
        # Less should be done on import!
        m.setattr(autopack, "USER_DIR", user_dir)
        m.setattr(autopack, "SESSIONS_DIR", sessions_dir)
        m.setattr(main_gui, "USER_DIR", user_dir)
        m.setattr(main_gui, "SESSIONS_DIR", sessions_dir)
        m.setattr(main_gui, "SETTINGS_PATH", gui_settings_path)
        yield


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
def ips(skip_in_ci):
    from autopack.ips import IPSInstance

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
