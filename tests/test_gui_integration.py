import pytest
import xarray as xr

from autopack.gui.main_gui import AutopackApp, init_panel


@pytest.fixture(autouse=True)
def mock_gui_paths(monkeypatch, tmp_path):
    """
    Mocks the paths used in the main_gui module to avoid polluting the
    user's home directory.
    """
    # FIXME: this is a bit ugly. The real fix would be to make the paths
    # used in main_gui injectable instead of hardcoded.
    with monkeypatch.context() as m:
        m.setattr("autopack.gui.main_gui.USER_DIR", tmp_path / "mock_user_dir")
        m.setattr("autopack.gui.main_gui.SESSIONS_DIR", tmp_path / "mock_sessions_dir")
        m.setattr(
            "autopack.gui.main_gui.SETTINGS_PATH",
            tmp_path / "gui-settings.json",
        )
        yield


@pytest.fixture
def app(ips_path):
    init_panel()
    app = AutopackApp()
    # Start the app to make sure that stuff doesn't break
    app_view = app.view()
    app_thread = app_view.show(threaded=True)
    yield app
    app_thread.stop()


def test_app_run_problem_smoke(app, test_scenes_path):
    assert app.dataset is None

    app.problem_path = str(test_scenes_path / "simple_plate.harness_setup.json")
    app.runtime_settings.run_ergo = True
    app.runtime_settings.ergo_use_rbpp = False
    app.runtime_settings.ergo_sample_ratio = 0.0001
    app.runtime_settings.opt_budget = 1
    app.runtime_settings.doe_samples = 2

    initial_session_path = app.session_path
    app.param.trigger("run_problem")

    run_dataset = app.dataset
    assert isinstance(run_dataset, xr.Dataset)
    assert len(run_dataset.solution) >= 8

    assert app.viz_manager.dataset is run_dataset

    # Make sure the session path changed to the newly created one
    assert app.session_path != initial_session_path
    # Try loading the session back
    app.param.trigger("load_session")

    loaded_dataset = app.dataset

    # We don't expect the loaded dataset to be *the same* object as the
    # run dataset (although it should be equal)
    assert loaded_dataset is not run_dataset
    assert isinstance(loaded_dataset, xr.Dataset)
