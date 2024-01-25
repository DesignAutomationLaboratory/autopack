from autopack.data_model import ErgoSettings, StudySettings
from autopack.gui.main_gui import RuntimeSettings


def test_runtime_settings_outputs():
    settings = RuntimeSettings()
    assert isinstance(settings.to_study_settings(), StudySettings)

    settings.run_ergo = False
    assert settings.to_ergo_settings() is None

    settings.run_ergo = True
    assert isinstance(settings.to_ergo_settings(), ErgoSettings)
