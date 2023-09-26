import pathlib

import pytest


@pytest.fixture
def test_scenes_path():
    return pathlib.Path(__file__).parent / "scenes"


@pytest.fixture
def ips_instance():
    from autopack.ips_communication.ips_class import IPSInstance

    ips = IPSInstance()
    ips.start()
    yield ips
    ips.kill()
