import pathlib
from importlib.metadata import version

import logbook

__version__ = version("autopack")

logger = logbook.Logger(__name__, level=logbook.DEBUG)

USER_DIR = pathlib.Path.home() / ".autopack"
