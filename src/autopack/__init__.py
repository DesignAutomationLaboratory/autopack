from importlib.metadata import version

import logbook

__version__ = version("autopack")

logger = logbook.Logger(__name__, level=logbook.DEBUG)
