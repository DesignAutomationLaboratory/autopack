import sys

import panel

from autopack.gui.main_gui import app

if __name__ == "__main__":
    panel.serve(app, show=True)
    sys.exit(0)
