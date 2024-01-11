import pathlib
import sys

import logbook
import panel

from autopack import USER_DIR, __version__, logger

DEBUG_LOG_PATH = USER_DIR / "debug.log"

log_handler = logbook.NestedSetup(
    [
        # Log debug info to file
        logbook.RotatingFileHandler(
            DEBUG_LOG_PATH,
            max_size=20 * 1024 * 1024,  # 20 MiB
            backup_count=5,
            level="DEBUG",
        ),
        # Log info to stdout, and also bubble it up to the debug log
        logbook.StreamHandler(sys.stdout, level="INFO", bubble=True),
    ]
)

if __name__ == "__main__":
    with log_handler.applicationbound():
        logger.notice(f"Starting Autopack v{__version__}")
        logger.notice(f"Writing debug log to {DEBUG_LOG_PATH}")
        # Import here so we get the logging up and running before
        from autopack.gui.main_gui import make_gui

        app = make_gui()

        panel.serve(app, show=True)
    sys.exit(0)
