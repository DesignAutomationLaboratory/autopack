import pathlib
import sys

import logbook
import panel

from autopack import USER_DIR, __version__, logger

DEBUG_LOG_PATH = USER_DIR / "debug.log"


def get_log_handler():
    return logbook.NestedSetup(
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


def init_app():
    USER_DIR.mkdir(parents=True, exist_ok=True)

    log_handler = get_log_handler()
    log_handler.push_application()
    logger.notice(f"Starting Autopack v{__version__}")
    logger.notice(f"Writing debug log to {DEBUG_LOG_PATH}")

    # Import here so we get the logging up and running before
    from autopack.gui.main_gui import make_gui  # noqa: E402

    app = make_gui()

    return app


if __name__ == "__main__":
    # Running as a script

    panel.serve(init_app(), show=True)
    sys.exit(0)

if __name__.startswith("bokeh_app"):
    # Running under `panel serve ...` or `bokeh serve ...`

    init_app().servable()
