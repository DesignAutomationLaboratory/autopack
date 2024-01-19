import datetime
import pathlib
import shutil
import tkinter as tk
from tkinter import filedialog

import holoviews as hv
import hvplot.xarray  # noqa: F401
import panel as pn
import param
import xarray as xr
from panel import theme
from plotly import express as px

from autopack import USER_DIR, __version__, logger
from autopack.data_model import HarnessSetup
from autopack.default_commands import create_default_prob_setup
from autopack.harness_optimization import global_optimize_harness
from autopack.io import load_session, save_session
from autopack.ips_communication.ips_class import IPSInstance
from autopack.utils import normalize

SETTINGS_PATH = USER_DIR / "gui-settings.json"
SESSIONS_DIR = USER_DIR / "sessions"

# For some reason, the sidebar is borked in MaterialTemplate.
SIDEBAR_CSS_HACK = """
#sidebar .mdc-list {
    width: 100%;
}
"""


def exception_handler(exc):
    logger.exception(exc)
    pn.state.notifications.error(
        "Something went wrong. See the console for details.",
    )


def init_panel():
    pn.extension(
        "tabulator",
        "plotly",
        sizing_mode="stretch_width",
        design=theme.Material,
        raw_css=[SIDEBAR_CSS_HACK],
        loading_indicator=True,
        exception_handler=exception_handler,
        notifications=True,
    )
    hv.extension("bokeh")


def make_session_name(problem_path: pathlib.Path):
    problem_name = problem_path.stem
    timestamp = datetime.datetime.utcnow()
    timestamp_str = timestamp.strftime("%Y-%m-%dT%H%M%S")
    return f"{problem_name}.{timestamp_str}.v{__version__}"


def section_header(text):
    return pn.pane.HTML(f"<b>{text}</b>")


def open_problem_path_dialog(
    initial_dir=None,
    title="Select a file",
):
    window = tk.Tk()
    window.wm_attributes("-topmost", 1)
    window.withdraw()
    path = filedialog.askopenfilename(
        parent=window,
        initialdir=initial_dir,
        title=title,
        filetypes=[("JSON files", "*.json")],
    )
    window.destroy()
    return pathlib.Path(path)


def open_directory_dialog(
    initial_dir=None,
    title="Select a directory",
):
    window = tk.Tk()
    window.wm_attributes("-topmost", 1)
    window.withdraw()
    path = filedialog.askdirectory(
        parent=window,
        initialdir=initial_dir,
        title=title,
    )
    window.destroy()
    return pathlib.Path(path)


def prune_dataset_for_viz(ds: xr.Dataset, drop_meta=False):
    dims_to_drop = ["cost_field"]
    vars_to_drop = ["harness"]
    if drop_meta:
        vars_to_drop.extend(
            var.name for var in ds.data_vars.values() if "meta." in var.name
        )
    return ds.drop_dims(dims_to_drop).drop_vars(vars_to_drop)


def to_viz_dataframe(ds: xr.Dataset, drop_meta=False):
    return prune_dataset_for_viz(ds=ds, drop_meta=drop_meta).to_dataframe()


class Settings(param.Parameterized):
    ips_path = param.Path(check_exists=False, label="IPS path")
    last_problem_path = param.Path(check_exists=False)
    last_session_path = param.Path(check_exists=False)

    @classmethod
    def load_or_new(cls):
        if SETTINGS_PATH.exists():
            return cls(**cls.param.deserialize_parameters(SETTINGS_PATH.read_text()))
        else:
            return cls()

    @param.depends("ips_path", "last_problem_path", "last_session_path", watch=True)
    def persist(self):
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        settings_to_persist = set(self.param.params().keys()) - {"name"}
        SETTINGS_PATH.write_text(
            self.param.serialize_parameters(subset=settings_to_persist)
        )


class PostProcessor(param.Parameterized):
    original_dataset = param.Parameter()
    processed_dataset = param.Parameter()
    processed_dataframe = param.Parameter()

    # `update` is a reserved name in Panel
    update_postproc = param.Event(label="Update post-processing")

    @param.depends("update_postproc", "original_dataset", watch=True, on_init=True)
    def _update_postproc(self):
        ds: xr.Dataset = self.original_dataset

        if ds is None:
            self.processed_dataset = None
            self.processed_dataframe = None
            return

        # FIXME: do stuff
        processed_ds = ds
        processed_df = (
            processed_ds.drop_dims("cost_field").drop_vars("harness").to_dataframe()
        )

        self.processed_dataset = processed_ds
        self.processed_dataframe = processed_df

    @param.depends("original_dataset")
    def view(self):
        section_attrs = {"sizing_mode": "stretch_both"}
        return pn.Row(
            pn.WidgetBox(
                section_header("Filtering"),
                pn.widgets.Checkbox(name="Foo", value=False),
                **section_attrs,
            ),
            pn.WidgetBox(
                section_header("Clustering"),
                pn.widgets.Checkbox(name="Bar", value=False),
                **section_attrs,
            ),
            pn.WidgetBox(
                section_header("Summary"),
                f"Number of solutions: {len(self.processed_dataframe)}",
                **section_attrs,
            ),
            pn.WidgetBox(
                section_header("Actions"),
                pn.widgets.Button.from_param(
                    self.param.update_postproc,
                    button_type="primary",
                ),
                "With selected solutions:",
                pn.widgets.Button(
                    name="Baz",
                    button_type="default",
                ),
                **section_attrs,
            ),
        )


class DataTable(param.Parameterized):
    dataset = param.Parameter()
    _dataframe = param.Parameter()

    @param.depends("dataset", watch=True, on_init=True)
    def update_dataframe(self):
        if self.dataset is None:
            self._dataframe = None
        else:
            self._dataframe = to_viz_dataframe(self.dataset)

    @param.depends("_dataframe")
    def view(self):
        if self.dataset is None:
            return pn.widgets.StaticText(value="No dataset loaded")
        return pn.WidgetBox(
            pn.widgets.Tabulator(
                self._dataframe,
                layout="fit_data_fill",
                # Disables editing
                disabled=True,
                height=500,
            )
        )


class InteractiveScatterPlot(param.Parameterized):
    dataset = param.Parameter()
    x = param.Selector()
    y = param.Selector()
    color = param.Selector(default=None)
    by = param.Selector(default=None)

    @param.depends("dataset", watch=True, on_init=True)
    def update_choices(self):
        if self.dataset is None:
            return
        else:
            selectable_dims = list(self.dataset.dims.keys())
            selectable_vars = list(self.dataset.data_vars.keys())

        self.param.x.objects = selectable_vars
        self.x = selectable_vars[0]
        self.param.y.objects = selectable_vars
        self.y = selectable_vars[1]
        self.param.color.objects = [None, *selectable_vars]
        self.color = None
        self.param.by.objects = [None, *selectable_dims]
        self.by = None

    def _plot_view(self):
        return pn.pane.HoloViews(
            self.dataset.hvplot.scatter(
                x=self.x,
                y=self.y,
                c=self.color,
                by=self.by,
                groupby=[],
                colormap="viridis",
                colorbar=True,
                responsive=True,
            ).opts(title=""),
        )

    def _select_view(self):
        return pn.Row(
            self.param.x,
            self.param.y,
            self.param.color,
            self.param.by,
        )

    @param.depends("dataset", "x", "y", "color", "by")
    def view(self):
        if self.dataset is None:
            return pn.widgets.StaticText(value="No dataset loaded")
        return pn.WidgetBox(
            self._select_view(),
            self._plot_view(),
        )


class ParallelCoordinatesPlot(param.Parameterized):
    dataset = param.Parameter()
    _dataframe = param.DataFrame()

    @param.depends("dataset", watch=True, on_init=True)
    def update_dataframe(self):
        if self.dataset is None:
            self._dataframe = None
        else:
            ds = prune_dataset_for_viz(self.dataset, drop_meta=True)
            score_multiplier = -1
            score_da = normalize(ds * score_multiplier).to_dataarray(name="score")
            norm_da = normalize(ds).to_dataarray(name="norm")
            value_da = ds.to_dataarray(name="value")

            self._dataframe = xr.merge([value_da, norm_da, score_da]).to_dataframe()

    @param.depends("_dataframe")
    def view(self):
        fig = px.line_polar(
            self._dataframe.reset_index(),
            color="solution",
            theta="variable",
            r="score",
            hover_data=["score", "value"],
            line_close=True,
        )
        fig.update_polars(
            radialaxis_showticklabels=True,
            radialaxis_title_text="score",
        )

        return pn.pane.Plotly(fig)


class VisualizationManager(param.Parameterized):
    dataset = param.Parameter()
    post_processor = param.Parameter(default=PostProcessor())
    scatter_plot = param.Parameter(default=InteractiveScatterPlot())
    parallel_coordinates_plot = param.Parameter(default=ParallelCoordinatesPlot())
    data_table = param.Parameter(default=DataTable())

    @param.depends("dataset", watch=True, on_init=True)
    def _update_dataset(self):
        self.post_processor.original_dataset = self.dataset

    @param.depends(
        "post_processor.processed_dataset",
        "post_processor.processed_dataframe",
        watch=True,
        on_init=True,
    )
    def _update_processed_data(self):
        self.scatter_plot.dataset = self.post_processor.processed_dataset
        self.parallel_coordinates_plot.dataset = self.post_processor.processed_dataset
        self.data_table.dataset = self.post_processor.processed_dataset

    @param.depends("dataset")
    def view(self):
        if self.dataset is None:
            return pn.Column(
                pn.VSpacer(),
                pn.pane.Markdown(
                    """
                    # No session loaded
                    Get started by running a problem or loading an existing session.
                    """,
                    styles={"text-align": "center"},
                    # To disable nifty features
                    renderer="markdown",
                ),
                pn.VSpacer(),
                sizing_mode="stretch_both",
                align="center",
            )

        return pn.Column(
            # FIXME: make the post-processor do something useful
            # self.post_processor.view,
            self.data_table.view,
            pn.Row(
                self.scatter_plot.view,
                self.parallel_coordinates_plot.view,
                height=500,
            ),
            sizing_mode="stretch_both",
        )


class MainState(param.Parameterized):
    settings = param.ClassSelector(class_=Settings)
    viz_manager = param.ClassSelector(class_=VisualizationManager)

    problem_path = param.String()
    session_path = param.String()

    run_problem = param.Event(label="Run problem")
    load_session = param.Event(label="Load session")

    working = param.Boolean(default=False)

    _ips = param.Parameter()
    dataset = param.Parameter()

    @property
    def ips(self):
        _ips = self._ips
        if _ips is None:
            _ips = IPSInstance(self.settings.ips_path)
            self._ips = _ips
        if not self._ips.connected:
            starting_toaster = pn.state.notifications.info(
                "Starting IPS...", duration=0
            )
            _ips.start()
            starting_toaster.destroy()

        return _ips

    @param.depends("dataset", watch=True, on_init=True)
    def _update_dataset(self):
        self.viz_manager.dataset = self.dataset

    def sidebar_view(self):
        problem_path_input = pn.widgets.TextInput.from_param(self.param.problem_path)
        run_btn = pn.widgets.Button.from_param(
            self.param.run_problem,
            button_type="primary",
        )
        session_path_input = pn.widgets.TextInput.from_param(self.param.session_path)
        load_btn = pn.widgets.Button.from_param(
            self.param.load_session,
            button_type="primary",
        )
        ips_path_input = pn.widgets.TextInput.from_param(self.settings.param.ips_path)

        def disable_when_working(working):
            problem_path_input.disabled = working
            session_path_input.disabled = working
            ips_path_input.disabled = working
            run_btn.disabled = working
            load_btn.disabled = working

        self.param.watch_values(disable_when_working, "working")

        run_problem_panes = (
            section_header("Run problem"),
            problem_path_input,
            run_btn,
        )

        load_session_panes = (
            section_header("Load session"),
            session_path_input,
            load_btn,
        )

        settings_panes = pn.WidgetBox(
            section_header("Settings"),
            ips_path_input,
        )

        layout = pn.Column(
            *run_problem_panes,
            "",
            *load_session_panes,
            "",
            *settings_panes,
            pn.VSpacer(),
            f"Autopack v{__version__}",
        )
        return layout

    def data_view(self):
        panel = pn.panel(self.viz_manager.view)

        def disable_when_working(working):
            panel.loading = working

        self.param.watch_values(disable_when_working, "working")

        return panel

    @param.depends("run_problem", "load_session", watch=True)
    def _update_working(self):
        self.working = True

    @param.depends("run_problem", watch=True)
    def _run_problem(self):
        try:
            self.working = True

            problem_path = pathlib.Path(self.problem_path)
            pn.state.notifications.info(
                f"Running problem from {problem_path}... See the console for progress details."
            )
            session_name = make_session_name(problem_path)
            session_path = SESSIONS_DIR / session_name
            session_path.mkdir(parents=True, exist_ok=False)

            self.settings.last_problem_path = problem_path

            shutil.copy(problem_path, session_path / "setup.json")

            harness_setup = HarnessSetup.from_json_file(problem_path)
            problem_setup = create_default_prob_setup(
                ips_instance=self.ips,
                harness_setup=harness_setup,
                create_imma=True,
            )

            dataset = global_optimize_harness(
                ips_instance=self.ips,
                problem_setup=problem_setup,
                init_samples=2,
                batches=2,
                batch_size=2,
            )

            save_session(dataset=dataset, ips=self.ips, session_dir=session_path)
            logger.notice(f"Saved session to {session_path}.")
            self.session_path = str(session_path)
            self.settings.last_session_path = str(session_path)

            self.dataset = dataset
        finally:
            self.working = False

    @param.depends("load_session", watch=True)
    def _load_session(self):
        try:
            self.working = True
            session_path = pathlib.Path(self.session_path)
            pn.state.notifications.info(f"Loading session from {session_path}...")
            dataset = load_session(ips=self.ips, session_dir=session_path)

            self.dataset = dataset

            self.settings.last_session_path = str(session_path)
        finally:
            self.working = False


def make_gui(**main_state_kwargs):
    init_panel()

    settings = Settings.load_or_new()
    viz_manager = VisualizationManager()
    main_state = MainState(
        settings=settings,
        viz_manager=viz_manager,
        problem_path=settings.last_problem_path,
        session_path=settings.last_session_path,
        **main_state_kwargs,
    )

    template = pn.template.MaterialTemplate(
        title="Autopack",
        header_background="#b31f2f",  # IPS red
        sidebar_width=400,
        sidebar=[
            main_state.sidebar_view(),
        ],
        main=[
            main_state.data_view(),
        ],
    )

    return template
