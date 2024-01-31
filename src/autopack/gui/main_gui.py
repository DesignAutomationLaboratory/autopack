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

from .. import SESSIONS_DIR, USER_DIR, __version__, logger
from ..data_model import ErgoSettings, HarnessSetup, StudySettings
from ..io import load_session, save_session
from ..ips import IPSInstance
from ..postprocessing import (
    functions_of_interest,
    only_dims,
    only_dtypes,
    score_multipliers,
)
from ..utils import appr_num_solutions, normalize, partition_opt_budget
from ..workflows import build_problem_and_run_study

SETTINGS_PATH = USER_DIR / "gui-settings.json"

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
    title="Select a file",
    initial_dir=None,
    initial_file=None,
):
    window = tk.Tk()
    window.wm_attributes("-topmost", 1)
    window.withdraw()
    path = filedialog.askopenfilename(
        parent=window,
        initialdir=initial_dir,
        initialfile=initial_file,
        title=title,
        filetypes=[("JSON files", "*.json"), ("All files", "*")],
    )
    window.destroy()
    return path or None


def open_directory_dialog(
    title="Select a directory",
    initial_dir=None,
    must_exist=True,
):
    window = tk.Tk()
    window.wm_attributes("-topmost", 1)
    window.withdraw()
    path = filedialog.askdirectory(
        parent=window,
        initialdir=initial_dir,
        title=title,
        mustexist=must_exist,
    )
    window.destroy()
    return path or None


class Settings(param.Parameterized):
    ips_path = param.Foldername(label="IPS path", check_exists=False)
    # String parameters here to not do any unexpected smartness
    last_problem_path = param.String()
    last_session_path = param.String()

    browse_ips_path = param.Action(
        lambda x: x.param.trigger("browse_ips_path"), label="Browse IPS path"
    )

    @classmethod
    def load_or_new(cls):
        if SETTINGS_PATH.exists():
            try:
                return cls(
                    **cls.param.deserialize_parameters(SETTINGS_PATH.read_text())
                )
            except Exception as exc:
                logger.error(f"Failed to load GUI settings: {exc}")

        return cls()

    @param.depends("ips_path", "last_problem_path", "last_session_path", watch=True)
    def persist(self):
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        settings_to_persist = set(self.param.params().keys()) - {
            "name",
            "browse_ips_path",
        }
        SETTINGS_PATH.write_text(
            self.param.serialize_parameters(subset=settings_to_persist)
        )

    @param.depends("browse_ips_path", watch=True)
    def _browse_ips_path(self):
        path = open_directory_dialog(
            title="Select the IPS directory",
            initial_dir=self.ips_path or None,
        )
        if path:
            # param.Foldername doesn't like pathlib.Path (why???)
            self.ips_path = path


class RuntimeSettings(param.Parameterized):
    run_ergo = param.Boolean(default=True, label="Build ergo cost fields")
    ergo_use_rbpp = param.Boolean(
        default=ErgoSettings.model_fields["use_rbpp"].default,
        label=ErgoSettings.model_fields["use_rbpp"].title,
        doc=ErgoSettings.model_fields["use_rbpp"].description,
    )
    ergo_sample_ratio = param.Number(
        # We expose this as a parameter primarily to make it easier to
        # change in testing
        default=ErgoSettings.model_fields["sample_ratio"].default,
    )
    doe_samples = param.Integer(
        # We expose this as a parameter primarily to make it easier to
        # change in testing
        default=StudySettings.model_fields["doe_samples"].default,
    )
    opt_budget = param.Integer(
        default=64,
        bounds=(0, 512),
        step=32,
        label="Optimization budget",
        doc="Appr. number of routings to carry out under optimization. 0 to disable optimization.",
    )

    @param.depends("opt_budget")
    def total_num_solutions(self):
        batch_size, num_batches = partition_opt_budget(self.opt_budget)
        opt_evals = batch_size * num_batches
        static_evals = 10 + self.doe_samples
        return appr_num_solutions(int(static_evals + opt_evals))

    def to_ergo_settings(self):
        if not self.run_ergo:
            return None
        return ErgoSettings(
            use_rbpp=self.ergo_use_rbpp,
            sample_ratio=self.ergo_sample_ratio,
        )

    def to_study_settings(self):
        batch_size, num_batches = partition_opt_budget(self.opt_budget)
        return StudySettings(
            opt_batches=num_batches,
            opt_batch_size=batch_size,
            doe_samples=self.doe_samples,
        )

    def view_panes(self):
        run_ergo_input = pn.widgets.Checkbox.from_param(self.param.run_ergo)
        ergo_use_rbpp_input = pn.widgets.Checkbox.from_param(self.param.ergo_use_rbpp)
        opt_budget_input = pn.widgets.IntInput.from_param(self.param.opt_budget)
        appr_num_solutions_input = pn.widgets.StaticText(
            name="Total number of solutions (appr.)",
            value=self.total_num_solutions,
        )

        def disable_when_no_ergo(value):
            ergo_use_rbpp_input.disabled = not value

        run_ergo_input.param.watch_values(disable_when_no_ergo, "value")

        return (
            run_ergo_input,
            ergo_use_rbpp_input,
            opt_budget_input,
            appr_num_solutions_input,
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
            ds = self.dataset
            ds = only_dims(ds, ["solution"])
            ds = only_dtypes(ds, [float, int, str, bool])
            self._dataframe = ds.to_dataframe()

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
            vizable_ds = only_dtypes(self.dataset, [float, int])
            selectable_dims = list(vizable_ds.dims.keys())
            selectable_vars = list(vizable_ds.data_vars.keys())

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
                hover_cols=["solution"],
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
            foi_ds = functions_of_interest(self.dataset)
            scaler_ds = score_multipliers(foi_ds)
            score_da = normalize(foi_ds * scaler_ds).to_dataarray(name="score")
            norm_da = normalize(foi_ds).to_dataarray(name="norm")
            value_da = foi_ds.to_dataarray(name="value")

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


class AutopackApp(param.Parameterized):
    settings = param.ClassSelector(class_=Settings)
    viz_manager = param.ClassSelector(class_=VisualizationManager)

    runtime_settings = param.ClassSelector(class_=RuntimeSettings)

    problem_path = param.Filename(check_exists=False)
    session_path = param.Foldername(check_exists=False)

    browse_problem_path = param.Action(lambda x: x.param.trigger("browse_problem_path"))
    browse_session_path = param.Action(lambda x: x.param.trigger("browse_session_path"))

    run_problem = param.Event(label="Run problem")
    load_session = param.Event(label="Load session")

    working = param.Boolean(default=False)

    _ips = param.ClassSelector(class_=IPSInstance, allow_None=True)
    dataset = param.ClassSelector(class_=xr.Dataset, allow_None=True)

    template = param.ClassSelector(class_=pn.template.MaterialTemplate)

    def __init__(self, **params):
        params.setdefault("settings", Settings.load_or_new())
        params.setdefault("runtime_settings", RuntimeSettings())
        params.setdefault("viz_manager", VisualizationManager())
        params.setdefault("problem_path", params["settings"].last_problem_path)
        params.setdefault("session_path", params["settings"].last_session_path)
        params.setdefault("template", pn.template.MaterialTemplate())

        super().__init__(**params)

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

    @param.depends("browse_problem_path", watch=True)
    def _browse_problem_path(self):
        path = open_problem_path_dialog(
            title="Select an Autopack problem file",
            initial_dir=self.problem_path or None,
            initial_file=self.problem_path or None,
        )
        if path:
            self.problem_path = path

    @param.depends("browse_session_path", watch=True)
    def _browse_session_path(self):
        path = open_directory_dialog(
            title="Select a session directory",
            initial_dir=self.session_path or SESSIONS_DIR or None,
            must_exist=True,
        )
        if path:
            self.session_path = path

    @param.depends("dataset", watch=True, on_init=True)
    def _update_dataset(self):
        self.viz_manager.dataset = self.dataset

    def sidebar_view(self):
        problem_path_input = pn.widgets.TextInput.from_param(self.param.problem_path)
        browse_problem_path_btn = pn.widgets.Button.from_param(
            self.param.browse_problem_path, name="Browse..."
        )
        run_btn = pn.widgets.Button.from_param(
            self.param.run_problem,
            button_type="primary",
        )

        session_path_input = pn.widgets.TextInput.from_param(self.param.session_path)
        browse_session_path_btn = pn.widgets.Button.from_param(
            self.param.browse_session_path, name="Browse..."
        )
        load_btn = pn.widgets.Button.from_param(
            self.param.load_session,
            button_type="primary",
        )

        ips_path_input = pn.widgets.TextInput.from_param(self.settings.param.ips_path)
        browse_ips_path_btn = pn.widgets.Button.from_param(
            self.settings.param.browse_ips_path, name="Browse..."
        )

        def disable_when_working(working):
            problem_path_input.disabled = working
            browse_problem_path_btn.disabled = working
            run_btn.disabled = working
            session_path_input.disabled = working
            browse_session_path_btn.disabled = working
            load_btn.disabled = working
            ips_path_input.disabled = working
            browse_ips_path_btn.disabled = working
            run_btn.disabled = working
            load_btn.disabled = working

        self.param.watch_values(disable_when_working, "working")

        run_problem_panes = (
            section_header("Run problem"),
            problem_path_input,
            browse_problem_path_btn,
            *self.runtime_settings.view_panes(),
            run_btn,
        )

        load_session_panes = (
            section_header("Load session"),
            session_path_input,
            browse_session_path_btn,
            load_btn,
        )

        settings_panes = pn.WidgetBox(
            section_header("Settings"),
            ips_path_input,
            browse_ips_path_btn,
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

    @param.depends("problem_path", "session_path", watch=True)
    def _persist_paths(self):
        self.settings.last_problem_path = self.problem_path
        self.settings.last_session_path = self.session_path

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

            shutil.copy(problem_path, session_path / "setup.json")

            dataset = build_problem_and_run_study(
                ips=self.ips,
                harness_setup=HarnessSetup.from_json_file(problem_path),
                ergo_settings=self.runtime_settings.to_ergo_settings(),
                study_settings=self.runtime_settings.to_study_settings(),
            )

            save_session(dataset=dataset, ips=self.ips, session_dir=session_path)
            logger.notice(f"Saved session to {session_path}.")
            self.session_path = str(session_path)

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
        finally:
            self.working = False

    def view(self):
        self.template.title = "Autopack"
        self.template.header_background = "#b31f2f"  # IPS red
        self.template.sidebar_width = 400
        self.template.sidebar[:] = [
            self.sidebar_view(),
        ]
        self.template.main[:] = [
            self.data_view(),
        ]
        return self.template
