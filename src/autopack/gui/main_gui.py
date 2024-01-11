import datetime
import pathlib
import shutil
import tkinter as tk
from tkinter import filedialog

import holoviews as hv
import hvplot.xarray  # noqa: F401
import panel as pn
import param

from autopack import USER_DIR, __version__
from autopack.data_model import HarnessSetup
from autopack.default_commands import create_default_prob_setup
from autopack.harness_optimization import global_optimize_harness
from autopack.io import load_dataset, save_dataset
from autopack.ips_communication.ips_class import IPSInstance
from autopack.ips_communication.ips_commands import load_scene, save_scene

SETTINGS_PATH = USER_DIR / "gui-settings.json"
SESSIONS_DIR = USER_DIR / "sessions"


def make_session_name(problem_path: pathlib.Path):
    problem_name = problem_path.stem
    timestamp = datetime.datetime.utcnow()
    timestamp_str = timestamp.strftime("%Y-%m-%dT%H%M%S")
    return f"{problem_name}.{timestamp_str}.v{__version__}"


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


def create_settings_view(settings):
    return pn.Param(
        settings,
        parameters=["ips_path"],
        widgets={"ips_path": {"type": pn.widgets.TextInput}},
    )


class PostProcessor(param.Parameterized):
    original_dataset = param.Parameter()
    processed_dataset = param.Parameter()
    processed_dataframe = param.Parameter()

    _update = param.Event()

    @param.depends("_update", "original_dataset", watch=True, on_init=True)
    def update(self):
        ds = self.original_dataset

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
        return pn.widgets.StaticText(value="Hello World")


class DataTable(param.Parameterized):
    dataframe = param.Parameter()

    @param.depends("dataframe")
    def view(self):
        if self.dataframe is None:
            return pn.widgets.StaticText(value="No dataset loaded")
        # FIXME: stack/pivot the dataframe properly, so that the cost
        # fields show up as hierarchical columns instead
        return pn.WidgetBox(
            pn.widgets.Tabulator(
                self.dataframe,
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
        return self.dataset.hvplot.scatter(
            x=self.x,
            y=self.y,
            c=self.color,
            by=self.by,
            groupby=[],
            colormap="viridis",
            colorbar=True,
        ).opts(title="")

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


class VisualizationManager(param.Parameterized):
    dataset = param.Parameter()
    post_processor = param.Parameter(default=PostProcessor())
    scatter_plot = param.Parameter(default=InteractiveScatterPlot())
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
        self.data_table.dataframe = self.post_processor.processed_dataframe

    @param.depends("dataset")
    def view(self):
        if self.dataset is None:
            return pn.widgets.StaticText(value="No dataset loaded")
        return pn.Column(
            self.post_processor.view,
            self.scatter_plot.view,
            self.data_table.view,
        )


class MainState(param.Parameterized):
    settings = param.ClassSelector(class_=Settings)
    viz_manager = param.ClassSelector(class_=VisualizationManager)

    problem_path = param.Path()
    session_path = param.Foldername()

    run_problem_evt = param.Event(label="Run problem")
    load_session_evt = param.Event(label="Load session")

    _ips = param.Parameter()
    dataset = param.Parameter()

    @property
    def ips(self):
        _ips = self._ips
        if _ips is None:
            _ips = IPSInstance(self.settings.ips_path)
            self._ips = _ips
        if not self._ips.connected:
            _ips.start()

        return _ips

    @param.depends("dataset", watch=True, on_init=True)
    def _update_dataset(self):
        self.viz_manager.dataset = self.dataset

    def session_manager_view(self):
        problem_path_input = pn.widgets.TextInput.from_param(self.param.problem_path)
        run_btn = pn.widgets.Button.from_param(self.param.run_problem_evt)

        session_path_input = pn.widgets.TextInput.from_param(self.param.session_path)
        load_btn = pn.widgets.Button.from_param(self.param.load_session_evt)

        layout = pn.Column(
            problem_path_input,
            run_btn,
            session_path_input,
            load_btn,
        )

        return layout

    @param.depends("run_problem_evt", watch=True)
    def run_problem(self):
        problem_path = pathlib.Path(self.problem_path)
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

        # Save the scene so we can inspect solutions later
        save_scene(self.ips, session_path / "scene.ips")
        save_dataset(dataset, session_path / "dataset.pkl")
        session_marker = session_path / ".autopack-session"
        session_marker.touch()

        self.session_path = session_path
        self.settings.last_session_path = session_path

        self.dataset = dataset

    @param.depends("load_session_evt", watch=True)
    def load_session(self):
        session_path = pathlib.Path(self.session_path)
        session_marker = session_path / ".autopack-session"
        if not session_marker.exists():
            raise ValueError(
                f"{self.session_path} is not a complete Autopack session directory"
            )

        self.dataset = load_dataset(session_path / "dataset.pkl")
        load_scene(self.ips, session_path / "scene.ips", clear=True)

        self.settings.last_session_path = session_path


def make_gui(**main_state_kwargs):
    pn.extension("tabulator")
    hv.extension("bokeh")

    settings = Settings.load_or_new()
    viz_manager = VisualizationManager()
    main_state = MainState(
        settings=settings,
        viz_manager=viz_manager,
        problem_path=settings.last_problem_path,
        session_path=settings.last_session_path,
        **main_state_kwargs,
    )

    return pn.template.BootstrapTemplate(
        title="Autopack",
        header_background="#b31f2f",  # IPS red
        sidebar=[
            main_state.session_manager_view(),
            create_settings_view(settings),
        ],
        main=[viz_manager.view],
    )
