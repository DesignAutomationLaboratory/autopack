import pandas as pd
import panel as pn
import param

from autopack.data_model import HarnessSetup
from autopack.default_commands import create_default_prob_setup
from autopack.gui.select_path import (
    dump_json,
    load_json,
    select_file_path,
    select_save_file_path,
)
from autopack.ips_communication.ips_class import IPSInstance
from autopack.optimization import global_optimize_harness

max_width_left_column = "200px"  # or whatever width you desire


class GuiSetup(param.Parameterized):
    ips_path = param.Parameter("", doc="Path to the IPS.exe")
    harness_path = param.Parameter(
        "", doc="Path to the json file describing the harness optimization setup"
    )
    run_imma = param.Parameter(
        False, doc="Boolean describing if an Imma analyse will be run or not"
    )
    problem_setup = None
    result = None


gui_setup = GuiSetup()


def create_ips_instance():
    ips = IPSInstance(gui_setup.ips_path)
    ips.start()
    return ips


def save_to_file(event):
    """
    Saves the GuiSetup instance to a file.
    """
    filename = select_save_file_path()
    print(gui_setup.ips_path)
    dump_json(gui_setup, filename)


def load_from_file(event):
    """
    Loads the GuiSetup instance from a file.
    """
    file_path = select_file_path()
    new_setup = load_json(file_path)
    gui_setup.ips_path = new_setup.ips_path
    gui_setup.harness_path = new_setup.harness_path
    gui_setup.run_imma = new_setup.run_imma
    gui_setup.cost_fields = new_setup.cost_fields


def click_load(event):
    file_path = select_file_path()
    gui_setup.ips_path = file_path


def click_select_IPS(event):
    file_path = select_file_path()
    gui_setup.ips_path = file_path


def click_select_optimization_setup(event):
    file_path = select_file_path()
    gui_setup.harness_path = file_path


def click_run_optimization(event):
    with open(gui_setup.harness_path, "r") as f:
        user_json_str = f.read()
    harness_setup = HarnessSetup.model_validate_json(user_json_str)
    ips_instance = create_ips_instance()
    prob_setup = create_default_prob_setup(
        ips_instance, harness_setup, create_imma=gui_setup.run_imma
    )
    gui_setup.problem_setup = prob_setup
    results = global_optimize_harness(
        ips_instance, prob_setup, init_samples=8, batches=4, batch_size=4
    )
    gui_setup.result = results
    # data = ["Item 1", "Item 2", "Item 3", "Item 4"]
    # gui_setup.cost_fields = "\n".join(f"- {item}" for item in data)


button_save = pn.widgets.Button(name="Save")
button_save.on_click(save_to_file)
button_load = pn.widgets.Button(name="Load")
button_load.on_click(load_from_file)
save_column = pn.Column(button_save, button_load, styles=dict(background="#FF0000"))
top_text = pn.widgets.StaticText(value="Autopack")
gspec_top = pn.GridSpec(sizing_mode="stretch_both")
gspec_top[0, :6] = pn.Row(top_text, styles=dict(background="#FF0000"))
gspec_top[0, 6:] = save_column

button_select_ips_path = pn.widgets.Button(name="Select IPS run path")
button_select_ips_path.on_click(click_select_IPS)
ips_path = pn.widgets.StaticText(
    name="IPS run path",
    value=gui_setup.param.ips_path,
    style={"max-width": max_width_left_column},
)

button_select_optimization_setup = pn.widgets.Button(name="Select harness setup")
button_select_optimization_setup.on_click(click_select_optimization_setup)
harness_path = pn.widgets.StaticText(
    name="Harness setup",
    value=gui_setup.param.harness_path,
    style={"max-width": max_width_left_column},
)

imma_checkbox = pn.widgets.Checkbox(
    name="Perform IMMA analyse", value=gui_setup.param.run_imma
)


button_run_optimization = pn.widgets.Button(name="Run optimization")
button_run_optimization.on_click(click_run_optimization)


setup_column = pn.Column(
    button_select_ips_path,
    ips_path,
    button_select_optimization_setup,
    harness_path,
    imma_checkbox,
    button_run_optimization,
    styles=dict(background="#ff9999"),
)


gspec = pn.GridSpec(sizing_mode="stretch_both")
gspec[0, :6] = gspec_top
gspec[1:9, 0] = setup_column
app = gspec

app.servable()

# To run use python -m panel serve  --autoreload --show gui/main_gui.py

# conda run -n autopack python -m bokeh serve --show app.py
#  python -m panel serve  --autoreload --show app.py #
