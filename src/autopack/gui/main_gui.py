import math
import os

import pandas as pd
import panel as pn
import param

import autopack.gui.visualisation_support.design_selection as design_selection
import autopack.gui.visualisation_support.elements as elements
import autopack.gui.visualisation_support.plots as plots
from autopack.data_model import HarnessSetup
from autopack.default_commands import create_default_prob_setup
from autopack.gui.select_path import (
    dump_json,
    load_json,
    select_file_path,
    select_save_file_path,
)
from autopack.harness_optimization import combine_cost_fields, global_optimize_harness
from autopack.ips_communication.ips_class import IPSInstance
from autopack.ips_communication.ips_commands import cost_field_vis

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
    pandas_result = pd.DataFrame([])
    ips_instance = None


def update_pandas_result(result_xarray):
    # Convert to pandas DataFrame
    ds = result_xarray
    df2 = ds["cost_field_weight"].to_dataframe().unstack(level="cost_field")
    df2.columns = [f"cost_field_weight_{col[1]}" for col in df2.columns]

    df3 = ds["bundling_factor"].to_dataframe().reset_index()

    df4 = ds["bundling_cost"].to_dataframe().unstack(level="cost_field")
    df4.columns = [f"bundling_cost_{col[1]}" for col in df4.columns]
    df4 = df4.reset_index(level="ips_solution", drop=True)

    df5 = ds["total_cost"].to_dataframe().unstack(level="cost_field")
    df5.columns = [f"total_cost_{col[1]}" for col in df5.columns]
    df5 = df5.reset_index(level="ips_solution", drop=True)

    df6 = (
        ds["num_estimated_clips"]
        .to_dataframe()
        .reset_index(level="ips_solution", drop=True)
    )

    result = df2.merge(df3, on="case", how="left")
    result = result.merge(df4, on="case", how="left")
    result = result.merge(df5, on="case", how="left")
    result = result.merge(df6, on="case", how="left")
    return result


gui_setup = GuiSetup()


def create_ips_instance():
    folder_path = os.path.dirname(gui_setup.ips_path)
    ips = IPSInstance(folder_path)
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
    computational_budget = numb_of_designs_to_run.value
    optimization_status.value = "Creating cost fields..."
    with open(gui_setup.harness_path, "r") as f:
        user_json_str = f.read()
    harness_setup = HarnessSetup.model_validate_json(user_json_str)
    ips_instance = create_ips_instance()
    prob_setup = create_default_prob_setup(
        ips_instance, harness_setup, create_imma=gui_setup.run_imma
    )
    gui_setup.problem_setup = prob_setup
    optimization_status.value = "Performing global optimization..."
    init_samples = max(2, int(computational_budget * 0.2))
    batches = max(1, int(math.sqrt(computational_budget * 0.8)))
    batch_size = max(2, int(math.sqrt(computational_budget * 0.8)))
    results = global_optimize_harness(
        ips_instance,
        prob_setup,
        init_samples=init_samples,
        batches=batches,
        batch_size=batch_size,
    )
    gui_setup.ips_instance = ips_instance
    gui_setup.result = results
    optimization_status.value = "Global optimization finished"
    gui_setup.pandas_result = update_pandas_result(results)
    new_scatter_row, new_table_row, new_radar_row = update_plots()
    scatter_row.objects = new_scatter_row.objects
    table_row.objects = new_table_row.objects
    radar_row.objects = new_radar_row.objects
    new_cost_field_vis_row = vis_cost_field_row(gui_setup.result)
    cost_field_vis_row.objects = new_cost_field_vis_row.objects
    # data = ["Item 1", "Item 2", "Item 3", "Item 4"]
    # gui_setup.cost_fields = "\n".join(f"- {item}" for item in data)


button_save = pn.widgets.Button(name="Save")
button_save.on_click(save_to_file)
button_load = pn.widgets.Button(name="Load")
button_load.on_click(load_from_file)
save_column = pn.Column(button_save, button_load, styles=dict(background="#FF0000"))
# top_text = pn.widgets.StaticText(value="Autopack")
top_text = pn.pane.HTML(
    '<div style="font-size: 60px; color: white;">Autopack</div>', height=110
)
gspec_top = pn.GridSpec(sizing_mode="stretch_both")
gspec_top[0, :6] = pn.Row(top_text, styles=dict(background="#FF0000"))
gspec_top[0, 6:] = save_column

button_select_ips_path = pn.widgets.Button(name="Select IPS run path")
button_select_ips_path.on_click(click_select_IPS)
ips_path = pn.widgets.StaticText(
    name="IPS run path",
    value=gui_setup.param.ips_path,
    styles={"max-width": max_width_left_column},
)

button_select_optimization_setup = pn.widgets.Button(name="Select harness setup")
button_select_optimization_setup.on_click(click_select_optimization_setup)
harness_path = pn.widgets.StaticText(
    name="Harness setup",
    value=gui_setup.param.harness_path,
    styles={"max-width": max_width_left_column},
)
imma_checkbox = pn.widgets.Checkbox(
    name="Perform IMMA analyse", value=gui_setup.param.run_imma
)


button_run_optimization = pn.widgets.Button(name="Run optimization")
button_run_optimization.on_click(click_run_optimization)
numb_of_designs_to_run = pn.widgets.IntInput(
    name="Number of designs wanted", value=5, step=1, start=0, end=1000
)

optimization_status = pn.widgets.StaticText(value="")

setup_column = pn.Column(
    button_select_ips_path,
    ips_path,
    button_select_optimization_setup,
    harness_path,
    imma_checkbox,
    numb_of_designs_to_run,
    button_run_optimization,
    optimization_status,
    styles=dict(background="#ff9999"),
)


def generate_scatter(df):
    scatter = plots.create_interactive_scatter(df, select_on_top=True)
    scatter_row = pn.Row(scatter)
    return scatter_row


def generate_radar(df):
    filtered_columns = df[
        [
            col
            for col in df.columns
            if col.startswith("bundling_cost")
            or col.startswith("total_")
            or col.startswith("num_estim")
        ]
    ]
    df = filtered_columns
    ####        Radar chart         ####
    radar_plot, fig = plots.create_radar_chart(df)
    row_select = elements.create_row_select(df)
    column_select = elements.create_column_select(df)

    def filter_dataframe(df, rows, cols):
        return df.loc[rows, cols]

    radar_button = pn.widgets.Button(name="Update radar", button_type="primary")

    def b(event):
        new_df = filter_dataframe(df, row_select.value, column_select.value)
        new_plot, fig = plots.create_radar_chart(new_df)
        radar_plot.object = fig

    radar_button.on_click(b)

    kmean_int_input = pn.widgets.IntInput(
        name="Nmb of designs", value=1, step=1, start=1, end=len(df.index)
    )
    kmean_button = pn.widgets.Button(name="Automatic select", button_type="primary")

    def kmean_filter(event):
        selected = design_selection.select_KMeans(df, kmean_int_input.value)
        row_select.value = selected

    kmean_button.on_click(kmean_filter)

    k_mean_row = pn.Row(kmean_int_input, kmean_button)
    radar_row = pn.Row(
        radar_plot, pn.Column(k_mean_row, row_select, column_select, radar_button)
    )
    return radar_row


def vis_cost_field(event):
    weights = []
    for widget in cost_field_vis_row:
        weights.append(widget.value)
    cost_fields = gui_setup.problem_setup.cost_fields
    cost_field = combine_cost_fields(cost_fields, weights, normalize_fields=True)
    cost_field_vis(gui_setup.ips_instance, cost_field)


button_vis_cost_field = pn.widgets.Button(name="Visulise cost field")
button_vis_cost_field.on_click(vis_cost_field)


def vis_cost_field_row(xarray):
    cost_field_vis_row = pn.Row()
    if xarray is not None:
        cost_field_array = xarray["cost_field"].values
        cost_field_list = cost_field_array.flatten().tolist()
        for field in cost_field_list:
            cost_field_vis_row.append(
                pn.widgets.FloatInput(name=field, value=0.5, step=1e-2, start=0, end=1)
            )
    return cost_field_vis_row
    # numb_of_designs_to_run = pn.widgets.IntInput(name='Number of designs wanted', value=5, step=1, start=0, end=1000)


def update_plots():
    df2 = gui_setup.pandas_result
    d = {"col1": [0], "col2": [0]}
    df1 = pd.DataFrame(data=d)
    # df2 = variables_used.my_setup.optimization_results

    if len(df2.columns) < 1:
        df = df1
    else:
        df = df2
    scatter_row = generate_scatter(df)
    table_row = pn.Row(pn.widgets.DataFrame(df))
    radar_row = generate_radar(df)
    return scatter_row, table_row, radar_row


scatter_row, table_row, radar_row = update_plots()
cost_field_vis_row = vis_cost_field_row(gui_setup.result)
plot_area = pn.Column(
    scatter_row, table_row, radar_row, cost_field_vis_row, button_vis_cost_field
)


gspec = pn.GridSpec(sizing_mode="stretch_both")
gspec[0, :6] = gspec_top
gspec[1:12, 0] = setup_column
gspec[1:12, 1:] = plot_area
app = gspec

app.servable()

# To run use python -m panel serve  --autoreload --show gui/main_gui.py

# conda run -n autopack python -m bokeh serve --show app.py
#  python -m panel serve  --autoreload --show app.py #
