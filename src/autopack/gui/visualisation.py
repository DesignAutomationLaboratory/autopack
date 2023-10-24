import panel as pn
import pandas as pd
import autopack.gui.visualisation_support.plots as plots
import autopack.gui.visualisation_support.elements as elements
import autopack.gui.visualisation_support.design_selection as design_selection


def generate_scatter(df2):
    d = {'col1': [0], 'col2': [0]}
    df1 = pd.DataFrame(data=d)
    #df2 = variables_used.my_setup.optimization_results

    if len(df2.columns) < 1:
        df = df1
    else:
        df = df2

    ####        Scatter plot        ####
    scatter = plots.create_interactive_scatter(df, select_on_top=True)
    table = pn.widgets.DataFrame(df)
    scatter_row = pn.Row(
        scatter,
        table
    )
    scatter_card = pn.Card(scatter_row, title='Scatter', styles={'background': 'WhiteSmoke'})
    ####        ------------        ####
    return scatter_card
scatter_card = generate_scatter()

def generate_radar(df2):
    d = {'col1': [0], 'col2': [0]}
    df1 = pd.DataFrame(data=d)
    #df2 = variables_used.my_setup.optimization_results

    if len(df2.columns) < 1:
        df = df1
    else:
        df = df2
    ####        Radar chart         ####
    radar_plot, fig = plots.create_radar_chart(df)
    row_select = elements.create_row_select(df)
    column_select = elements.create_column_select(df)

    def filter_dataframe(df, rows, cols):
        return df.loc[rows, cols]

    radar_button = pn.widgets.Button(name='Update radar', button_type='primary')
    def b(event):
        new_df = filter_dataframe(df, row_select.value, column_select.value)
        new_plot, fig = plots.create_radar_chart(new_df)
        radar_plot.object = fig
    radar_button.on_click(b)

    kmean_int_input = pn.widgets.IntInput(name='Nmb of designs', value=1, step=1, start=1, end=len(df.index))
    kmean_button = pn.widgets.Button(name='Automatic select', button_type='primary')
    def kmean_filter(event):
        selected = design_selection.select_KMeans(df, kmean_int_input.value)
        row_select.value = selected
    kmean_button.on_click(kmean_filter)

    k_mean_row = pn.Row(kmean_int_input, kmean_button)
    radar = pn.Column(
        pn.Row(
            radar_plot,
            pn.Column(
                k_mean_row,
                row_select,
                column_select,
                radar_button
            )
        )
    )
    radar_card = pn.Card(radar, title='Radar', styles={'background': 'WhiteSmoke'}, collapsed=True)
    ####        ------------        ####
    return radar_card
radar_card = generate_radar()


update_optimization_btn = pn.widgets.Button(name='Update optimazation', button_type='primary')
def update_optimization(event):
    scatter_card2 = generate_scatter()
    scatter_card.objects = scatter_card2.objects
    radar_card2 = generate_radar()
    radar_card.objects = radar_card2.objects
update_optimization_btn.on_click(update_optimization)

tab4 = pn.Column(
    update_optimization_btn,
    scatter_card,
    radar_card
)
