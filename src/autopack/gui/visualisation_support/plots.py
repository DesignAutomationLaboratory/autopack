import pandas as pd
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
import matplotlib
matplotlib.use('agg')
#pn.extension('matplotlib')
import matplotlib.pyplot as plt
from math import pi
import numpy as np

def normalize(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols] / df[numeric_cols].max()
    return df

def create_interactive_scatter(df, width=1200, height=600, select_on_top=True):
    # Add an index column to the DataFrame
    df['index'] = df.index
    
    source = ColumnDataSource(data={'x': df[df.columns[0]], 'y': df[df.columns[1]]})

    # Create a blank figure
    p = figure(title='Scatter Plot', width=width, height=height)

    # Add a scatter plot to the figure
    scatter = p.scatter(x='x', y='y', source=source)

    # Create a hover tool
    tooltips = [("index", "@index"), ("x value", "@{x}"), ("y value", "@{y}")]
    hover_tool = HoverTool(tooltips=tooltips)
    p.add_tools(hover_tool)

    # Function to update the scatter plot
    def update_x(event):
        source.data = {'x': df[event.new], 'y': source.data['y']}
        p.xaxis.axis_label = event.new

    def update_y(event):
        source.data = {'x': source.data['x'], 'y': df[event.new]}
        p.yaxis.axis_label = event.new

    # Create widgets to select columns for the x and y axes
    x_select = pn.widgets.Select(name='x_col', options=list(df.columns))
    y_select = pn.widgets.Select(name='y_col', options=list(df.columns))

    x_select.param.watch(update_x, 'value')
    y_select.param.watch(update_y, 'value')

    # Create a row layout for the widgets and plot
    if select_on_top:
        layout = pn.Column(pn.Row(x_select, y_select), p)
    else:
        layout = pn.Row(pn.Column(x_select, y_select), p)

    return layout
    
def create_radar_chart(df, normalize_df=True):
    if normalize_df:
        df = normalize(df)
    # Number of variables
    categories = list(df)
    N = len(categories)

    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(df.index))

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Compute angle of each axis in the plot
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(color="grey", size=7)
    plt.ylim(0, 1)

    # Plot each individual
    for i, idx in enumerate(df.index):
        values = df.loc[idx].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=my_palette(i), linewidth=2, linestyle='solid', label=str(idx))
        ax.fill(angles, values, color=my_palette(i), alpha=0.4)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.close(fig)
    
    return pn.pane.Matplotlib(fig, tight=True), fig
