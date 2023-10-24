import pandas as pd
import panel as pn

def create_row_select(df):
    # Create a SelectMultiple widget with options being the DataFrame's index
    select_rows = pn.widgets.MultiSelect(name='Select designs', options=list(df.index))
    return select_rows


def create_column_select(df):
    # Create a MultiSelect widget with options being the DataFrame's columns
    multiselect = pn.widgets.MultiSelect(name='Select columns', options=list(df.columns))
    return multiselect