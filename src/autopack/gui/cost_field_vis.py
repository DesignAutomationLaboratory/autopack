import pyvista as pv
def no_update(value):
    pass
def cost_field_vis(setup, ips_instance):
    p = pv.Plotter
    for cost_field in setup.cost_fields:
        slider = p.add_slider_widget(
            callback=no_update,
            rng = [5, 100],
            title="Resolution",
            title_opacity=0.5,
            title_color="red",
            fmt="%0.9f",
            title_height=0.08,
        )
    p.show()
