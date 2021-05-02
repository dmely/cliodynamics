#!/usr/bin/env python

from collections import defaultdict
from collections import deque
import logging
import random
import time

from bokeh.document import Document
from bokeh.layouts import row, layout
from bokeh.plotting import figure, show
from bokeh.palettes import Category20b, Magma
from bokeh.transform import factor_cmap
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider, RangeTool
from bokeh.models import LinearColorMapper
from bokeh.models import LogColorMapper
from bokeh.models import ColorBar
from bokeh.models.widgets import Panel, Tabs
from bokeh.server.server import Server
import colorcet as cc
import numpy as np
import pandas as pd
from tqdm import tqdm

from cliodynamics.models.frontier import MetaethnicFrontierModel


logging.basicConfig(level=logging.INFO)


def app(doc: Document, max_history=500, max_generations=500, size=50):

    ## Simulate system
    history_membership = deque(maxlen=max_history)
    history_asabiya = deque(maxlen=max_history)
    history_power = deque(maxlen=max_history)
    history_areas = defaultdict(list)
    history_dates_per_empire = defaultdict(list)
    history_dates = []

    model = MetaethnicFrontierModel.empty_model(size)
    
    history_membership.append(model.membership.copy())
    history_asabiya.append(model.asabiya.copy())
    history_power.append(np.zeros_like(model.asabiya))

    for t in tqdm(range(max_generations)):
        powers = model.step()
        areas = model.get_areas()

        history_membership.append(model.membership.copy())
        history_asabiya.append(model.asabiya.copy())
        history_power.append(powers)
        history_dates.append(t)

        for empire, area in areas.items():
            history_areas[empire].append(area)
            history_dates_per_empire[empire].append(t)

    ## View atlas

    # Plot simulations results
    source = ColumnDataSource(data={
        "membership": [history_membership[0]],
        "asabiya": [history_asabiya[0]],
        "power": [history_power[0]],
    })

    # Categorical color palette with good visibility
    empires_palette = cc.b_glasbey_bw_minc_20_maxl_70
    empires_palette.insert(0, "#c9c9c9")

    p1 = figure(plot_width=300, plot_height=300, title="World atlas")
    p1.image(
        image="membership",
        source=source,
        x=0,
        y=0,
        dw=10,
        dh=10,
        level="image",
        color_mapper=LinearColorMapper(
            palette=empires_palette,
            low=0.0,
            high=len(empires_palette) - 1,
        ),
    )
    p1.grid.grid_line_width = 0.5

    p2 = figure(plot_width=300, plot_height=300, title="Asabiya")
    p2_color_mapper = LinearColorMapper(palette=Magma[256], low=0.0, high=1)
    p2.image(
        image="asabiya",
        source=source,
        x=0,
        y=0,
        dw=10,
        dh=10,
        level="image",
        color_mapper=p2_color_mapper,
    )
    p2.grid.grid_line_width = 0.5
    p2_color_bar = ColorBar(color_mapper=p2_color_mapper, label_standoff=12)
    p2.add_layout(p2_color_bar, "below")

    p3 = figure(plot_width=300, plot_height=300, title="Power")
    p3_color_mapper = LinearColorMapper(palette=Magma[256])
    p3.image(
        image="power",
        source=source,
        x=0,
        y=0,
        dw=10,
        dh=10,
        level="image",
        color_mapper=p3_color_mapper,
    )
    p3.grid.grid_line_width = 0.5
    p3_color_bar = ColorBar(color_mapper=p3_color_mapper, label_standoff=12)
    p3.add_layout(p3_color_bar, "below")

    slider = Slider(
        start=0,
        end=max_history - 1,
        value=0,
        step=1,
        title="Generation",
    )

    def update(attr, old, new):
        source.data = {
            "membership": [history_membership[slider.value]],
            "asabiya": [history_asabiya[slider.value]],
            "power": [history_power[slider.value]],
        }

    slider.on_change("value", update)

    atlas_view = layout([[p1, p2, p3], slider])
    tab1 = Panel(child=atlas_view, title="Atlas view")

    ## Compute and view time series
    show_date_interval_size = len(history_dates) // 5
    timeseries_view = figure(
        plot_width=750,
        plot_height=350,
        x_range=(history_dates[0], history_dates[show_date_interval_size]),
    )
    thumbnail_view = figure(
        plot_width=750,
        plot_height=150,
        y_range=timeseries_view.y_range,
        x_axis_type="datetime",
        y_axis_type=None,
        tools="",
        toolbar_location=None,
        background_fill_color="#efefef",
    )

    for empire_id, area_timeseries in history_areas.items():
        glyph_kwargs = dict(
            x=history_dates_per_empire[empire_id],
            y=area_timeseries,
            color=empires_palette[empire_id],
            line_width=2.0,
        )
        timeseries_view.line(
            **glyph_kwargs, legend_label=str(empire_id), alpha=0.8)
        thumbnail_view.line(**glyph_kwargs)

    range_tool = RangeTool(x_range=timeseries_view.x_range)
    range_tool.overlay.fill_color = "gray"
    range_tool.overlay.fill_alpha = 0.2

    thumbnail_view.ygrid.grid_line_color = None
    thumbnail_view.add_tools(range_tool)
    thumbnail_view.toolbar.active_multi = range_tool

    tab2 = Panel(
        child=column(timeseries_view, thumbnail_view),
        title="Empire areas",
    )

    ## Display application
    doc.add_root(Tabs(tabs=[tab1, tab2]))
    doc.title = "Metaethnic frontier theory"


# Setting num_procs here means we can't touch the IOLoop before now, we must
# let Server handle that. If you need to explicitly handle IOLoops then you
# will need to use the lower level BaseServer class.
server = Server({"/": app}, num_procs=1)
server.start()


if __name__ == '__main__':
    logging.info("Opening Bokeh application on http://localhost:5006/")
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
