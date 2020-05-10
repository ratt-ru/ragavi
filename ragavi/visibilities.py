# -*- coding: utf-8 -*-
from __future__ import division

import sys

from collections import namedtuple, OrderedDict
from datetime import datetime
from itertools import combinations

import colorcet as cc
import dask.array as da
import daskms as xm
import datashader as ds
import numpy as np
import xarray as xr

from dask import compute, delayed, config as dask_config
from dask.diagnostics import ProgressBar
from daskms.table_schemas import MS_SCHEMA

from bokeh.plotting import figure
from bokeh.layouts import column, gridplot, row, grid
from bokeh.io import (output_file, output_notebook, show, save, curdoc)
from bokeh.models import (BasicTicker, ColorBar, ColumnDataSource,  CustomJS,
                          DatetimeTickFormatter, Div, Grid, ImageRGBA,
                          LinearAxis, LinearColorMapper,
                          PrintfTickFormatter, Plot, PreText,
                          Range1d, Text, Title, Toolbar)
from bokeh.models.tools import (BoxZoomTool, HoverTool, ResetTool, PanTool,
                                WheelZoomTool, SaveTool)


from ragavi.averaging import get_averaged_ms
from ragavi.plotting import create_bk_fig, add_axis
import ragavi.utils as vu

from dask.distributed import Client, LocalCluster

logger = vu.logger
excepthook = vu.sys.excepthook
time_wrapper = vu.time_wrapper
wrapper = vu.textwrap.TextWrapper(initial_indent='',
                                  break_long_words=True,
                                  subsequent_indent=''.rjust(50),
                                  width=160)

# some variables
PLOT_WIDTH = int(1920 * 0.95)
PLOT_HEIGHT = int(1080 * 0.85)


class DataCoreProcessor:
    """Process Measurement Set data into forms desirable for visualisation.

    Parameters
    ----------
    chan : :obj:`slice` or :obj:`int`
        Channels that will be selected. Defaults to all
    cbin : :obj:`int`
        Channel averaging bin size
    corr : :obj:`int` or :obj:`slice`
        Correlation indices to be selected. Defaults to all
    datacol : :obj:`str`
        Data column to be selected. Defaults to 'DATA'
    ddid : :obj:`slice`
        Spectral windows to be selected. Defaults to all
    flag : :obj:`bool`
        To flag or not. Defaults to True
    ms_name : :obj:`str`
        Name / path to the Measurement Set
    xds_table_obj : :obj:`xarray.Dataset`
        Table object from which data will be extracted
    xaxis : :obj:`str`
        x-axis to be selected for plotting
    yaxis : :obj:`str`
        y-axis to be selected for plotting
    """

    def __init__(self, xds_table_obj, ms_name, xaxis, yaxis,
                 chan=None, corr=None, cbin=None,
                 ddid=int, datacol="DATA", flag=True):

        self.xds_table_obj = xds_table_obj
        self.ms_name = ms_name
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.corr = corr
        self.ddid = ddid
        self.chan = chan
        self.datacol = datacol
        self.flag = flag
        self.cbin = cbin

    def get_xaxis_data(self):
        """Get x-axis data. This function also returns the relevant x-axis label.
        Returns
        -------
        xdata : :obj:`xarray.DataArray`
            X-axis data depending  x-axis selected.
        x_label : :obj:`str`
            Label to appear on the x-axis of the plots.
        """

        if self.xaxis == "antenna1":
            xdata = self.xds_table_obj.ANTENNA1
            x_label = "Antenna1"
        elif self.xaxis == "antenna2":
            xdata = self.xds_table_obj.ANTENNA2
            x_label = "Antenna2"
        elif self.xaxis == "amplitude":
            xdata = self.xds_table_obj[self.datacol]
            x_label = "Amplitude"
        elif self.xaxis == "frequency" or self.xaxis == "channel":
            xdata = vu.get_frequencies(self.ms_name, spwid=self.ddid,
                                       chan=self.chan, cbin=self.cbin)
            x_label = "Frequency GHz"
        elif self.xaxis == "imaginary":
            xdata = self.xds_table_obj[self.datacol]
            x_label = "Imaginary"
        elif self.xaxis == "phase":
            xdata = self.xds_table_obj[self.datacol]
            x_label = "Phase [deg]"
        elif self.xaxis == "real":
            xdata = self.xds_table_obj[self.datacol]
            x_label = "Real"
        elif self.xaxis == "scan":
            xdata = self.xds_table_obj.SCAN_NUMBER
            x_label = "Scan"
        elif self.xaxis == "time":
            xdata = self.xds_table_obj.TIME
            x_label = "Time [s]"
        elif self.xaxis == "uvdistance":
            xdata = self.xds_table_obj.UVW
            x_label = "UV Distance [m]"
        elif self.xaxis == "uvwave":
            xdata = self.xds_table_obj.UVW
            try:
                x_label = "UV Wave [K{}]".format(u"\u03bb")
            except UnicodeEncodeError:
                x_label = "UV Wave [K{}]".format(u"\u03bb".encode("utf-8"))
        else:
            logger.error("Invalid xaxis name")
            return

        return xdata, x_label

    def get_yaxis_data(self):
        """Extract the required column for the y-axis data.

        Returns
        -------
        ydata: :obj:`xarray.DataArray`
            y-axis data depending  y-axis selected.
        y_label: :obj:`str`
            Label to appear on the y-axis of the plots.
        """
        if self.yaxis == "amplitude":
            y_label = "Amplitude"
        elif self.yaxis == "imaginary":
            y_label = "Imaginary"
        elif self.yaxis == "phase":
            y_label = "Phase [deg]"
        elif self.yaxis == "real":
            y_label = "Real"

        try:
            ydata = self.xds_table_obj[self.datacol]
        except KeyError:
            logger.exception("Column '{}' not Found".format(self.datacol))
            return sys.exit(-1)

        return ydata, y_label

    def prep_xaxis_data(self, xdata, freq=None):
        """Prepare the x-axis data for plotting. Selections also performed here.

        Parameters
        ----------
        xdata: :obj:`xarray.DataArray`
            x-axis data depending  x-axis selected
        freq: :obj:`xarray.DataArray` or :obj:`float`
            Frequency(ies) from which corresponding wavelength will be obtained.
            Note
            ----
                Only required when xaxis specified is 'uvwave'

        Returns
        -------
        prepdx: :obj:`xarray.DataArray`
            Prepared :attr:`xdata`
        """
        if self.xaxis == "channel" or self.xaxis == "frequency":
            prepdx = xdata / 1e9
        elif self.xaxis in ["phase", "amplitude", "real", "imaginary"]:
            prepdx = xdata
        elif self.xaxis == "time":
            prepdx = vu.time_convert(xdata)
        elif self.xaxis == "uvdistance":
            prepdx = vu.calc_uvdist(xdata)
        elif self.xaxis == "uvwave":
            prepdx = vu.calc_uvwave(xdata, freq) / 1e3
        elif self.xaxis == "antenna1" or self.xaxis == "antenna2" or self.xaxis == "scan":
            prepdx = xdata
        return prepdx

    def prep_yaxis_data(self, ydata, yaxis=None):
        """Process data for the y-axis.
        This includes:

            * Flagging
            * Conversion form complex to the required form

        Data selection and flagging are done by this function itself

        Parameters
        ----------
        ydata: :obj:`xarray.DataArray`
            y-axis data to be processed

        Returns
        -------
        y: :obj:`xarray.DataArray`
           Processed :attr:`ydata` data.
        """
        flags = vu.get_flags(self.xds_table_obj)

        # Doing this because some of xaxis data must be processed here
        if yaxis is None:
            yaxis = self.yaxis

        # if flagging enabled return a list of DataArrays otherwise return a
        # single dataarray
        if self.flag:
            if np.all(flags.values == True):
                logger.warning(
                    "All data appears to be flagged. Unable to continue.")
                logger.warning(
                    "Please use -nf or --no-flagged to deactivate flagging if you still wish to generate this plot.")
                logger.warning(" Exiting.")
                sys.exit(0)
            processed = self.process_data(ydata, yaxis=yaxis, wrap=True)
            y = processed.where(flags == False)
        else:
            y = self.process_data(ydata, yaxis=yaxis)

        return y

    def process_data(self, ydata, wrap=True, yaxis=None):
        """Abstraction for processing y-data passes it to the processing function.

        Parameters
        ----------
        ydata: :obj:`xarray.DataArray`
            y-data to process
        yaxis: :obj:`str`
            Selected yaxis

        Returns
        -------
        y : :obj:`xarray.DataArray`
           Processed :obj:`ydata`
        """
        if yaxis is None:
            yaxis = self.yaxis
        if yaxis == "amplitude":
            y = vu.calc_amplitude(ydata)
        elif yaxis == "imaginary":
            y = vu.calc_imaginary(ydata)
        elif yaxis == "phase":
            y = vu.calc_phase(ydata, wrap=wrap)
        elif yaxis == "real":
            y = vu.calc_real(ydata)
        return y

    def blackbox(self):
        """Get raw input data and churn out processed x and y data.

        This function incorporates all function in the class to get the desired result. Takes in all inputs from the instance initialising object. It performs:

            - xaxis data preparation and processing
            - yaxis data preparation and processing

        Returns
        -------
        d : :obj:`collections.namedtuple`
            A named tuple containing all processed x-axis data, errors and label, as well as both pairs of y-axis data, their error margins and labels. Items from this tuple can be gotten by using the dot notation.
        """

        Data = namedtuple("Data", "x xlabel y ylabel")

        xs = self.x_only()

        ys = self.y_only()

        x_prepd = xs.x
        xlabel = xs.xlabel
        y_prepd = ys.y
        ylabel = ys.ylabel

        if self.xaxis == "channel" or self.xaxis == "frequency":
            if y_prepd.ndim == 3:
                y_prepd = y_prepd.transpose("chan", "row", "corr")
            else:
                y_prepd = y_prepd.T

        d = Data(x=x_prepd, xlabel=xlabel, y=y_prepd, ylabel=ylabel)

        return d

    def act(self):
        """Activate the :meth:`ragavi.ragavi.DataCoreProcessor.blackbox`
        """
        return self.blackbox()

    def x_only(self):
        """Return only x-axis data and label

        Returns
        -------
        d : :obj:`collections.namedtuple`
            Named tuple containing x-axis data and x-axis label. Items in the tuple can be accessed by using the dot notation.
        """
        Data = namedtuple("Data", "x xlabel")

        x_data, xlabel = self.get_xaxis_data()
        if self.xaxis == "uvwave":
            # compute uvwave using the available selected frequencies
            freqs = vu.get_frequencies(self.ms_name, spwid=self.ddid,
                                       chan=self.chan,
                                       cbin=self.cbin).values

            x = self.prep_xaxis_data(x_data, freq=freqs)

        # if we have x-axes corresponding to ydata
        elif self.xaxis in ["phase", "amplitude", "real", "imaginary"]:
            x = self.prep_yaxis_data(x_data, yaxis=self.xaxis)
        else:
            x = self.prep_xaxis_data(x_data)

        d = Data(x=x, xlabel=xlabel)

        return d

    def y_only(self):
        """Return only y-axis data and label

        Returns
        -------
        d : :obj:`collections.namedtuple`
            Named tuple containing x-axis data and x-axis label. Items in the tuple can be accessed by using the dot notation.
        """

        Data = namedtuple("Data", "y ylabel")
        y_data, ylabel = self.get_yaxis_data()

        y = self.prep_yaxis_data(y_data)

        d = Data(y=y, ylabel=ylabel)

        return d


##################### Plot related functions ###########################

def gen_image(df, x_min, x_max, y_min, y_max,  c_height, c_width,  cat=None,
              color=None, i_labels=None, ph=PLOT_HEIGHT, pw=PLOT_WIDTH,
              x_axis_type="linear", x_name=None, x=None, xlab=None,
              y_axis_type="linear", y_name=None, ylab=None, title=None,
              xds_table_obj=None,  **kwargs):
    """ Generate single bokeh figure """

    add_cbar = kwargs.get("add_cbar", True)
    add_title = kwargs.get("add_title", True)
    add_subtitle = kwargs.get("add_subtitle", False)
    add_xaxis = kwargs.get("add_xaxis", True)
    add_yaxis = kwargs.get("add_yaxis", True)

    x_range = (x_min, x_max)
    y_range = (y_min, y_max)
    dw, dh = x_max - x_min, y_max - y_min

    # shorten some names
    tf = ds.transfer_functions

    # perform data aggregation
    # shaded data and the aggregated data
    logger.info("Launching datashader")
    agg = image_callback(df, xr=x_range, yr=y_range, w=c_width, h=c_height,
                         x=x_name, y=y_name, cat=cat)

    logger.info("Creating Bokeh Figure")
    fig = create_bk_fig(xlab=xlab, ylab=ylab, title=title, x_min=x_min,
                        x_max=x_max,
                        x_axis_type=x_axis_type, x_name=x_name, y_name=y_name,
                        pw=pw, ph=ph, add_title=add_title,
                        add_xaxis=add_xaxis, add_yaxis=add_yaxis,
                        fix_plotsize=True)
    if cat:
        img = tf.shade(agg, color_key=color[:agg[cat].size])
        if add_cbar:
            cbar = make_cbar(agg[cat].values, cat, cmap=color[:agg[cat].size])
            fig.add_layout(cbar, "right")
    else:
        img = tf.shade(agg, cmap=color)

    cds = ColumnDataSource(data=dict(image=[img.data], x=[x_min],
                                     y=[y_min], dw=[dw], dh=[dh]))

    image_glyph = ImageRGBA(image="image", x='x', y='y', dw="dw", dh="dh",
                            dilate=False)

    if add_xaxis:
        # some formatting on the x and y axes
        if x_name.lower() in ["channel", "frequency"]:
            try:
                p_title = fig.above.pop()
            except IndexError:
                p_title = None
            fig = add_axis(fig=fig, axis_range=x.chan.values,
                           ax_label="Channel")
            if p_title:
                fig.add_layout(p_title, "above")
        elif x_name.lower() == "time":
            xaxis = fig.select(name="p_x_axis")[0]
            xaxis.formatter = DatetimeTickFormatter(hourmin=["%H:%M"])
        elif x_name.lower() == "phase":
            xaxis = fig.select(name="p_x_axis")[0]
            xaxis.formatter = PrintfTickFormatter(format=u"%f\u00b0")

    if add_yaxis:
        if y_name.lower() == "phase":
            yaxis = fig.select(name="p_y_axis")[0]
            yaxis.formatter = PrintfTickFormatter(format=u"%f\u00b0")

    # only to be added in iteration mode
    if add_subtitle:
        # getting the iteration axis
        chunk_attrs = xds_table_obj.attrs

        # data desc id is the default grouping mode
        if len(chunk_attrs) > 1:
            i_axis = list(chunk_attrs.keys())
            i_axis.remove("DATA_DESC_ID")
        else:
            i_axis = list(chunk_attrs.keys())

        h_tool = fig.select(name="p_htool")[0]

        p_txtt = Text(x="x", y="y", text="text", text_font="monospace",
                      text_font_style="bold", text_font_size="10pt",
                      text_align="center")

        p_txtt_src = ColumnDataSource(data=dict(x=[x_min + (dw * 0.5)],
                                                y=[y_max * 0.87]))
        if i_labels is not None:
            # Only in the case of a baseline
            if {"ANTENNA1", "ANTENNA2"} <= set(i_axis):
                i_axis.sort()
                # Don't split the next line, will result in bad formatting
                p_txtt_src.add(
                    [f"""{i_labels[chunk_attrs.get(i_axis[0])]}, {i_labels[chunk_attrs.get(i_axis[1])]}"""],
                    name="text")
                i_axis_data = f"""{ i_labels[chunk_attrs[i_axis[0]]] },
                { i_labels[chunk_attrs.get(i_axis[1])] }"""

                h_tool.tooltips.append(("Baseline", "@i_axis"))
            else:
                p_txtt_src.add(
                    [f"{i_labels[chunk_attrs.get(i_axis[0])]}"],
                    name="text")
                i_axis_data = i_labels[chunk_attrs.get(i_axis[0])]
                h_tool.tooltips.append((f"{i_axis[0].capitalize()}", "@i_axis"))

        else:
            # only get the associated ID
            p_txtt_src.add([f"{i_axis[0].capitalize()}: {chunk_attrs[i_axis[0]]}"],
                           name="text")
            i_axis_data = chunk_attrs[i_axis[0]]
            h_tool.tooltips.append((f"{i_axis[0].capitalize()}", "@i_axis"))

        if "DATA_DESC_ID" not in i_axis:
            cds.add([chunk_attrs["DATA_DESC_ID"]], "spw")
            h_tool.tooltips.append(("Spw", "@spw"))

        cds.add([i_axis_data], "i_axis")

        fig.add_glyph(p_txtt_src, p_txtt)

    fig.add_glyph(cds, image_glyph)
    return fig


def gen_grid(df, x_min, x_max, y_min, y_max, c_height, c_width, cat=None,
             cat_vals=None, color=None,  ms_name=None, ncols=9,
             pw=190, ph=100, title=None, x=None, x_axis_type="linear",
             x_name=None, xlab=None, y_name=None, ylab=None,
             xds_table_obj=None):
    """ Generate bokeh grid of figures"""
    n_grid = []

    nrows = int(np.ceil(cat_vals.size / ncols))

    # my ideal case is 9 columns and for this, the ideal width is 205
    pw = int(pw / ncols)
    ph = int(0.83 * pw)

    # if there are more columns than there are items
    if ncols > cat_vals.size:
        pw = int((pw * ncols) / cat_vals.size)
        ph = int(PLOT_HEIGHT * 0.90)
        ncols = cat_vals.size

    # If there is still space extend height
    if (nrows * ph) < (PLOT_HEIGHT * 0.85):
        ph = int((PLOT_HEIGHT * 0.85) / nrows)

    x_range = (x_min, x_max)
    y_range = (y_min, y_max)
    dw, dh = x_max - x_min, y_max - y_min

    # shorten some names
    tf = ds.transfer_functions

    # perform data aggregation
    # shaded data and the aggregated data
    logger.info("Launching datashader")
    agg = image_callback(df, x_range, y_range, c_width, c_height, x=x_name,
                         y=y_name, cat=cat)

    # get actual field or corr names
    i_names = None
    if cat.lower() == "corr":
        i_names = vu.get_polarizations(ms_name)
    elif cat.lower() == "field_id":
        i_names = vu.get_fields(ms_name).values.tolist()
    elif cat.lower() in ["antenna1", "antenna2"]:
        i_names = vu.get_antennas(ms_name).values.tolist()
    elif cat.lower() == "baseline":
        # get the only bl numbers. This function returns an iterator obj
        ubl = create_bl_data_array(xds_table_obj, bl_combos=True)
        ant_names = vu.get_antennas(ms_name).values

        # create list of unique baselines with the ant_names
        i_names = [f"{bl}: {ant_names[pair[0]]}, {ant_names[pair[1]]}"
                   for bl, pair in enumerate(ubl)]

    logger.info("Creating Bokeh grid")

    for i, c_val in enumerate(cat_vals):
        # some text title for the iterated columns
        p_txtt = Text(x="x", y="y", text="text", text_font="monospace",
                      text_font_style="bold", text_font_size="10pt",
                      text_align="center")

        p_txtt_src = ColumnDataSource(data=dict(x=[x_min + (dw * 0.5)],
                                                y=[y_max * 0.87],
                                                text=[""]))
        if i_names:
            p_txtt_src.add([f"{i_names[c_val]}"], name="text")
            i_axis_data = [i_names[c_val]]
        else:
            p_txtt_src.add([f"{c_val}"], name="text")
            i_axis_data = [c_val]

        s_agg = agg.sel(**{cat: c_val})
        s_img = tf.shade(s_agg, cmap=color)

        s_ds = ColumnDataSource(data=dict(image=[s_img.data], x=[x_min],
                                          i_axis=i_axis_data,
                                          y=[y_min], dw=[dw], dh=[dh]))
        s_ds.add(i_axis_data, "i_axis")

        ir = ImageRGBA(image="image", x='x', y='y', dw="dw", dh="dh",
                       dilate=False)

        # Add x-axis to all items in the last row
        if (i >= ncols * (nrows - 1)):
            add_xaxis = True
        else:
            add_xaxis = False

        # Add y-axis to all items in the first columns
        if (i % ncols == 0):
            add_yaxis = True
        else:
            add_yaxis = False

        f = create_bk_fig(xlab=xlab, ylab=ylab, title=title, x_min=x_min,
                          x_max=x_max,
                          x_axis_type=x_axis_type, x_name=x_name,
                          y_name=y_name, fix_plotsize=True,
                          pw=pw, ph=ph, add_title=False, add_xaxis=add_xaxis,
                          add_yaxis=add_yaxis)

        f.add_glyph(s_ds, ir)
        f.add_glyph(p_txtt_src, p_txtt)

        # Add some information on the tooltip
        h_tool = f.select(name="p_htool")[0]
        h_tool.tooltips.append((cat, "@i_axis"))

        if add_xaxis:
            if x_name.lower() in ["channel", "frequency"]:
                f = add_axis(fig=f, axis_range=x.chan.values,
                             ax_label="Channel")
            elif x_name.lower() == "time":
                xaxis = f.select(name="p_x_axis")[0]
                xaxis.formatter = DatetimeTickFormatter(hourmin=["%H:%M"])
            elif x_name.lower() == "phase":
                xaxis = f.select(name="p_x_axis")[0]
                xaxis.formatter = PrintfTickFormatter(format=u"%f\u00b0")

        if add_yaxis:
            # Set formating if yaxis is phase
            if y_name.lower() == "phase":
                yaxis = f.select(name="p_y_axis")[0]
                yaxis.formatter = PrintfTickFormatter(format=u"%f\u00b0")

        n_grid.append(f)

    # title_div = Div(text=title, align="center", width=PLOT_WIDTH,
    #                 style={"font-size": "24px", "height": "50px",
    #                        "text-align": "centre",
    #                        "font-weight": "bold",
    #                        "font-family": "monospace"},
    #                 sizing_mode="stretch_width")
    n_grid = grid(children=n_grid, ncols=ncols, nrows=nrows,
                  sizing_mode="stretch_both")
    n_grid.tags = [ncols, nrows]
    # final_grid = gridplot(children=[title_div, n_grid], ncols=1)

    return n_grid


@time_wrapper
def image_callback(xy_df, xr, yr, w, h, x=None, y=None, cat=None):
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=xr, y_range=yr)
    logger.info("Datashader aggregation starting")

    if cat:
        with ProgressBar():
            agg = cvs.points(xy_df, x, y, ds.count_cat(cat))
    else:
        with ProgressBar():
            agg = cvs.points(xy_df, x, y, ds.count())

    logger.info("Aggregation done")

    return agg


def make_cbar(cats, category, cmap=None):
    """Initiate a colorbar for categorical data
    Parameters
    ----------
    cats: :obj:`np.ndarray`
        Available category ids e.g scan_number, field id etc
    category: :obj:`str`
        Name of the categorizing axis
    cmap: :obj:`matplotlib.cmap`
        Matplotlib colormap

    Returns
    -------
    cbar: :obj:`bokeh.models`
        Colorbar instance
    """
    lo = np.min(cats)
    hi = np.max(cats)
    cats = cats.astype(str)
    category = category.capitalize()
    ncats = cats.size

    # format the cateogry name for cbar title
    if "_" in category:
        if category == "Data_desc_id":
            category = "Spw"
        else:
            category = category.split("_")[0]

    # not using categorical because we'll need to define our own colors
    cmapper = LinearColorMapper(palette=cmap[:ncats], low=lo, high=hi)
    b_ticker = BasicTicker(desired_num_ticks=ncats,
                           num_minor_ticks=0)

    cbar = ColorBar(color_mapper=cmapper, label_standoff=12,
                    border_line_color=None, ticker=b_ticker,
                    location=(0, 0), title=category, title_standoff=5,
                    title_text_font_size="10pt", title_text_align="left",
                    title_text_font="monospace", minor_tick_line_width=0,
                    title_text_font_style="normal")

    return cbar


def plotter(x, y, xaxis, xlab='', yaxis="amplitude", ylab='',
            c_height=None, c_width=None, chunk_no=None, color="blue",
            colour_axis=None, i_labels=None, iter_axis=None, ms_name=None,
            ncols=9, nrows=None, plot_height=None, plot_width=None,
            x_min=None, x_max=None, y_min=None, y_max=None,
            xds_table_obj=None):
    """Perform plotting

    This is responsible for:

    - Selection of the iteration column. ie. Setting it to a categorical column
    - Calculation of maximums and minimums for the plot

    Parameters
    ----------
    x : :obj:`xarray.DataArray`
        X data to plot
    y:  :obj:`xarray.DataArray`
        Y data to plot
    xaxis : :obj:`str`
        xaxis selected for plotting
    xlab : :obj:`str`
        Label to appear on x-axis
    yaxis : :obj:`str`
        Y-axis selected for plotting
    ylab : :obj:`str`
        Label to appear on y-axis
    iter_axis : :obj:`str`
        Column in the dataset over which to iterate.

    color :  :obj:`str`, :obj:`colormap`, :obj:`itertools.cycler`

    xds_table_obj : :obj:`xarray.Dataset`
        Dataset object containing the columns of the MS. This is passed on in case there are items required from the actual dataset.
    ms_name : :obj:`str`
        Name or [can include path] to Measurement Set
    x_min: :obj:`float`
        Minimum x value to be plotted

        Note
        ----
        This may be difficult to achieve in the case where :obj:`xaxis` is time because time in ``ragavi-vis`` is converted into milliseconds from epoch for ease of plotting by ``bokeh``.
    x_max: :obj:`float`
        Maximum x value to be plotted
    y_min: :obj:`float`
        Minimum y value to be plotted
    y_max: :obj:`float`
        Maximum y value to be plotted

    Returns
    -------
    fig: :obj:`bokeh.plotting.figure`
    """

    # change xaxis name to frequency if xaxis is channel. For df purpose
    xaxis = "frequency" if xaxis == "channel" else xaxis

    # changing the Name of the dataArray to the name provided as xaxis
    try:
        x.name = xaxis.capitalize()
    except AttributeError:
        logger.error(
            "Invalid operation. Iteration axis may be contained in the x-axis, or may be the same as the x-axis.")
        logger.error(
            "Please change the x-axis or the iteration axis and try again. Exiting.")
        sys.exit(-1)
    y.name = yaxis.capitalize()

    # set plot title name
    title = f"{y.name} vs {x.name}"

    if xaxis == "time":
        # bounds for x
        bds = np.array([np.nanmin(x), np.nanmax(x)]).astype(datetime)
        # creating the x-xaxis label
        xlab = "Time {} to {}".format(bds[0].strftime("%Y-%m-%d %H:%M:%S"),
                                      bds[-1].strftime("%Y-%m-%d %H:%M:%S"))
        # convert to milliseconds from epoch for bokeh
        x = x.astype("float64") * 1000
        x_axis_type = "datetime"
    else:
        x_axis_type = "linear"

    # setting maximum and minimums if they are not user defined
    if x_min == None:
        x_min = x.min().data

    if x_max == None:
        x_max = x.max().data

    if x_min == x_max:
        x_max += 1

    if y_min == None:
        y_min = y.min().data

    if y_max == None:
        y_max = y.max().data

    if y_min == y_max:
        y_max += 1

    logger.info("Calculating x and y Min and Max ranges")
    with ProgressBar():
        x_min, x_max, y_min, y_max = compute(x_min, x_max, y_min, y_max)
    logger.info("Done")

    # inputs to gen image
    im_inputs = dict(c_height=c_height, c_width=c_width,
                     chunk_no=chunk_no, color=color, i_labels=i_labels,
                     title=title, x=x, x_axis_type=x_axis_type, xlab=xlab,
                     x_name=x.name, y_name=y.name, ylab=ylab,
                     xds_table_obj=xds_table_obj, fix_plotsize=True,
                     add_grid=True, add_subtitle=True, add_title=False)

    if iter_axis and colour_axis:
        # NOTE!!!
        # Iterations are done over daskms groupings except for chan & corr
        # Colouring can therefore be done here by categorization
        title += f""" Iterated By: {iter_axis.capitalize()} Colourised By: {colour_axis.capitalize()}"""
        # Add x-axis to all items in the last row
        if (chunk_no >= ncols * (nrows - 1)):
            add_xaxis = True
        else:
            add_xaxis = False

        # Add y-axis to all items in the first columns
        if (chunk_no % ncols == 0):
            add_yaxis = True
        else:
            add_yaxis = False

        xy_df, cat_values = create_categorical_df(colour_axis, x, y,
                                                  xds_table_obj)

        # generate resulting image
        image = gen_image(xy_df, x_min, x_max, y_min, y_max,
                          cat=colour_axis, ph=plot_height, pw=plot_width,
                          add_cbar=False, add_xaxis=add_xaxis,
                          add_yaxis=add_yaxis, **im_inputs)

    elif iter_axis:
        # NOTE!!!
        # Iterations are done over daskms groupings except for chan & corr
        title += f" Iterated By: {iter_axis.capitalize()}"

        if (chunk_no >= ncols * (nrows - 1)):
            add_xaxis = True
        else:
            add_xaxis = False

        # Add y-axis to all items in the first columns
        if (chunk_no % ncols == 0):
            add_yaxis = True
        else:
            add_yaxis = False

        xy_df = create_df(x, y, iter_data=None)

        image = gen_image(xy_df, x_min, x_max, y_min, y_max, cat=None,
                          ph=plot_height, pw=plot_width, add_cbar=False,
                          add_xaxis=add_xaxis, add_yaxis=add_yaxis,
                          **im_inputs)

    elif colour_axis:
        title += f" Colourised By: {colour_axis.capitalize()}"

        xy_df, cat_values = create_categorical_df(colour_axis, x, y,
                                                  xds_table_obj)

        # generate resulting image
        image = gen_image(xy_df, x_min, x_max, y_min, y_max, cat=colour_axis,
                          ph=PLOT_HEIGHT, pw=PLOT_WIDTH, add_cbar=True,
                          add_xaxis=True, add_yaxis=True, **im_inputs)

    else:
        logger.info("Creating Dataframe")
        xy_df = create_df(x, y, iter_data=None)[[x.name, y.name]]

        # generate resulting image
        im_inputs.update(dict(add_subtitle=False, add_title=True))
        image = gen_image(xy_df, x_min, x_max, y_min, y_max, cat=None,
                          ph=PLOT_HEIGHT, pw=PLOT_WIDTH, add_cbar=False,
                          add_xaxis=True, add_yaxis=True, **im_inputs)

    return image, title


###################### MS related functions ############################
def antenna_iter(ms_name, columns, **kwargs):
    """ Return a list containing iteration over antennas in respective SPWs
    Parameters
    ----------
    ms_name: :obj:`str`
        Name of the MS

    columns : :obj:`list`
        Columns that should be present in the dataset

    Returns
    -------
    outp: :obj:`list`
        A list containing data for each individual antenna. This list is
        ordered by antenna and SPW.
    """

    taql_where = kwargs.get("taql_where", "")
    table_schema = kwargs.get("table_schema", None)
    chunks = kwargs.get("chunks", 1000)

    # Shall be prepended to the later antenna selection
    if taql_where:
        taql_where += " && "

    outp = []
    # ant_names = vu.get_antennas(ms_name).values.tolist()
    n_ants = vu.get_antennas(ms_name).size
    n_spws = vu.get_frequencies(ms_name).row.size

    for d in range(n_spws):
        for a in range(n_ants):
            sel_str = taql_where + \
                f"ANTENNA1=={a} || ANTENNA2=={a} && DATA_DESC_ID=={d}"

            # do not group to capture all the data
            sub = xm.xds_from_ms(ms_name,
                                 taql_where=sel_str,
                                 table_schema=table_schema,
                                 chunks=chunks, columns=columns,
                                 group_cols=[])[0]

            # Add the selection attributes
            sub.attrs = dict(ANTENNA=a, DATA_DESC_ID=d)

            outp.append(sub)

    return outp


def corr_iter(subs):
    """ Return a list containing iteration over corrs in respective SPWs

    Parameters
    ----------
    subs: :obj:`list`
        List containing subtables for the daskms grouped data

    Returns
    -------
    outp: :obj:`list`
        A list containing data for each individual corr. This list is
        ordered by corr and SPW.

    # NOTE: can also be used for chan iteration. Will require name change
    """
    outp = []
    n_corrs = subs[0].corr.size

    for sub in subs:
        for c in range(n_corrs):
            nsub = sub.copy(deep=True).sel(corr=c)
            nsub.attrs["Corr"] = c
            outp.append(nsub)
    return outp


def get_ms(ms_name,  ants=None, cbin=None, chan_select=None, chunks=None,
           corr_select=None, colour_axis=None, data_col="DATA", ddid=None,
           fid=None, iter_axis=None, scan=None, tbin=None,  where=None,
           x_axis=None):
    """Get xarray Dataset objects containing Measurement Set columns of the selected data

    Parameters
    ----------
    ms_name: :obj:`str`
        Name of your MS or path including its name
    chunks: :obj:`str`
        Chunk sizes for the resulting dataset.
    ants: :obj:`str`
        Values for antennas in ANTENNA1 whose baselines will be selected
    cbin: :obj:`int`
        Number of channels binned together for channel averaging
    colour_axis: :obj:`str`
        Axis to be used for colouring
    iter_axis: :obj:`str`
        Axis to iterate over
    data_col: :obj:`str`
        Data column to be used. Defaults to 'DATA'
    ddid: :obj:`int`
        DATA_DESC_ID or spectral window to choose. Defaults to all
    fid: :obj:`int`
        Field id to select. Defaults to all
    scan: :obj:`int`
        SCAN_NUMBER to select. Defaults to all
    tbin: :obj:`float`
        Time in seconds to bin for time averaging
    x_axis: :obj:`
        The chosen x-axis
    where: :obj:`str`
        TAQL where clause to be used with the MS.
    chan_select: :obj:`int` or :obj:`slice`
        Channels to be selected 
    corr_select: :obj:`int` or :obj:`slice`
        Correlations to be selected

    Returns
    -------
    tab_objs: :obj:`list`
        A list containing the specified table objects as  :obj:`xarray.Dataset`
    """
    group_cols = set()

    # Always group by DDID
    group_cols.update({"DATA_DESC_ID"})

    sel_cols = {"FLAG"}

    if iter_axis and iter_axis not in ["corr", "antenna"]:
        if iter_axis == "Baseline":
            group_cols.update({"ANTENNA1", "ANTENNA2"})
        else:
            group_cols.update({iter_axis})

    sel_cols.update(group_cols)
    sel_cols.add(data_col)

    if colour_axis and colour_axis not in ["corr", "channel", "frequency"]:
        if colour_axis == "Baseline":
            sel_cols.update({"ANTENNA1", "ANTENNA2"})
        else:
            sel_cols.add(colour_axis)

    if x_axis in ["channel", "chan", "corr", "frequency"]:
        x_axis = data_col
    else:
        x_axis = get_colname(x_axis, data_col=data_col)
    sel_cols.add(x_axis)

    ms_schema = MS_SCHEMA.copy()
    ms_schema["WEIGHT_SPECTRUM"] = ms_schema["DATA"]

    # defining part of the gain table schema
    if data_col not in ["DATA", "CORRECTED_DATA"]:
        ms_schema[data_col] = ms_schema["DATA"]

    if chunks is None:
        chunks = dict(row=10000)
    else:
        dims = ["row", "chan", "corr"]
        chunks = {k: int(v) for k, v in zip(dims, chunks.split(','))}

    # always ensure that where stores something
    if where is None:
        where = []
    else:
        where = [where]

    if ddid is not None:
        where.append("DATA_DESC_ID IN {}".format(ddid))
    if fid is not None:
        where.append("FIELD_ID IN {}".format(fid))
    if scan is not None:
        where.append("SCAN_NUMBER IN {}".format(scan))
    if ants is not None:
        where.append("ANTENNA1 IN {}".format(ants))

    # combine the strings to form the where clause
    where = " && ".join(where)

    sel_cols = list(sel_cols)
    group_cols = list(group_cols)

    xds_inputs = dict(chunks=chunks, taql_where=where,
                      columns=sel_cols, group_cols=group_cols,
                      table_schema=ms_schema)
    try:
        if cbin or tbin:
            logger.info("Averaging active")
            xds_inputs.update(dict(columns=sel_cols))
            del xds_inputs["table_schema"]
            tab_objs = get_averaged_ms(ms_name, tbin=tbin, cbin=cbin,
                                       chan=chan_select, corr=corr_select,
                                       data_col=data_col, iter_axis=iter_axis,
                                       **xds_inputs)
            if iter_axis == "corr":
                corr_select = None
                tab_objs = corr_iter(tab_objs)
        else:
            if iter_axis == "antenna":
                tab_objs = antenna_iter(ms_name, **xds_inputs)
            else:
                tab_objs = xm.xds_from_ms(ms_name, **xds_inputs)
                if iter_axis == "corr":
                    corr_select = None
                    tab_objs = corr_iter(tab_objs)
            # select channels
            if chan_select is not None:
                tab_objs = [_.sel(chan=chan_select) for _ in tab_objs]
            # select corrs
            if corr_select is not None:
                tab_objs = [_.sel(corr=corr_select) for _ in tab_objs]

        # get some info about the data
        chunk_sizes = tab_objs[0].FLAG.data.chunksize
        chunk_p = tab_objs[0].FLAG.data.npartitions

        logger.info("Chunk sizes: {}".format(chunk_sizes))
        logger.info("Number of Partitions: {}".format(chunk_p))

        return tab_objs
    except:
        logger.error(
            "Invalid DATA_DESC_ID, FIELD_ID, SCAN_NUMBER or TAQL clause")
        sys.exit(-1)


###################### DF related functions ############################
def create_bl_data_array(xds_table_obj, bl_combos=False):
    """Make a dataArray containing baseline numbers

    Parameters
    ----------
    xds_table_obj: :obj:`xarray.Dataset`
        Daskms dataset object
    bl_combos: :obj:`Bool`
        Whether to return only the available baseline combinations
    Returns
    -------
    baseline: :obj:`xarray.DataArray`
        DataArray containing baseline numbers
    """
    ant1 = xds_table_obj.ANTENNA1
    ant2 = xds_table_obj.ANTENNA2

    u_ants = np.unique(np.hstack((np.unique(ant1), np.unique(ant2))))
    u_bls_combos = combinations(np.arange(u_ants.size), 2)

    if bl_combos:
        return u_bls_combos

    logger.info("Populating baseline data")

    # create a baseline array of the same shape as antenna1
    baseline = np.full_like(ant1.values, 0)
    for bl, (p, q) in enumerate(u_bls_combos):
        baseline[(ant1.values == p) & (ant2.values == q)] = bl

    baseline = da.asarray(a=baseline).rechunk(ant1.data.chunksize)
    baseline = ant1.copy(deep=True, data=baseline)
    baseline.name = "Baseline"
    logger.info("Done")
    return baseline


def create_categorical_df(it_axis, x_data, y_data, xds_table_obj):
    """
    it_axis: :obj:`str`
        Column over which to iterate / colourise
    x_data: :obj:`xr.DataArray`
        x-axis data
    y_data: :obj:`xr.DataArray`
        y-axis data
    xds_table_obj: :obj:`xr.Dataset`
        Daskms partition for this chunk of data

    Returns
    -------
    xy_df: :obj:`dask.DataFrame`
        Dask dataframe with the required category
    cat_values: :obj:`np.array`
        Array containing the unique identities of the iteration axis
    """
    # iteration key word: data column name
    if it_axis in ["corr", "chan"]:
        iter_data = xr.DataArray(da.arange(y_data[it_axis].size),
                                 name=it_axis, dims=[it_axis])
    elif it_axis == "Baseline":
        # should only be applicable to color axis
        # get the data array over which to iterate and merge it to x and y
        iter_data = create_bl_data_array(xds_table_obj)
    else:
        if it_axis in xds_table_obj.data_vars.keys():
            iter_data = xds_table_obj[it_axis]
        elif it_axis in xds_table_obj.attrs.keys():
            iter_data = xr.DataArray(
                da.full_like(x_data, xds_table_obj.attrs[it_axis]),
                name=it_axis, dims=x_data.dims, coords=y_data.coords)

        else:
            logger.error("Specified data column not found.")
            sys.exit(-1)

    logger.info("Creating Dataframe")

    xy_df = create_df(x_data, y_data, iter_data=iter_data)[[x_data.name,
                                                            y_data.name,
                                                            it_axis]]
    cat_values = np.unique(iter_data.values)

    xy_df = xy_df.astype({it_axis: "category"})
    xy_df[it_axis] = xy_df[it_axis].cat.as_known()

    return xy_df, cat_values


def create_df(x, y, iter_data=None):
    """Create a dask dataframe from input x, y and iterate columns if 
    available. This function flattens all the data into 1-D Arrays

    Parameters
    ----------
    x: :obj:`dask.array`
        Data for the x-axis
    y: :obj:`dask.array`
        Data for the y-axis
    iter_ax: :obj:`dask.array`
        iteration axis if possible

    Returns
    -------
    new_ds: :obj:`dask.Dataframe`
        Dataframe containing the required columns
    """

    x_name = x.name
    y_name = y.name

    # flatten and chunk the data accordingly
    nx, ny = massage_data(x, y, get_y=True)

    # declare variable names for the xarray dataset
    var_names = {x_name: ("row", nx),
                 y_name: ("row", ny)}

    if iter_data is not None:
        i_name = iter_data.name
        iter_data = massage_data(iter_data, y, get_y=False, iter_ax=i_name)
        var_names[i_name] = ("row", iter_data)

    logger.info("Creating dataframe")
    new_ds = xr.Dataset(data_vars=var_names)
    new_ds = new_ds.to_dask_dataframe()
    new_ds = new_ds.dropna()
    return new_ds


def massage_data(x, y, get_y=False, iter_ax=None):
    """Massages x-data into a size similar to that of y-axis data via
    the necessary repetitions. This function also flattens y-axis data
    into 1-D.

    Parameters
    ----------
    x: :obj:`xr.DataArray`
        Data for the x-axis
    y: :obj:`xr.DataArray`
        Data for the y-axis
    get_y: :obj:`bool`
        Choose whether to return y-axis data or not

    Returns
    -------
    x: :obj:`dask.array`
        Data for the x-axis
    y: :obj:`dask.array`
        Data for the y-axis
    """

    iter_cols = ["ANTENNA1", "ANTENNA2", "FIELD_ID", "SCAN_NUMBER",
                 "DATA_DESC_ID", "Baseline"]
    # can add 'chan','corr' to this list

    # available dims in the x and y axes
    y_dims = set(y.dims)
    x_dims = set(x.dims)

    # find dims that are not available in x and only avail in y
    req_x_dims = y_dims - x_dims

    if not isinstance(x.data, da.Array):
        nx = da.asarray(x.data)
    else:
        nx = x.data

    if x.ndim > 1:
        nx = nx.ravel()

    if len(req_x_dims) > 0:

        sizes = []

        # calculate dim sizes for repetion of x_axis
        for item in req_x_dims:
            sizes.append(y[item].size)

        if iter_ax == "corr":
            nx = nx.reshape(1, nx.size).repeat(np.prod(sizes), axis=0).ravel()
            nx = nx.rechunk(y.data.ravel().chunks)

        elif y.dims[0] == "chan" and iter_ax in iter_cols:
            # Because data is transposed this must be done
            nx = nx.reshape(1, nx.size).repeat(np.prod(sizes), axis=0).ravel()
            nx = nx.rechunk(y.data.ravel().chunks)

        else:
            try:
                nx = nx.map_blocks(np.repeat,
                                   np.prod(sizes),
                                   chunks=y.data.ravel().chunks)
            except ValueError:
                # for the non-xarray dask data
                nx = nx.repeat(np.prod(sizes)).rechunk(y.data.ravel().chunks)
    if get_y:
        # flatten y data
        # get_data also
        ny = y.data.ravel()
        return nx, ny
    else:
        return nx


###################### Validation functions ############################
def get_colname(inp, data_col=None):
    aliases = {
        "amplitude": data_col,
        "antenna": "antenna",
        "antenna1": "ANTENNA1",
        "antenna2": "ANTENNA2",
        "baseline": "Baseline",
        # "baseline": ["ANTENNA1", "ANTENNA2"],
        "chan": "chan",
        "corr": "corr",
        "frequency": "chan",
        "field": "FIELD_ID",
        "imaginary": data_col,
        "phase": data_col,
        "real": data_col,
        "scan": "SCAN_NUMBER",
        "spw": "DATA_DESC_ID",
        "time": "TIME",
        "uvdistance": "UVW",
        "uvwave": "UVW"
    }
    if inp != None:
        col_name = aliases[inp]
    else:
        col_name = None
    return col_name


def validate_axis_inputs(inp):
    """Check if the input axes tally with those that are available
    Parameters
    ----------
    inp: :obj:`str`
        User's axis input
    choices: :obj:`list`
        Available choices for a specific axis
    alts: :obj:`dict`
        All the available altenatives for the various axes

    Returns
    -------
    inp: :obj:`str`
        Validated string
    """
    alts = {}
    alts["amp"] = alts["Amp"] = "amplitude"
    alts["ant1"] = alts["Antenna1"] = "antenna1"
    alts["ant2"] = alts["Antenna2"] = "antenna2"
    alts["ant"] = alts["Antenna"] = "antenna"
    alts["Baseline"] = alts["bl"] = "baseline"
    alts["chan"] = alts["Channel"] = "channel"
    alts["correlation"] = alts["Corr"] = "corr"
    alts["freq"] = alts["Frequency"] = "frequency"
    alts["Field"] = "field"
    alts["imaginary"] = alts["Imag"] = alts["imag"] = "imaginary"
    alts["Real"] = "real"
    alts["Scan"] = "scan"
    alts["Spw"] = "spw"
    alts["Time"] = "time"
    alts["Phase"] = "phase"
    alts["UVdist"] = alts["uvdist"] = "uvdistance"
    alts["uvdistl"] = alts["uvdist_l"] = alts["UVwave"] = "uvwave"

    # convert to proper name if in other name
    if inp in alts:
        inp = alts[inp]

    return inp


def link_grid_plots(plot_list):
    """Link all the plots in the X and Y axes

    """
    if not isinstance(plot_list[0], Plot):
        plots = []
        plot_list = plot_list[0].children
        for item in plot_list:
            plots.append(item[0])
    else:
        plots = plot_list

    n_plots = len(plots)
    init_xr = plots[0].x_range
    init_yr = plots[0].y_range
    for i in range(1, n_plots):
        plots[i].x_range = init_xr
        plots[i].y_range = init_yr

    return 0


############################## Main Function ###########################
def main(**kwargs):
    """Main function that launches the visibilities plotter"""

    if "options" in kwargs:
        NB_RENDER = False
        options = kwargs.get("options", None)
        ants = options.ants
        chan = options.chan
        chunks = options.chunks
        c_width = options.c_width
        c_height = options.c_height
        colour_axis = validate_axis_inputs(options.colour_axis)
        corr = options.corr
        data_column = options.data_column
        ddid = options.ddid
        fields = options.fields
        flag = options.flag
        iter_axis = validate_axis_inputs(options.iter_axis)
        html_name = options.html_name
        mycmap = options.mycmap
        mytabs = options.mytabs
        mem_limit = options.mem_limit
        n_cols = options.n_cols
        n_cores = options.n_cores
        scan = options.scan
        where = options.where  # taql where
        xaxis = validate_axis_inputs(options.xaxis)
        xmin = options.xmin
        xmax = options.xmax
        ymin = options.ymin
        ymax = options.ymax
        yaxis = validate_axis_inputs(options.yaxis)
        tbin = options.tbin
        cbin = options.cbin

    # change iteration axis names to actual column names
    colour_axis = get_colname(colour_axis)
    iter_axis = get_colname(iter_axis)

    # number of processes or threads/ cores
    dask_config.set(num_workers=n_cores, memory_limit=mem_limit)

    if len(mytabs) > 0:
        mytabs = [x.rstrip("/") for x in mytabs]
    else:
        logger.error("ragavi exited: No Measurement set specified.")
        sys.exit(-1)

    # ensure that x and y axes are not the same
    try:
        assert xaxis != yaxis
    except AssertionError:
        logger.error("Invalid operation. X- and Y- axis provided are similar")
        sys.exit(-1)

    for mytab in mytabs:
        ################################################################
        ############## Form valid selection statements #################

        if fields is not None:
            if '~' in fields or ':' in fields or fields.isdigit():
                fields = vu.resolve_ranges(fields)
            elif ',' in fields:
                # check if all are digits in fields, if not convert field
                # name to field id and join all the resulting field ids with a
                # comma
                fields = ",".join(
                    [str(vu.name_2id(mytab, x)) if not x.isdigit() else x for x in fields.split(',')])
                fields = vu.resolve_ranges(fields)
            else:
                fields = str(vu.name_2id(mytab, fields))
                fields = vu.resolve_ranges(fields)

        if options.scan != None:
            scan = vu.resolve_ranges(scan)
        if options.ants != None:
            ants = vu.resolve_ranges(ants)

        if options.ddid != None:
            n_ddid = vu.slice_data(ddid)
            ddid = vu.resolve_ranges(ddid)
        else:
            # set data selection to all unless ddid is specified
            n_ddid = slice(0, None)

        # capture channels or correlations to be selected
        chan = vu.slice_data(chan)

        ################################################################
        ################### Iterate over subtables #####################

        # translate corr labels to indices if need be
        try:
            corr = vu.slice_data(corr)
        except ValueError:
            corr = corr.split(",")
            corr_labs = vu.get_polarizations(mytab)
            for c in range(len(corr)):
                corr[c] = corr_labs.index(corr[c])

        # if there are id labels to be gotten, get them
        if iter_axis == "corr":
            i_labels = vu.get_polarizations(mytab)
        elif iter_axis == "FIELD_ID":
            i_labels = vu.get_fields(mytab).values.tolist()
        elif iter_axis in ["antenna", "ANTENNA1", "ANTENNA2", "Baseline"]:
            i_labels = vu.get_antennas(mytab).values.tolist()
        else:
            i_labels = None

        # open MS perform averaging and select desired fields, scans and spws
        partitions = get_ms(mytab, ants=ants,
                            cbin=cbin, chan_select=chan, chunks=chunks,
                            colour_axis=colour_axis, corr_select=corr,
                            data_col=data_column, ddid=ddid, fid=fields,
                            iter_axis=iter_axis, scan=scan, tbin=tbin,
                            where=where, x_axis=xaxis)

        if colour_axis:
            if mycmap:
                mycmap = vu.get_cmap(mycmap, fall_back="glasbey_bw",
                                     src="colorcet")
            else:
                mycmap = vu.get_cmap("glasbey_bw", src="colorcet")
        else:
            if mycmap:
                mycmap = vu.get_cmap(mycmap, fall_back="blues",
                                     src="colorcet")
            else:
                mycmap = vu.get_cmap("blues", src="colorcet")

        # configure number of rows and columns for grid
        n_partitions = len(partitions)
        n_rows = int(np.ceil(n_partitions / n_cols))
        # n_cols known from argparser

        pw = PLOT_WIDTH
        ph = PLOT_HEIGHT

        if iter_axis:
            # my ideal case is 9 columns and for this, the ideal width is 205
            pw = int((205 * 9) / n_cols)
            ph = int(0.83 * pw)
            if not c_width:
                c_width = 200
                logger.info(
                    "Shrinking canvas width {} for iteration".format(c_width))
            if not c_height:
                c_height = 200
                logger.info(
                    "Shrinking canvas height to {} for iteration".format(c_height))
        else:
            if not c_width:
                c_width = 1080
            if not c_height:
                c_height = 720

        # if there are more columns than there are items
        if n_cols > n_partitions:
            pw = int((pw * n_cols) / n_partitions)
            ph = int(PLOT_HEIGHT * 0.90)
            # because the grid is not generated by daskms iteration
            n_cols = n_partitions

        # If there is still space extend height
        if (n_rows * ph) < (PLOT_HEIGHT * 0.85):
            ph = int((PLOT_HEIGHT * 0.85) / n_rows)

        oup_a = []

        # iterating over spectral windows (spws)
        for count, chunk in enumerate(partitions):

            # We check current DATA_DESC_ID
            c_spw = chunk.DATA_DESC_ID

            logger.info("Starting data processing.")

            p_data = DataCoreProcessor(chunk, mytab, xaxis, yaxis, chan=chan,
                                       corr=corr, flag=flag, ddid=c_spw,
                                       datacol=data_column, cbin=cbin)
            ready = p_data.act()

            logger.info(f"\033[92m Plotting {count+1}/{n_partitions}.")
            logger.info(f"{chunk.attrs}\033[0m")

            # black box returns an plottable element / composite element
            # as well a the title to be used for the plot
            fig, title_txt = plotter(
                x=ready.x, xaxis=xaxis, xlab=ready.xlabel,
                y=ready.y, yaxis=yaxis, ylab=ready.ylabel,
                c_height=c_height, c_width=c_width, chunk_no=count,
                color=mycmap, colour_axis=colour_axis,
                i_labels=i_labels, iter_axis=iter_axis,
                ms_name=mytab, ncols=n_cols, nrows=n_rows,
                plot_width=pw, plot_height=ph, xds_table_obj=chunk,
                x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax)

            oup_a.append(fig)

        if iter_axis:
            title_div = Div(text=title_txt, align="center",
                            width=int(PLOT_WIDTH * 0.95),
                            style={"font-size": "24px", "height": "50px",
                                   "text-align": "centre",
                                   "font-weight": "bold",
                                   "font-family": "monospace"},
                            sizing_mode="stretch_width")

            # Link all the plots
            link_grid_plots(oup_a)

            info_text = f"MS: {mytab}\nGrid size: {n_cols} x {n_rows}"
            pre = PreText(text=info_text, width=int(PLOT_WIDTH * 0.95),
                          height=50, align="start", margin=(0, 0, 0, 0))
            final_plot = gridplot(children=oup_a, ncols=n_cols,
                                  sizing_mode="stretch_width")
            final_plot = column(children=[title_div, pre, final_plot])

        else:
            final_plot = oup_a[0]

        logger.info("Plotting complete.")

        if html_name:
            if "html" not in html_name:
                html_name += ".html"
            fname = html_name
        else:
            fname = "{}_{}_{}.html".format(
                mytab.split('/')[-1], yaxis, xaxis)

        output_file(fname, title=fname)
        save(final_plot)

        logger.info("Rendered plot to: {}".format(fname))
        logger.info(wrapper.fill(",\n".join(
            ["{}: {}".format(k, v) for k, v in options.__dict__.items() if v != None])))

        logger.info(">" * (len(fname) + 19))
