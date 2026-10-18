# -*- coding: utf-8 -*-
import sys
import json
import logging
import os

from collections import namedtuple
from datetime import datetime
from itertools import combinations, cycle
from psutil import cpu_count, virtual_memory

import dask.array as da
import dask.dataframe as dd
import daskms as xm
import datashader as ds
import numpy as np
import xarray as xr

from dask import compute, config as dask_config
from dask.diagnostics import ProgressBar
from daskms.table_schemas import MS_SCHEMA

from bokeh.layouts import column, gridplot
from bokeh.io import (output_file, save)
from bokeh.models import (ColorBar, ColumnDataSource, CheckboxGroup,
                          DataRange1d, Line, Legend, LegendItem,
                          CustomJS, DatetimeTickFormatter, Div, ImageRGBA,
                          LinearColorMapper, FixedTicker,
                          PrintfTickFormatter, Plot, PreText,
                          Text)
import ragavi.utils as vu

from ragavi import __version__
from ragavi.averaging import get_averaged_ms
from ragavi.plotting import create_bk_fig, add_axis

logger = logging.getLogger(__name__)

time_wrapper = vu.time_wrapper

# some variables
PLOT_WIDTH = 1920
PLOT_HEIGHT = 1080


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

    def __repr__(self):
        return "DataCoreProcessor(%r, %r, %r, %r)" % (
            self.xds_table_obj, self.ms_name, self.xaxis, self.yaxis)

    def get_xaxis_data(self):
        """Get x-axis data. This function also returns the relevant x-axis label.
        Returns
        -------
        xdata : :obj:`xarray.DataArray`
            X-axis data depending  x-axis selected.
        x_label : :obj:`str`
            Label to appear on the x-axis of the plots.
        """

        logger.debug("DCP: Getting x-axis data")

        if self.xaxis == "antenna1":
            xdata = self.xds_table_obj.ANTENNA1
            x_label = "Antenna1"
        elif self.xaxis == "antenna2":
            xdata = self.xds_table_obj.ANTENNA2
            x_label = "Antenna2"
        elif self.xaxis == "amplitude":
            xdata = self.xds_table_obj[self.datacol]
            x_label = "Amplitude"
        elif self.xaxis in ["frequency", "channel"]:
            xdata = vu.get_frequencies(self.ms_name, spwid=self.ddid,
                                       chan=self.chan, cbin=self.cbin)
            xdata = xdata.chunk(self.xds_table_obj.chunks['chan'])

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
                x_label = "UV Wave [{}]".format(u"\u03bb")
            except UnicodeEncodeError:
                x_label = "UV Wave [K{}]".format(u"\u03bb".encode("utf-8"))
        else:
            logger.error("Invalid xaxis name")
            return

        logger.debug(f"x-axis: {x_label}, column: {xdata.name} selected")

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

        logger.debug("DCP: Getting y-axis data")

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

        logger.debug(f"y-axis: {y_label}, column: {ydata.name} selected")

        return ydata, y_label

    def prep_xaxis_data(self, xdata):
        """Prepare the x-axis data for plotting. Selections also performed here.

        Parameters
        ----------
        xdata: :obj:`xarray.DataArray`
            x-axis data depending  x-axis selected

        Returns
        -------
        prepdx: :obj:`xarray.DataArray`
            Prepared :attr:`xdata`
        """
        logger.debug("DCP: Preping x-axis data")

        if self.xaxis == "channel" or self.xaxis == "frequency":
            prepdx = xdata / 1e9
        elif self.xaxis in ["phase", "amplitude", "real", "imaginary"]:
            logger.debug(f"Switching x-axis data prep to y-axis data prep because of x-axis: {self.xaxis}")
            prepdx = self.prep_yaxis_data(xdata, yaxis=self.xaxis)
        elif self.xaxis == "time":
            prepdx = vu.time_convert(xdata)
        elif self.xaxis == "uvdistance":
            prepdx = vu.calc_uvdist(xdata)
        elif self.xaxis == "uvwave":
            # compute uvwave using the available selected frequencies
            freqs = vu.get_frequencies(self.ms_name, spwid=self.ddid,
                                       chan=self.chan,
                                       cbin=self.cbin).values

            # results are returned in kilo lamnda
            prepdx = vu.calc_uvwave(xdata, freqs)
        elif self.xaxis in ["antenna1", "antenna2", "scan"]:
            prepdx = xdata

        logger.debug("x-axis data ready")

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

        logger.debug("DCP: Preping y-axis data")

        # Doing this because some of xaxis data must be processed here
        if yaxis is None:
            yaxis = self.yaxis

        # if flagging enabled return a list of DataArrays otherwise return a
        # single dataarray
        if self.flag:
            logger.debug("Flagging active. Getting flags")

            flags = self.xds_table_obj.FLAG

            logger.debug("Flags ready. Ensuring there are inactive flags")

            """
            # Checking if there is any un-flagged data
            # active flags eval to True, inverting data so that active flags
              eval to false and in-active eval to true
            # np.any only checks for where values are true
            # will return true if there is any un-flagged data
            # Invert all_flagged to make gramatical sense.

            """
            all_flagged = not np.any(np.logical_not(flags.data)).compute()

            logger.debug(f"Any un-flagged data? {all_flagged}")

            if all_flagged:
                logger.warning(f"All {yaxis} data is flagged. Use -if or --include-flagged to also plot flagged data")
                logger.warning(vu.ctext("Plot will be empty", 'r'))

            processed = self.process_data(ydata, yaxis=yaxis, unwrap=False)

            logger.debug("Applying flags")

            y = processed.where(flags == False)
        else:
            y = self.process_data(ydata, yaxis=yaxis, unwrap=False)

        if self.xaxis in ["channel", "frequency"]:
            # the input shape
            i_shape = str(y.shape)
            if y.ndim == 3:
                y = y.transpose("chan", "row", "corr")
            else:
                y = y.T
            logger.debug(f"xaxis: {self.xaxis}, changing y shape {i_shape} --> {str(y.shape)}")

        logger.debug("y-axis data ready")

        return y

    def process_data(self, ydata, unwrap=False, yaxis=None):
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
        logger.debug("DCP: Calculating y-axis data")

        if yaxis is None:
            yaxis = self.yaxis
        if yaxis == "amplitude":
            y = vu.calc_amplitude(ydata)
        elif yaxis == "imaginary":
            y = vu.calc_imaginary(ydata)
        elif yaxis == "phase":
            y = vu.calc_phase(ydata, unwrap=unwrap)
        elif yaxis == "real":
            y = vu.calc_real(ydata)

        logger.debug("Done")

        return y

    def blackbox(self):
        """Get raw input data and churn out processed x and y data.

        This function incorporates all function in the class to get the
        desired result. Takes in all inputs from the instance initialising
        object. It performs:

            - xaxis data preparation and processing
            - yaxis data preparation and processing

        Returns
        -------
        d : :obj:`collections.namedtuple`
            A named tuple containing all processed x-axis data, errors and
            label, as well as both pairs of y-axis data, their error margins
            and labels. Items from this tuple can be gotten by using the dot
            notation.
        """

        Data = namedtuple("Data", "x xlabel y ylabel")

        xs = self.x_only()

        ys = self.y_only()

        x_prepd = xs.x
        xlabel = xs.xlabel
        y_prepd = ys.y
        ylabel = ys.ylabel

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

def gen_image(df, x_min, x_max, y_min, y_max,
              c_height, c_width,  cat=None, c_labels=None,
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

    # change dtype to uint32 if agg contains a negative number
    if (agg.values < 0).any():
        agg = agg.astype(np.uint32)

    logger.info("Creating Bokeh Figure")

    fig = create_bk_fig(xlab=xlab, ylab=ylab, title=title, x_min=x_min,
                        x_max=x_max,
                        x_axis_type=x_axis_type, x_name=x_name, y_name=y_name,
                        pw=pw, ph=ph, add_title=add_title,
                        add_xaxis=add_xaxis, add_yaxis=add_yaxis,
                        fix_plotsize=True)

    if cat:
        n_cats = agg[cat].size

        color = cycle(color[:n_cats])

        if cat == "Baseline":
            bls = create_bl_data_array(xds_table_obj, bl_combos=True)
            for _b in range(len(bls)):
                bls[_b] = ", ".join(np.array(c_labels)[[bls[_b]]].tolist())
            c_labels = bls

        if np.max(agg.values) > 0:
            img = tf.shade(agg, color_key=color)

            if add_cbar:
                append_cbar(cats=agg[cat].values, category=cat,
                            cmap=color, labels=c_labels,
                            ax=fig, x_min=x_min, y_min=y_min)

        else:
            # when no data is available remove extra categorical dim
            # failure to do this will cause a ValueError
            img = tf.shade(agg[:, :, 0], color_key=color[:n_cats])

    else:
        img = tf.shade(agg, cmap=color)

    cds = ColumnDataSource(data=dict(image=[img.data], x=[x_min],
                                     y=[y_min], dw=[dw], dh=[dh]),
                           name="image_data")

    image_glyph = ImageRGBA(image="image", x='x', y='y', dw="dw", dh="dh",
                            dilate=False)

    if add_xaxis:
        # some formatting on the x and y axes
        logger.debug("Formatting x-axis")

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
        logger.debug("Formatting y-axis")

        if y_name.lower() == "phase":
            yaxis = fig.select(name="p_y_axis")[0]
            yaxis.formatter = PrintfTickFormatter(format=u"%f\u00b0")

    # only to be added in iteration mode
    if add_subtitle:

        logger.debug("Formatting and adding sub-title")
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

        p_txtt_src = ColumnDataSource(
            name="p_sub_data",
            data=dict(x=[x_min + (dw * 0.5)], y=[y_max * 0.87]))

        # Are there associated names? e.g ant / field / corr names
        if i_labels is not None:
            # Only in the case of a baseline
            if {"ANTENNA1", "ANTENNA2"} <= set(i_axis):
                i_axis.sort()

                a1 = i_labels[chunk_attrs.get(i_axis[0])]
                a2 = i_labels[chunk_attrs.get(i_axis[1])]

                p_txtt_src.add([f"{a1}, {a2}"], name="text")

                i_axis_data = f"{ a1 }, { a2 }"

                h_tool.tooltips.append(("Baseline", "@i_axis"))
            else:
                it_val = i_labels[chunk_attrs.get(i_axis[0])]

                p_txtt_src.add([f"{it_val}"], name="text")
                i_axis_data = it_val
                h_tool.tooltips.append(
                    (f"{i_axis[0].capitalize()}", "@i_axis"))

        else:
            # only get the associated ID: it_val:iterator value
            it_val = chunk_attrs[i_axis[0]]

            p_txtt_src.add([f"{i_axis[0].capitalize()}: {it_val}"],
                           name="text")
            i_axis_data = it_val
            h_tool.tooltips.append((f"{i_axis[0].capitalize()}", "@i_axis"))

        if "DATA_DESC_ID" not in i_axis:
            cds.add([chunk_attrs["DATA_DESC_ID"]], "spw")
            h_tool.tooltips.append(("Spw", "@spw"))

        cds.add([i_axis_data], "i_axis")

        # Add some statistics to the iterated plot
        if x_name != "Time":
            h_tool.tooltips.append(
                ("(min_x, max_x)", f"({x_min:.2f}, {x_max:.2f})"))
        h_tool.tooltips.append(
            ("(min_y, max_y)", f"({y_min:.2f}, {y_max:.2f})"))

        fig.add_glyph(p_txtt_src, p_txtt)

    logger.debug("Adding glyph to figure")
    fig.add_glyph(cds, image_glyph)

    return fig


@time_wrapper
def image_callback(xy_df, xr, yr, w, h, x=None, y=None, cat=None):
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=xr, y_range=yr)
    logger.info("Datashader aggregation starting")

    if cat:
        with ProgressBar():
            agg = cvs.points(xy_df, x, y, ds.by(cat, ds.count()))
    else:
        with ProgressBar():
            agg = cvs.points(xy_df, x, y, ds.count())

    logger.info("Aggregation done")

    return agg


def append_cbar(cats, category, cmap, ax, x_min, y_min, labels=None):
    """Add a colourbar for categorical data
    Parameters
    ----------
    cats: :obj:`np.ndarray`
        Available category ids e.g scan_number, field id etc
    category: :obj:`str`
        Name of the categorizing axis
    cmap: :obj:`matplotlib.cmap`
        Matplotlib colormap
    labels: :obj:`list`
        Labels containing names for the iterated stuff
    ax: :obj:`bokeh.models.figure`
        Figure to append the color bar to
    """
    ax.frame_width = int(ax.frame_width * 0.98)

    logger.debug("Adding colour bar")

    cats = cats.tolist()

    category = category.capitalize()

    # format the cateogry name for cbar title
    if "_" in category:
        if category == "Data_desc_id":
            category = "Spw"
        else:
            category = category.split("_")[0]

    if labels is not None:
        labels = np.array(labels)[cats].tolist()
    else:
        labels = [str(_c) for _c in cats]

    rends = []

    # legend height
    lh = int((ax.frame_height / len(cats)) * 0.95)

    for _c in range(len(cats)):
        ssq = ColumnDataSource(data=dict(x=[x_min, x_min], y=[y_min, y_min]))
        sq = Line(x="x", y="y", line_color=next(cmap), line_width=lh)
        ren = ax.add_glyph(ssq, sq)
        rends.append(ren)

    legend = Legend(
        name="p_legend", title_standoff=2, border_line_width=1,
        title=category, glyph_height=lh, glyph_width=25,
        title_text_line_height=0.3, padding=4,
        title_text_font="monospace", title_text_font_style="normal",
        title_text_font_size="10pt", title_text_align="left",
        label_text_font_style="bold", spacing=0, margin=0,
        label_height=5, label_width=10, label_text_font_size="8pt",
        items=[LegendItem(label=lab, renderers=[ren])
               for lab, ren in zip(labels, rends)]
    )

    ax.add_layout(legend, "right")
    logger.debug("Done")


def plotter(x, y, xaxis, xlab='', yaxis="amplitude", ylab='', c_labels=None,
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

    color :  :obj:`str`, :obj:`colormap`, :obj:`itertools.cycle`

    xds_table_obj : :obj:`xarray.Dataset`
        Dataset object containing the columns of the MS. This is passed on in
        case there are items required from the actual dataset.
    ms_name : :obj:`str`
        Name or [can include path] to Measurement Set
    x_min: :obj:`float`
        Minimum x value to be plotted

        Note
        ----
        This may be difficult to achieve in the case where :obj:`xaxis` is
        time because time in ``ragavi-vis`` is converted into milliseconds
        from epoch for ease of plotting by ``bokeh``.
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
    if x_min is None:
        x_min = x.min().data
    if x_max is None:
        x_max = x.max().data

    if y_min is None:
        y_min = y.min().data
    if y_max is None:
        y_max = y.max().data

    logger.info("Calculating x and y Min and Max ranges")
    with ProgressBar():
        x_min, x_max, y_min, y_max = compute(x_min, x_max, y_min, y_max)

    if x_min == x_max:
        logger.debug("Incrementing x-max + 1 because x-min==x-max")
        x_max += 1
    if y_min == y_max:
        logger.debug("Incrementing y-max + 1 one because y-min==y-max")
        y_max += 1

    logger.info(vu.ctext(f"x: {xaxis:.4}, min: {x_min:10.2f}, max: {x_max:10.2f}"))
    logger.info(vu.ctext(f"y: {yaxis:.4}, min: {y_min:10.2f}, max: {y_max:10.2f}"))
    logger.info("Done")

    if np.isnan(y_min) or np.isnan(y_max):
        y_min = x_min = 0
        y_max = x_max = 1

    # inputs to gen image
    im_inputs = dict(c_labels=c_labels, c_height=c_height, c_width=c_width,
                     chunk_no=chunk_no, color=color, i_labels=i_labels,
                     title=title, x=x, x_axis_type=x_axis_type, xlab=xlab,
                     x_name=x.name, y_name=y.name, ylab=ylab,
                     xds_table_obj=xds_table_obj, fix_plotsize=True,
                     add_grid=True)

    if iter_axis and colour_axis:
        # NOTE!!!
        # Iterations are done over daskms groupings except for chan & corr
        # Colouring can therefore be done here by categorization
        title += f""" Iterated By: {iter_axis.capitalize()} Colourised By: {colour_axis.capitalize()}"""

        add_xaxis = add_yaxis = True

        xy_df, cat_values = create_categorical_df(colour_axis, x, y,
                                                  xds_table_obj)

        # generate resulting image
        image = gen_image(xy_df, x_min, x_max, y_min, y_max,
                          cat=colour_axis, ph=plot_height, pw=plot_width,
                          add_cbar=True, add_xaxis=add_xaxis,
                          add_subtitle=True,
                          add_yaxis=add_yaxis, add_title=False, **im_inputs)

    elif iter_axis:
        # NOTE!!!
        # Iterations are done over daskms groupings except for chan & corr
        title += f" Iterated By: {iter_axis.capitalize()}"

        add_xaxis = add_yaxis = True

        xy_df = create_df(x, y, iter_data=None)

        image = gen_image(xy_df, x_min, x_max, y_min, y_max, cat=None,
                          ph=plot_height, pw=plot_width, add_cbar=False,
                          add_xaxis=add_xaxis, add_yaxis=add_yaxis,
                          add_title=False, add_subtitle=True, **im_inputs)

    elif colour_axis:
        title += f" Colourised By: {colour_axis.capitalize()}"

        xy_df, cat_values = create_categorical_df(colour_axis, x, y,
                                                  xds_table_obj)

        # generate resulting image
        image = gen_image(xy_df, x_min, x_max, y_min, y_max, cat=colour_axis,
                          ph=plot_height, pw=plot_width, add_cbar=True,
                          add_xaxis=True, add_yaxis=True, add_title=True,
                          add_subtitle=False, **im_inputs)
        # allow resizing
        image.frame_width = None
        image.frame_height = None
        image.select(name="p_title")[0].text = title

    else:
        xy_df = create_df(x, y, iter_data=None)[[x.name, y.name]]

        # generate resulting image
        im_inputs.update(dict(add_subtitle=False, add_title=True))
        image = gen_image(xy_df, x_min, x_max, y_min, y_max, cat=None,
                          ph=plot_height, pw=plot_width, add_cbar=False,
                          add_xaxis=True, add_yaxis=True, **im_inputs)
        # allow resizing
        image.frame_width = None
        image.frame_height = None

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
    logger.debug("Creating antenna iterable")

    taql_where = kwargs.get("taql_where", "")
    table_schema = kwargs.get("table_schema", None)
    chunks = kwargs.get("chunks", 5000)

    # Shall be prepended to the later antenna selection
    if taql_where:
        taql_where += " && "

    outp = []
    # ant_names = vu.get_antennas(ms_name).values.tolist()
    n_ants = vu.get_antennas(ms_name).size
    n_spws = vu.get_frequencies(ms_name).row.size

    for d in range(n_spws):
        for a in range(n_ants):

            logger.debug(f"Spw: {d}, antenna: {a}")

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

    logger.debug("Done")
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

    logger.debug("Creating correlation iterable")

    outp = []
    n_corrs = subs[0].corr.size

    for sub in subs:
        for c in range(n_corrs):

            logger.debug(f"Corr: {c}")

            nsub = sub.copy(deep=True).sel(corr=c)
            nsub.attrs["Corr"] = c
            outp.append(nsub)

    logger.debug("Done")
    return outp


def get_ms(ms_name,  ants=None, cbin=None, chan_select=None, chunks=None,
           corr_select=None, colour_axis=None, data_col="DATA", ddid=None,
           fid=None, iter_axis=None, scan=None, tbin=None,  where=None,
           x_axis=None):
    """Get xarray Dataset objects containing Measurement Set columns of the
    selected data

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
        A list containing the specified table objects as
        :obj:`xarray.Dataset`
    """
    logger.debug("Starting MS acquisition")

    group_cols = set()

    # Always group by DDID
    group_cols.update({"DATA_DESC_ID"})

    sel_cols = {"FLAG", "FLAG_ROW"}

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
        chunks = dict(row=5000)
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

    logger.debug(f"Selected columns: {', '.join(sel_cols)}")
    logger.debug(f"Grouping by: {', '.join(group_cols)}")
    logger.debug(f"TAQL selection: {where}")

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
                logger.debug("Selecting channels")
                tab_objs = [_.sel(chan=chan_select) for _ in tab_objs]
            # select corrs
            if corr_select is not None:
                logger.debug("Selecting correlations")
                tab_objs = [_.sel(corr=corr_select) for _ in tab_objs]

        if len(tab_objs) == 0:
            logger.error("No Data found in the MS")
            logger.error("Please check your selection criteria")
            sys.exit(-1)

        # get some info about the data
        chunk_sizes = list(tab_objs[0].FLAG.data.chunksize)
        chunk_sizes = " * ".join([str(_) for _ in chunk_sizes])
        chunk_p = tab_objs[0].FLAG.data.npartitions

        logger.info(f"Chunk size: {chunk_sizes}")
        logger.info("Number of Partitions: {}".format(chunk_p))

        return tab_objs
    except Exception as ex:
        logger.error("MS data acquisition failed")
        for _ in ex.args[0].split("\n"):
            logger.error(_)
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

    u_ants = np.unique(np.hstack((ant1, ant2)))

    uniques = list(combinations(u_ants, 2))

    u_bls_combos = []
    for _p in np.unique(ant1):
        for p, q in uniques:
            if _p == p:
                u_bls_combos.append((p, q))
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


def create_dask_df(inp, idx):
    """
    Parameters
    ----------
    inp: :obj:`dict`
        A dictionary containing column name as the key, and the dictionary
        value is the dask array to be associated with the column name.
    ids: :obj:`da.array`
        A dask array to form the index of the resulting dask dataframe

    Returns
    -------
    ddf: :obj:`dd.Dataframe`
        Dask dataframe containing the presented data
    """
    series_list = []

    idx = dd.from_dask_array(idx, columns=["Index"])

    for col, darr in inp.items():
        series = dd.from_dask_array(darr, columns=[col])
        series_list.append(series)

    series_list.append(idx)

    ddf = dd.concat(series_list, axis=1)

    # logger.info("Indexing Dataframe")

    # Will be activated when indexing is important
    # with ProgressBar():
    #     ddf = ddf.set_index("Index", sorted=True)

    # logger.info("Index ready")

    return ddf


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

    logger.info("Starting creation of categorical dataframe")

    xy_df = create_df(x_data, y_data, iter_data=iter_data)[[x_data.name,
                                                            y_data.name,
                                                            it_axis]]
    cat_values = np.unique(iter_data.values)

    xy_df = xy_df.astype({it_axis: "category"})
    xy_df[it_axis] = xy_df[it_axis].cat.as_known()

    return xy_df, cat_values


def create_df(x, y, iter_data=None):
    """
    Create a dask dataframe from input x, y and iterate columns if
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
    var_names = {x_name: nx,
                 y_name: ny}

    if iter_data is not None:
        i_name = iter_data.name
        iter_data = massage_data(iter_data, y, get_y=False, iter_ax=i_name)
        var_names[i_name] = iter_data

    logger.info("Creating dataframe")

    # manuanally create a dask array index
    idx = da.arange(nx.size, chunks=nx.chunksize[0])

    new_ds = create_dask_df(var_names, idx)

    new_ds = new_ds.dropna()

    logger.info("DataFrame ready")
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
    logger.debug("Flattening x-axis and y-axis data")

    iter_cols = ["ANTENNA1", "ANTENNA2", "FIELD_ID", "SCAN_NUMBER",
                 "DATA_DESC_ID", "Baseline"]
    # can add 'chan','corr' to this list

    # available dims in the x and y axes
    y_dims = set(y.dims)
    x_dims = set(x.dims)

    logger.debug(f"x shape: {str(x.shape)}")
    logger.debug(f"y shape: {str(y.shape)}")

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

    logger.debug(f"New x shape: {str(nx.shape)}")

    if get_y:
        # flatten y data
        # get_data also
        ny = y.data.ravel()
        logger.debug(f"New y shape: {str(ny.shape)}")
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
    if inp is not None:
        col_name = aliases[inp]
    else:
        col_name = None

    logger.debug(f"Column name for {inp} --> {col_name}")
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
    oup: :obj:`str`
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
        oup = alts[inp]
    else:
        oup = inp

    logger.debug(f"Alias for axis {inp} --> {oup}")
    return oup


######################### Random functions #############################
def link_grid_plots(plot_list, ncols, nrows):
    """Link all the plots in the X and Y axes

    """

    logger.debug(f"Linking {len(plot_list)} generated plots")

    if not isinstance(plot_list[0], Plot):
        plots = []
        plot_list = plot_list[0].children
        for item in plot_list:
            plots.append(item[0])
    else:
        plots = plot_list

    maxxs, maxys, minxs, minys = [], [], [], []

    # get the global maxs and mins
    for _p in plots:
        # See how image glyph is specified for justification
        im_data = _p.select(name="image_data")[0].data

        if np.max(im_data["image"]) > 0:
            maxxs.append(im_data["x"][0] + im_data["dw"][0])
            maxys.append(im_data["y"][0] + im_data["dh"][0])
            minxs.append(im_data["x"][0])
            minys.append(im_data["y"][0])

    minx = np.min(minxs)
    miny = np.min(minys)
    maxx = np.max(maxxs)
    maxy = np.max(maxys)

    # initialise the shared data ranges
    init_xr = DataRange1d(start=minx, end=maxx, only_visible=True)
    init_yr = DataRange1d(start=miny, end=maxy, only_visible=True)

    for _i, _p in enumerate(plots):

        # make plots share their ranges
        _p.x_range = init_xr
        _p.y_range = init_yr

        # move plot subtitle depending on universal min/max
        im_data = _p.select(name="image_data")[0].data
        p_sub_data = _p.select(name="p_sub_data")[0]

        if (_i < ncols * (nrows - 1)):
            _p.xaxis.visible = False

        # Add y-axis to all items in the first columns
        if (_i % ncols != 0):
            _p.yaxis.visible = False

        if _p.legend:
            p_legend = _p.select(name="p_legend")[0]
            # legends to be on the last col items
            if _i % ncols != ncols - 1:
                p_legend.visible = False

        if np.max(im_data["image"]) < 1:
            # This means the image is empty
            p_sub_data.data.update(x=[np.mean([minx, maxx])],
                                   y=[np.mean([miny, maxy])])
        else:
            p_sub_data.data.update(x=[np.mean([minx, maxx])],
                                   y=[0.85 * maxy])

    logger.debug("Plots linked")


def mod_unlinked_grid_plots(plot_list, nrows, ncols):
    """
    * Move tick marks into the plots
    * Reduce frame size of plots
    * Switch on only axis label for first item in row
    """

    for _i, _p in enumerate(plot_list):
        # move tick marks into plot
        _p.xaxis.major_label_standoff = 2
        _p.xaxis.major_tick_out = 0
        _p.xaxis.major_tick_in = 4

        _p.yaxis.major_label_standoff = 2
        _p.yaxis.major_tick_out = 0
        _p.yaxis.major_tick_in = 4

        if _p.legend:
            p_legend = _p.select(name="p_legend")[0]
            # legends to be on the last col items
            if _i % ncols != ncols - 1:
                p_legend.visible = False

        if _i % ncols != 0:
            _p.yaxis.axis_label = ""

        if (_i < ncols * (nrows - 1)):
            _p.xaxis.axis_label = ""

        pw = int((PLOT_WIDTH * 0.99) / ncols)
        ph = int(0.85 * pw)

        if (nrows * ph) < (PLOT_HEIGHT * 0.715):
            ph = int((PLOT_HEIGHT * 0.715) / nrows)

        _p.plot_width = pw
        _p.plot_height = ph

        _p.frame_width = int(pw * 0.80)
        _p.frame_height = int(ph * 0.80)
        _p.sizing_mode = "fixed"

        im_data = _p.select(name="image_data")[0].data
        p_sub_data = _p.select(name="p_sub_data")[0]

        if np.max(im_data["image"]) < 1:
            # This means the image is empty
            p_sub_data.data.update(x=[0.5],
                                   y=[0.5])


def resource_defaults():
    # Value of 1GB
    _GB_ = 2**30

    # setting memory limit in GB
    ml = 1

    # get size of 90% of the RAM available in GB
    mems = virtual_memory()

    logger.info(f"Total RAM size: ~{(mems.total / _GB_):.2f} GB")
    total_mem = int((mems.total * 0.9) / _GB_)

    # set cores to half the amount available
    cores = cpu_count()
    logger.info(f"Total number of Cores: {cores}")
    cores = cores / 2

    if cores > 10:
        cores = 10

    # Because memory is assigned per core
    if (cores * ml) >= total_mem:
        cores = int(total_mem // ml)

    ml = f"{ml}GB"
    return int(cores), ml


############################## Main Function ###########################
@time_wrapper
def main(**kwargs):
    """Main function that launches the visibilities plotter"""

    if "options" in kwargs:
        options = kwargs.get("options", None)
        ants = options.ants
        chan = options.chan
        chunks = options.chunks
        c_width = options.c_width
        c_height = options.c_height
        corr = options.corr
        data_column = options.data_column
        ddid = options.ddid
        fields = options.fields
        flag = options.flag
        html_name = options.html_name
        mycmap = options.mycmap
        mytabs = options.mytabs
        scan = options.scan
        where = options.where  # taql where
        xmin = options.xmin
        xmax = options.xmax
        ymin = options.ymin
        ymax = options.ymax
        tbin = options.tbin
        cbin = options.cbin

    # Set the debug level
    if options.debug:
        vu.update_log_levels(logger, "debug")
    else:
        vu.update_log_levels(logger, "info")

    # Default logfile name is ragavi.log
    if options.logfile:
        vu.update_logfile_name(logger, options.logfile)
    else:
        vu.update_logfile_name(logger, "ragavi.log")

    # set defaults unless otherwise
    n_cores, mem_limit = resource_defaults()

    if options.n_cores:
        n_cores = int(options.n_cores)

    if options.mem_limit:
        mem_limit = options.mem_limit

    if options.n_cols:
        n_cols = options.n_cols
    else:
        n_cols = 5

    # change iteration axis names to actual column names
    colour_axis = validate_axis_inputs(options.colour_axis)
    colour_axis = get_colname(colour_axis)

    iter_axis = validate_axis_inputs(options.iter_axis)
    iter_axis = get_colname(iter_axis)

    # validate the x and y axis names
    xaxis = validate_axis_inputs(options.xaxis)
    yaxis = validate_axis_inputs(options.yaxis)

    # number of processes or threads/ cores
    dask_config.set(num_workers=n_cores, memory_limit=mem_limit)

    logger.info(f"Using {n_cores} cores")
    logger.info(f"Memory limit per core: {mem_limit}")

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

        if options.corr:
            if options.corr in ["diag", "diagonal"]:
                corr = "xx,yy"
            elif options.corr in ["off-diag", "off-diagonal"]:
                corr = "xy,yx"

            corr_labs = vu.get_polarizations(mytab)
            corr = corr.upper().split(",")

            logger.info(f"Available corrs: {','.join(corr_labs)}")

            for _ci, _corr in enumerate(corr):
                if _corr.isalpha():
                    try:
                        corr[_ci] = corr_labs.index(_corr)
                    except ValueError:
                        logger.warning(f"Chosen corr {_corr} not available")
                        corr[_ci] = -1
                else:
                    corr[_ci] = int(_corr)

            corr = [str(_c) for _c in corr if _c != -1]

            if len(corr) == 0:
                logger.error(
                    f"Selected corrs: {options.corr} unavailable. Exiting.")
                sys.exit(-1)
            else:
                corr = ",".join(corr)

            corr = vu.slice_data(corr)

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

        if options.scan is not None:
            scan = vu.resolve_ranges(scan)
        if options.ants is not None:
            ants = vu.resolve_ranges(ants)

        if options.ddid is not None:
            n_ddid = vu.slice_data(ddid)
            ddid = vu.resolve_ranges(ddid)
        else:
            # set data selection to all unless ddid is specified
            n_ddid = slice(0, None)

        # capture channels or correlations to be selected
        chan = vu.slice_data(chan)

        ################################################################
        ################### Iterate over subtables #####################

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

            # set the colouring axis labels
            if colour_axis == "corr":
                c_labels = vu.get_polarizations(mytab)
            elif colour_axis == "FIELD_ID":
                c_labels = vu.get_fields(mytab).values.tolist()
            elif colour_axis in ["antenna", "ANTENNA1", "ANTENNA2",
                                 "Baseline"]:
                c_labels = vu.get_antennas(mytab).values.tolist()
            else:
                c_labels = None
        else:
            c_labels = None
            if mycmap:
                mycmap = vu.get_cmap(mycmap, fall_back="blues",
                                     src="colorcet")
            else:
                mycmap = vu.get_cmap("blues", src="colorcet")

        # configure number of rows and columns for grid
        n_partitions = len(partitions)
        n_rows = int(np.ceil(n_partitions / n_cols))

        if iter_axis:
            pw = int((PLOT_WIDTH * 0.961) / n_cols)
            ph = int(0.83 * pw)

            # if there are more columns than there are items
            if n_cols > n_partitions:
                pw = int((PLOT_WIDTH * 0.961) / n_partitions)
                ph = int(PLOT_HEIGHT * 0.755)
                # because the grid is not generated by dask-ms iteration
                n_cols = n_partitions

            # If there is still space extend height
            if (n_rows * ph) < (PLOT_HEIGHT * 0.715):
                ph = int((PLOT_HEIGHT * 0.715) / n_rows)

            if not c_width:
                c_width = 200
                logger.info(
                    f"""Shrinking canvas width {c_width} for iteration""")
            if not c_height:
                c_height = 200
                logger.info(f"Shrinking canvas height to {c_height} for iteration")

            if iter_axis == "corr":
                i_labels = vu.get_polarizations(mytab)
            elif iter_axis == "FIELD_ID":
                i_labels = vu.get_fields(mytab).values.tolist()
            elif iter_axis in ["antenna", "ANTENNA1", "ANTENNA2", "Baseline"]:
                i_labels = vu.get_antennas(mytab).values.tolist()
            else:
                i_labels = None
        else:
            pw = int(PLOT_WIDTH * 0.95)
            ph = int(PLOT_HEIGHT * 0.84)

            i_labels = None
            if not c_width:
                c_width = 1080
            if not c_height:
                c_height = 720

        oup_a = []

        # iterating over spectral windows (spws)
        for count, chunk in enumerate(partitions):

            # We check current DATA_DESC_ID
            c_spw = chunk.DATA_DESC_ID

            if isinstance(c_spw, xr.DataArray):
                c_spw = c_spw.values[0]

            logger.info("Starting data processing.")

            p_data = DataCoreProcessor(chunk, mytab, xaxis, yaxis, chan=chan,
                                       corr=corr, flag=flag, ddid=c_spw,
                                       datacol=data_column, cbin=cbin)
            ready = p_data.act()

            logger.info(vu.ctext(f"Plotting {count+1}/{n_partitions}"))
            logger.info(f"{chunk.attrs}\033[0m")

            # black box returns an plottable element / composite element
            # as well a the title to be used for the plot
            fig, title_txt = plotter(
                x=ready.x, xaxis=xaxis, xlab=ready.xlabel,
                y=ready.y, yaxis=yaxis, ylab=ready.ylabel,
                c_height=c_height, c_width=c_width, chunk_no=count,
                color=mycmap, colour_axis=colour_axis, c_labels=c_labels,
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

            if options.link_plots:
                # Link all the plots
                link_grid_plots(oup_a, ncols=n_cols, nrows=n_rows)
            else:
                mod_unlinked_grid_plots(oup_a, ncols=n_cols, nrows=n_rows)

            info_text = f"ragavi   : v{__version__}\n"
            info_text += f"MS       : {mytab}\n"
            info_text += f"Grid size: {n_cols} x {n_rows}"
            info_text += f" | Linked: {options.link_plots}"

            pre = PreText(text=info_text, width=int(PLOT_WIDTH * 0.961),
                          height=50, align="start", margin=(0, 0, 0, 0))
            final_plot = gridplot(children=oup_a, ncols=n_cols,
                                  sizing_mode="stretch_width")
            final_plot = column(children=[title_div, pre, final_plot])

        else:
            if len(oup_a) > 1:
                final_plot = gridplot(
                    children=oup_a,
                    sizing_mode="stretch_width", ncols=2,
                    plot_width=PLOT_WIDTH // len(oup_a),
                    plot_height=PLOT_HEIGHT // len(oup_a))
            else:
                final_plot = oup_a[0]

            info_text = f"MS: {mytab}"
            pre = PreText(text=info_text, width=int(PLOT_WIDTH * 0.961),
                          height=50, align="start", margin=(0, 0, 0, 0))
            final_plot = column([pre, final_plot],
                                sizing_mode="stretch_width")

        logger.info("Plotting complete.")

        if html_name:
            if "html" not in html_name:
                html_name += ".html"
            fname = html_name
        else:
            # Naming format: ms_name-yaxis_vs_xaxis-colour_axis-iter_axis-dcol
            fname = f"""{os.path.basename(mytab)}_{yaxis:.3}_vs_{xaxis:.3}_color_{options.colour_axis}_iterate_{options.iter_axis}_{options.data_column}_flagged_{options.flag}.html"""

        output_file(fname, title=fname)
        save(final_plot)

        logger.info("Rendered plot to: {}".format(fname))
        logger.info("Specified options:")
        parsed_opts = {k: v for k, v in options.__dict__.items()
                       if v is not None}
        parsed_opts = json.dumps(parsed_opts, indent=2, sort_keys=True)
        for _x in parsed_opts.split('\n'):
            logger.info(_x)
        logger.info(">" * 70)
