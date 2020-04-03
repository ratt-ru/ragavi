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
                          DatetimeTickFormatter, Div, ImageRGBA, LinearAxis,
                          LinearColorMapper, PrintfTickFormatter, Plot,
                          Range1d, Text, Title)
from bokeh.models.tools import (BoxZoomTool, HoverTool, ResetTool, PanTool,
                                WheelZoomTool, SaveTool)

from ragavi import utils as vu

from dask.distributed import Client, LocalCluster

logger = vu.logger
excepthook = vu.sys.excepthook
time_wrapper = vu.time_wrapper
wrapper = vu.textwrap.TextWrapper(initial_indent='',
                                  break_long_words=True,
                                  subsequent_indent=''.rjust(50),
                                  width=160)

PLOT_WIDTH = int(1920 * 0.95)
PLOT_HEIGHT = int(1080 * 0.9)


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
                 ddid=int, datacol='DATA', flag=True):

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

    def process_data(self, ydata, yaxis, wrap=True):
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
        if yaxis == 'amplitude':
            y = vu.calc_amplitude(ydata)
        elif yaxis == 'imaginary':
            y = vu.calc_imaginary(ydata)
        elif yaxis == 'phase':
            y = vu.calc_phase(ydata, wrap=wrap)
        elif yaxis == 'real':
            y = vu.calc_real(ydata)
        return y

    def get_xaxis_data(self, xds_table_obj, ms_name, xaxis, datacol='DATA', ddid=0, cbin=None, chan=None):
        """Get x-axis data. This function also returns the relevant x-axis label.

        Parameters
        ----------
        ms_name : :obj:`str`
            Name of Measurement Set
        xaxis : :obj:`str`
            Name of xaxis
        xds_table_obj: :obj:`xarray.Dataset`
            MS as xarray dataset from xarrayms

        Returns
        -------
        xdata : :obj:`xarray.DataArray`
            X-axis data depending  x-axis selected.
        x_label : :obj:`str`
            Label to appear on the x-axis of the plots.
        """

        if xaxis == 'antenna1':
            xdata = xds_table_obj.ANTENNA1
            x_label = 'Antenna1'
        elif xaxis == 'antenna2':
            xdata = xds_table_obj.ANTENNA2
            x_label = 'Antenna2'
        elif xaxis == 'frequency' or xaxis == 'channel':
            xdata = vu.get_frequencies(ms_name, spwid=ddid, chan=chan,
                                       cbin=cbin)
            x_label = 'Frequency GHz'
        elif xaxis == 'phase':
            xdata = xds_table_obj[datacol]
            x_label = 'Phase [deg]'
        elif xaxis == 'real':
            xdata = xds_table_obj[datacol]
            x_label = 'Real'
        elif xaxis == 'scan':
            xdata = xds_table_obj.SCAN_NUMBER
            x_label = 'Scan'
        elif xaxis == 'time':
            xdata = xds_table_obj.TIME
            x_label = 'Time [s]'
        elif xaxis == 'uvdistance':
            xdata = xds_table_obj.UVW
            x_label = 'UV Distance [m]'
        elif xaxis == 'uvwave':
            xdata = xds_table_obj.UVW
            try:
                x_label = 'UV Wave [{}]'.format(u'\u03bb')
            except UnicodeEncodeError:
                x_label = 'UV Wave [{}]'.format(u'\u03bb'.encode('utf-8'))
        else:
            logger.error("Invalid xaxis name")
            return

        return xdata, x_label

    def get_yaxis_data(self, xds_table_obj, ms_name, yaxis, datacol='DATA'):
        """Extract the required column for the y-axis data.

        Parameters
        ----------
        datacol: :obj:`str`
            Data column to be selected. Defaults to DATA
        ms_name: :obj:`str`
            Name of measurement set.
        xds_table_obj: :obj:`xarray.Dataset`
            MS as xarray dataset from xarrayms
        yaxis: :obj:`str`
            yaxis to plot.

        Returns
        -------
        ydata: :obj:`xarray.DataArray`
            y-axis data depending  y-axis selected.
        y_label: :obj:`str`
            Label to appear on the y-axis of the plots.
        """
        if yaxis == 'amplitude':
            y_label = 'Amplitude'
        elif yaxis == 'imaginary':
            y_label = 'Imaginary'
        elif yaxis == 'phase':
            y_label = 'Phase [deg]'
        elif yaxis == 'real':
            y_label = 'Real'

        try:
            ydata = xds_table_obj[datacol]
        except KeyError:
            logger.exception('Column "{}" not Found'.format(datacol))
            return sys.exit(-1)

        return ydata, y_label

    def prep_xaxis_data(self, xdata, freq=None,
                        xaxis='time'):
        """Prepare the x-axis data for plotting. Selections also performed here.

        Parameters
        ----------
        xdata: :obj:`xarray.DataArray`
            x-axis data depending  x-axis selected
        xaxis: :obj:`str`
            xaxis to plot

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
        if xaxis == 'channel' or xaxis == 'frequency':
            prepdx = xdata / 1e9
        elif xaxis == 'phase':
            prepdx = xdata
        elif xaxis == 'time':
            prepdx = vu.time_convert(xdata)
        elif xaxis == 'uvdistance':
            prepdx = vu.calc_uvdist(xdata)
        elif xaxis == 'uvwave':
            prepdx = vu.calc_uvwave(xdata, freq)
        elif xaxis == 'antenna1' or xaxis == 'antenna2' or xaxis == 'scan':
            prepdx = xdata
        return prepdx

    def prep_yaxis_data(self, xds_table_obj, ms_name, ydata,
                        corr=None, flag=True, yaxis='amplitude'):
        """Process data for the y-axis.
        This includes:

            * Correlation selection
            * Channel selection is now done as during MS acquisition
            * Flagging
            * Conversion form complex to the required form

        Data selection and flagging are done by this function itself, however ap and ri conversion are handled by :meth:`ragavi.ragavi.DataCoreProcessor.compute_ydata`

        Parameters
        ----------
        corr: :obj:`int`
            Correlation number to select
        flag: :obj:`bool`
            Option on whether to flag the data or not. Defaults to True
        ms_name: :obj:`str`
            Name of measurement set.
        xds_table_obj: :obj:`xarray.Dataset`
            MS as xarray dataset from xarrayms
        yaxis: :obj:`str`
            selected y-axis
        ydata: :obj:`xarray.DataArray`
            y-axis data to be processed

        Returns
        -------
        y: :obj:`xarray.DataArray`
           Processed :attr:`ydata` data.
        """
        flags = vu.get_flags(xds_table_obj)
        if corr is not None:
            ydata = ydata.sel(dict(corr=corr))
            flags = flags.sel(dict(corr=corr))

        # if flagging enabled return a list of DataArrays otherwise return a
        # single dataarray
        if flag:
            processed = self.process_data(ydata, yaxis=yaxis, wrap=True)
            y = processed.where(flags == False)
        else:
            y = self.process_data(ydata, yaxis=yaxis)

        return y

    def blackbox(self, xds_table_obj, ms_name, xaxis, yaxis, cbin=None,
                 chan=None, corr=None, datacol='DATA', ddid=None,
                 flag=True):
        """Get raw input data and churn out processed data.

        This function incorporates all function in the class to get the desired result. Takes in all inputs from the instance initialising object. It performs:

            - xaxis data and error data acquisition
            - xaxis data and error preparation and processing
            - yaxis data and error data acquisition
            - yaxis data and error preparation and processing

        Returns
        -------
        d : :obj:`collections.namedtuple`
            A named tuple containing all processed x-axis data, errors and label, as well as both pairs of y-axis data, their error margins and labels. Items from this tuple can be gotten by using the dot notation.
        """

        Data = namedtuple('Data', "x xlabel y ylabel")

        xs = self.x_only(xds_table_obj, ms_name, xaxis, corr=corr, chan=chan,
                         ddid=ddid, flag=flag, datacol=datacol, cbin=cbin)

        ys = self.y_only(xds_table_obj, ms_name, yaxis, corr=corr, flag=flag,
                         datacol=datacol)

        x_prepd = xs.x
        xlabel = xs.xlabel
        y_prepd = ys.y
        ylabel = ys.ylabel

        if xaxis == 'channel' or xaxis == 'frequency':
            if y_prepd.ndim == 3:
                y_prepd = y_prepd.transpose('chan', 'row', 'corr')
            else:
                y_prepd = y_prepd.T

            # remove the row dim because of compatibility with merge issues
            x_prepd = x_prepd.sel(row=0)

        d = Data(x=x_prepd, xlabel=xlabel, y=y_prepd, ylabel=ylabel)

        return d

    def act(self):
        """Activate the :meth:`ragavi.ragavi.DataCoreProcessor.blackbox`
        """
        return self.blackbox(self.xds_table_obj, self.ms_name, self.xaxis,
                             self.yaxis, chan=self.chan, corr=self.corr,
                             datacol=self.datacol, ddid=self.ddid,
                             flag=self.flag, cbin=self.cbin)

    def x_only(self, xds_table_obj, ms_name, xaxis, flag=True, corr=None,
               chan=None, ddid=0, datacol='DATA', cbin=None):
        """Return only x-axis data and label

        Returns
        -------
        d : :obj:`collections.namedtuple`
            Named tuple containing x-axis data and x-axis label. Items in the tuple can be accessed by using the dot notation.
        """
        Data = namedtuple('Data', 'x xlabel')

        x_data, xlabel = self.get_xaxis_data(xds_table_obj, ms_name, xaxis,
                                             datacol=datacol, chan=chan,
                                             cbin=cbin)
        if xaxis == 'uvwave':
            # compute uvwave using the available selected frequencies
            freqs = vu.get_frequencies(ms_name, spwid=ddid,
                                       chan=chan, cbin=cbin).compute()

            x = self.prep_xaxis_data(x_data, xaxis=xaxis, freq=freqs)

        # if we have x-axes corresponding to ydata
        elif xaxis == 'real' or xaxis == 'phase':
            x = self.prep_yaxis_data(xds_table_obj, ms_name, x_data,
                                     yaxis=xaxis, corr=corr, flag=flag)
        else:
            x = self.prep_xaxis_data(x_data, xaxis=xaxis)

        d = Data(x=x, xlabel=xlabel)

        return d

    def y_only(self, xds_table_obj, ms_name, yaxis, corr=None, datacol='DATA', flag=True):
        """Return only y-axis data and label

        Returns
        -------
        d : :obj:`collections.namedtuple`
            Named tuple containing x-axis data and x-axis label. Items in the tuple can be accessed by using the dot notation.
        """

        Data = namedtuple('Data', 'y ylabel')
        y_data, ylabel = self.get_yaxis_data(xds_table_obj, ms_name, yaxis,
                                             datacol=datacol)

        y = self.prep_yaxis_data(xds_table_obj, ms_name, y_data, yaxis=yaxis,
                                 corr=corr, flag=flag)

        d = Data(y=y, ylabel=ylabel)

        return d


def add_axis(fig, axis_range, ax_label):
    """Add an extra axis to the current figure

    Parameters
    ----------
    fig: :obj:`bokeh.plotting.figure`
        The figure onto which to add extra axis

    axis_range: :obj:`list`, :obj:`tuple`
        A range of sorted values or a tuple with 2 values containing or the
        order (min, max). This can be any ordered iteraable that can be
        indexed.

    Returns
    -------
    fig : :obj:`bokeh.plotting.figure`
        Bokeh figure with an extra axis added
    """
    if fig.extra_x_ranges is None:
        fig.extra_x_ranges = {}
    fig.extra_x_ranges["fxtra"] = Range1d(start=axis_range[0],
                                          end=axis_range[-1])
    linaxis = LinearAxis(x_range_name="fxtra", axis_label=ax_label,
                         minor_tick_line_alpha=0, name="chan_xtra_linaxis",
                         major_label_orientation='horizontal',
                         ticker=BasicTicker(desired_num_ticks=30))
    fig.add_layout(linaxis, 'above')
    return fig


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
                    title_text_font_size='10pt', title_text_align="left",
                    title_text_font="monospace", minor_tick_line_width=0,
                    title_text_font_style="normal")

    return cbar


def create_bk_fig(x_range, y_range, x=None, xlab=None, ylab=None,
                  title=None, pw=1920, ph=1080, x_axis_type='linear',
                  y_axis_type='linear', x_name=None, y_name=None):

    fig = figure(tools='pan,box_zoom,wheel_zoom,reset,save',
                 x_range=tuple(x_range), x_axis_label=xlab, y_axis_label=ylab,
                 y_range=tuple(y_range), x_axis_type=x_axis_type,
                 y_axis_type=y_axis_type, title=title,
                 plot_width=pw, plot_height=ph,
                 sizing_mode='stretch_both')

    h_tool = HoverTool(tooltips=[(x_name, '$x'),
                                 (y_name, '$y')],
                       point_policy='snap_to_data')
    if x_axis_type == "datetime":
        h_tool.formatters["$x"] = x_axis_type
    fig.add_tools(h_tool)

    x_axis = fig.xaxis[0]
    x_axis.major_label_orientation = 45
    x_axis.ticker.desired_num_ticks = 30

    if x_name.lower() in ['channel', 'frequency']:
        fig = add_axis(fig=fig,
                       axis_range=x.chan.values,
                       ax_label='Channel')
    if y_name.lower() == 'phase':
        fig.yaxis[0].formatter = PrintfTickFormatter(format=u"%f\u00b0")

    fig.axis.axis_label_text_font_style = "normal"
    fig.axis.axis_label_text_font = "monospace"
    fig.axis.axis_label_text_font_size = "15px"
    fig.title.text = title
    fig.title.text_font = "monospace"
    fig.title.text_font_size = "24px"
    fig.title.align = 'center'

    return fig


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

    u_ants = da.unique(ant1.data).compute()
    u_bls_combos = combinations(np.arange(u_ants.size + 1), 2)

    if bl_combos:
        return u_bls_combos

    logger.info("Populating baseline data")

    # create a baseline array of the same shape as antenna1
    baseline = np.empty_like(ant1.values)
    for bl, a in enumerate(u_bls_combos):
        a1 = a[0]
        a2 = a[-1]
        baseline[(ant2.values == a2) & (ant1.values == a1)] = bl

    baseline = da.asarray(a=baseline).rechunk(ant1.data.chunksize)
    baseline = ant1.copy(deep=True, data=baseline)
    baseline.name = 'Baseline'
    return baseline


def average_ms(ms_name, tbin=None, cbin=None, chunk_size=None, taql='',
               columns=None, chan=None):
    """ Perform MS averaging
    Parameters
    ----------
    ms_name : :obj:`str`
        Name of the input MS
    tbin : :obj:`float`
        Time bin in seconds
    cbin : :obj:`int`
        Number of channels to bin together
    chunk_size : :obj:`dict` 
        Size of resulting MS chunks. 
    taql: :obj:`str`
        TAQL clause to pass to xarrayms

    Returns
    -------
    x_dataset: :obj:`list`
        List of :obj:`xarray.Dataset` containing averaged MS. The MSs are split by Spectral windows

    """
    from xova.apps.xova import averaging as av

    if chunk_size is None:
        chunk_size = dict(row=100000)
    if tbin is None:
        tbin = 1
    if cbin is None:
        cbin = 1

    # must be grouped this way because of time averaging
    ms_obj = xm.xds_from_ms(ms_name, group_cols=['DATA_DESC_ID',
                                                 'FIELD_ID',
                                                 'SCAN_NUMBER'],
                            taql_where=taql)

    # some channels have been selected
    if chan is not None:
        ms_obj = [_.sel(chan=chan) for _ in ms_obj]

    # unique ddids available
    unique_spws = np.unique([ds.DATA_DESC_ID for ds in ms_obj])

    logger.info("Averaging MAIN table")

    # perform averaging to the MS
    avg_mss = av.average_main(ms_obj, tbin, cbin, 100000, False)

    x_datasets = []
    spw_ds = []

    logger.info("Creating averaged xarray Dataset")

    for ams in avg_mss:
        data_vars = OrderedDict()
        coords = OrderedDict()

        for k, v in sorted(ams.data_vars.items()):
            if k in columns:
                data_vars[k] = xr.DataArray(v,
                                            dims=v.dims, attrs=v.attrs)
        """
        for k, v in sorted(ams.coords.items()):
            if k in columns:
                coords[k] = xr.DataArray(v,
                                         dims=v.dims, attrs=v.attrs)
        """

        # create datasets and rename dimension "[uvw]"" to "uvw"
        x_datasets.append(xr.Dataset(data_vars,
                                     attrs=dict(ams.attrs),
                                     coords=coords).rename_dims({"[uvw]":
                                                                 "uvw"}))

    # Separate datasets in different SPWs
    for di in unique_spws:
        spw_ds.append(
            [x for x in x_datasets if x.DATA_DESC_ID[0] == di])

    x_datasets = []
    # produce multiple datasets incase of multiple SPWs
    # previous step results in  list of the form [[spw1_data], [spw2_data]]
    # for each spw
    for spw in spw_ds:
        # Compact averaged MS datasets into a single dataset for ragavi
        ms_obj = xr.combine_nested(spw, concat_dim='row',
                                   compat='no_conflicts', data_vars='all',
                                   coords='different', join='outer')
        # chunk the data as user wanted it
        ms_obj = ms_obj.chunk(chunk_size)
        x_datasets.append(ms_obj)

    logger.info("Averaging completed.")

    return x_datasets


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

    iter_cols = ['ANTENNA1', 'ANTENNA2',
                 'FIELD_ID', 'SCAN_NUMBER', 'DATA_DESC_ID', 'Baseline']
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
    var_names = {x_name: ('row', nx),
                 y_name: ('row', ny)}

    if iter_data is not None:
        i_name = iter_data.name
        iter_data = massage_data(iter_data, y, get_y=False, iter_ax=i_name)
        var_names[i_name] = ('row', iter_data)

    new_ds = xr.Dataset(data_vars=var_names)
    new_ds = new_ds.to_dask_dataframe()
    new_ds = new_ds.dropna()
    return new_ds


@time_wrapper
def image_callback(xy_df, xr, yr, w, h, x=None, y=None, cat=None):
    cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=xr, y_range=yr)
    logger.info("Datashader aggregation starting")

    if cat:
        with ProgressBar():
            agg = cvs.points(xy_df, x, y, ds.count_cat(cat))
    else:
        with ProgressBar():
            agg = cvs.points(xy_df, x, y, ds.any())

    logger.info("Aggregation done")

    return agg


def gen_image(df, x_min, x_max, y_min, y_max, c_width, c_height, x_name=None,
              y_name=None, xlab=None, ylab=None, title=None, x=None,
              x_axis_type="linear", pw=PLOT_WIDTH, ph=PLOT_HEIGHT,
              cat=None, color=None):

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
    fig = create_bk_fig(x_range=(x_min, x_max), y_range=(y_min, y_max),
                        xlab=xlab, ylab=ylab, title=title, x=x,
                        x_axis_type=x_axis_type, x_name=x_name, y_name=y_name,
                        pw=pw, ph=ph)

    if cat:
        img = tf.shade(agg, color_key=color[:agg[cat].size])
        cbar = make_cbar(agg[cat].values, cat, cmap=color[:agg[cat].size])
        fig.add_layout(cbar, 'right')
    else:
        img = tf.shade(agg, cmap=color)

    cds = ColumnDataSource(data=dict(image=[img.data], x=[x_min],
                                     y=[y_min], dw=[dw], dh=[dh]))

    fig.image_rgba(source=cds, image='image', x='x', y='y',
                   dw='dw', dh='dh', dilate=False)

    return fig


def gen_grid(df, x_min, x_max, y_min, y_max, c_width, c_height, ncols=9,
             x_name=None, y_name=None, xlab=None, ylab=None, title=None,
             x_axis_type="linear", pw=190, ph=100, x=None,
             color=None, cat=None, cat_vals=None, ms_name=None,
             xds_table_obj=None):

    n_grid = []

    nrows = int(np.ceil(cat_vals.size / ncols))

    # if there are more columns than there are items
    if ncols > cat_vals.size:
        pw = int(PLOT_WIDTH / cat_vals.size)
        ph = int(PLOT_HEIGHT * 0.8)

    # If there is still space extend height
    if (nrows * ph) < (PLOT_HEIGHT * 0.85):
        ph = int((PLOT_HEIGHT * 0.85) / nrows)

    # frame dimensions. Actual size of plot without axes
    fw = int(0.93 * pw)
    fh = int(0.93 * ph)

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

    bk_xr = Range1d(*x_range)
    bk_yr = Range1d(*y_range)

    h_tool = HoverTool(tooltips=[(f"{cat.capitalize()}", "@i_axis"),
                                 (f"({xlab[:4]}, {ylab[:4]})", "($x, $y)")])

    # title for each single plot
    p_title = Text(x="x", y="y", text="text", text_font="monospace",
                   text_font_style="bold", text_font_size="10pt",
                   text_align="center")

    if x_name.lower() == "time":
        h_tool.formatters["$x"] = "datetime"

    tools = [BoxZoomTool(), h_tool, ResetTool(), WheelZoomTool(), PanTool(),
             SaveTool()]

    logger.info("Creating Bokeh grid")
    for i, c_val in enumerate(cat_vals):
        p_title_src = ColumnDataSource(data=dict(x=[x_min + (dw * 0.5)],
                                                 y=[y_max * 0.87],
                                                 text=[""]))

        if i_names:
            p_title_src.data["text"][0] = f"{i_names[c_val]}"
        else:
            p_title_src.data["text"][0] = f"{c_val}"

        f = Plot(frame_width=fw, frame_height=fh, plot_width=pw,
                 plot_height=ph, x_range=bk_xr,
                 y_range=bk_yr, sizing_mode="stretch_width",
                 outline_line_color="#18c3cf", outline_line_alpha=0.3,
                 outline_line_width=2)

        s_agg = agg.sel(**{cat: c_val})
        s_img = tf.shade(s_agg, cmap=color)

        if i_names:
            i_axis_data = [i_names[c_val]]
        else:
            i_axis_data = [c_val]

        s_ds = ColumnDataSource(data=dict(image=[s_img.data], x=[x_min],
                                          i_axis=i_axis_data,
                                          y=[y_min], dw=[dw], dh=[dh]))

        ir = ImageRGBA(image='image', x='x', y='y', dw='dw', dh='dh',
                       dilate=False)

        f.add_glyph(s_ds, ir)
        f.add_glyph(p_title_src, p_title)

        for tool in tools:
            f.add_tools(tool)

        # some common axis specifications
        axis_specs = dict(axis_label_text_font="monospace",
                          axis_label_text_font_style="normal",
                          axis_label_text_font_size="8pt",
                          minor_tick_line_width=0,
                          major_tick_out=2,
                          major_tick_in=2,
                          major_label_text_font_size="8pt",
                          ticker=BasicTicker(desired_num_ticks=4))

        # Add x-axis to all items in the last row
        if (i >= ncols * (nrows - 1)):
            f.extra_x_ranges = {"xr": bk_xr}
            f.add_layout(LinearAxis(x_range_name="xr",
                                    axis_label=xlab, **axis_specs),
                         'below')

            if x_name.lower() == "time":
                f.xaxis[0].formatter = DatetimeTickFormatter(hourmin=['%H:%M']
                                                             )

        # Add y-axis to all items in the first columns
        if (i % ncols == 0):
            f.extra_y_ranges = {"yr": bk_yr}
            f.add_layout(LinearAxis(y_range_name="yr",
                                    axis_label=ylab, **axis_specs),
                         'left')

            # Set formating if yaxis is phase
            if y_name.lower() == 'phase':
                f.yaxis[0].formatter = PrintfTickFormatter(
                    format=u"%f\u00b0")

        if x_name.lower() in ['channel', 'frequency']:
            # Add this extra axis on all items in the first row
            if (i < ncols):
                f = add_axis(fig=f,
                             axis_range=x.chan.values,
                             ax_label='Channel')

                # name of axis known from add_axis()
                # update with the new changes
                f.select(selector={
                    "name": "chan_xtra_linaxis"})[0].update(**axis_specs)

        n_grid.append(f)

    title_div = Div(text=title, align="center", width=PLOT_WIDTH,
                    style={"font-size": "24px", "height": "50px",
                           "text-align": "centre",
                           "font-weight": "bold",
                           "font-family": "monospace"},
                    sizing_mode="stretch_width")
    n_grid = grid(children=n_grid, ncols=ncols,
                  nrows=nrows, sizing_mode="stretch_both")
    final_grid = gridplot(children=[title_div, n_grid], ncols=1)

    return final_grid


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
    alts['amp'] = alts['Amp'] = 'amplitude'
    alts['ant1'] = alts['Antenna1'] = alts[
        'Antenna'] = alts['ant'] = 'antenna1'
    alts['ant2'] = alts['Antenna2'] = 'antenna2'
    alts['Baseline'] = alts['bl'] = 'baseline'
    alts['chan'] = alts['Channel'] = 'channel'
    alts['correlation'] = alts['Corr'] = 'corr'
    alts['freq'] = alts['Frequency'] = 'frequency'
    alts['Field'] = 'field'
    alts['imaginary'] = alts['Imag'] = 'imaginary'
    alts['Real'] = 'real'
    alts['Scan'] = 'scan'
    alts['Spw'] = 'spw'
    alts['Time'] = 'time'
    alts['Phase'] = 'phase'
    alts['UVdist'] = alts['uvdist'] = 'uvdistance'
    alts['uvdistl'] = alts['uvdist_l'] = alts['UVwave'] = 'uvwave'
    alts['ant'] = alts['Antenna'] = 'antenna1'

    # convert to proper name if in other name
    if inp in alts:
        inp = alts[inp]

    return inp


def hv_plotter(x, y, xaxis, xlab='', yaxis='amplitude', ylab='',
               color='blue', xds_table_obj=None, ms_name=None, iterate=None,
               x_min=None, x_max=None, y_min=None, y_max=None, c_width=None,
               c_height=None, colour_axis=None):
    """Responsible for plotting in this script.

    This is responsible for:

    - Selection of the iteration column. ie. Setting it to a categorical column
    - Creating the image callback to Datashader
    - Creating Bokeh canvas onto which image will be placed
    - Calculation of maximums and minimums for the plot
    - Formatting fonts, axes and titles

    Parameters
    ----------
    x : :obj:`xarray.DataArray`
        x data to plot
    y:  :obj:`xarray.DataArray`
        y data to plot
    xaxis : :obj:`str`
        xaxis selected for plotting
    xlab : :obj:`str`
        Label to appear on x-axis
    yaxis : :obj:`str`
        yaxis selected for plotting
    ylab : :obj:`str`
        Label to appear on y-axis
    iterate : :obj:`str`
        Column in the dataset over which to iterate. It should be noted that currently iteration is done using colors to denote the different parts of the iteration axis. These colors are explicitly selected in the code and are cycled through. i.e repetitive. This option is akin to the colorise_by function in CASA.
    ititle : :obj:`str`
        Title to appear incasea of iteration
    color :  :obj:`str`, :obj:`colormap`, :obj:`itertools.cycler`
        Color scheme to be used in the plot. It could be a string containing a color, a matplotlib or bokeh or colorcet colormap of a cycler containing specified colors.
    xds_table_obj : :obj:`xarray.Dataset`
        Dataset object containing the columns of the MS. This is passed on in case there are items required from the actual dataset.
    ms_name : :obj:`str`
        Name or [can include path] to Measurement Set
    xmin: :obj:`float`
        Minimum x value to be plotted

        Note
        ----
        This may be difficult to achieve in the case where :obj:`xaxis` is time because time in ``ragavi-vis`` is converted into milliseconds from epoch for ease of plotting by ``bokeh``.
    xmax: :obj:`float`
        Maximum x value to be plotted
    ymin: :obj:`float`
        Minimum y value to be plotted
    ymax: :obj:`float`
        Maximum y value to be plotted

    Returns
    -------
    fig: :obj:`bokeh.plotting.figure`
    """

    # iteration key word: data column name
    iter_cols = {'antenna1': 'ANTENNA1',
                 'antenna2': 'ANTENNA2',
                 'chan': 'chan',
                 'corr': 'corr',
                 'field': 'FIELD_ID',
                 'scan': 'SCAN_NUMBER',
                 'spw': 'DATA_DESC_ID',
                 'baseline': 'Baseline',
                 None: None}

    # change xaxis name to frequency if xaxis is channel. For df purpose
    xaxis = 'frequency' if xaxis == 'channel' else xaxis

    # changing the Name of the dataArray to the name provided as xaxis
    x.name = xaxis.capitalize()
    y.name = yaxis.capitalize()

    # set plot title name
    title = "{} vs {}".format(y.name, x.name)

    if xaxis == 'time':
        # bounds for x
        bds = np.array([np.nanmin(x), np.nanmax(x)]).astype(datetime)
        # creating the x-xaxis label
        xlab = "Time {} to {}".format(bds[0].strftime("%Y-%m-%d %H:%M:%S"),
                                      bds[-1].strftime("%Y-%m-%d %H:%M:%S"))
        # convert to milliseconds from epoch for bokeh
        x = x.astype('float64') * 1000
        x_axis_type = "datetime"
    else:
        x_axis_type = 'linear'

    # setting maximum and minimums if they are not user defined
    if x_min == None:
        x_min = x.min().data

    if x_max == None:
        x_max = x.max().data

    if y_min == None:
        y_min = y.min().data

    if y_max == None:
        y_max = y.max().data

    logger.info("Calculating x and y Min and Max ranges")
    with ProgressBar():
        x_min, x_max, y_min, y_max = compute(x_min, x_max, y_min, y_max)
    logger.info("Done")

    if iterate != None:
        title = title + " Iterated By: {}".format(iterate.capitalize())
        if iterate in ['corr', 'chan']:
            if iterate == 'corr':
                iter_data = xr.DataArray(da.arange(y[iterate].size),
                                         name=iterate, dims=[iterate])
        elif iter_cols[iterate] == 'Baseline':
            # get the data array over which to iterate and merge it to x and y
            iter_data = create_bl_data_array(xds_table_obj)
        else:
            try:
                iter_data = xds_table_obj[iter_cols[iterate]]
            except:
                logger.error("Specified data column not found.")

        logger.info("Creating Dataframe")
        xy_df = create_df(x, y, iter_data=iter_data)[[x.name, y.name,
                                                      iter_cols[iterate]]]
        cats = np.unique(iter_data.values)

        # change value of iterate to the required data column
        iterate = iter_cols[iterate]

        xy_df = xy_df.astype({iterate: 'category'})
        xy_df[iterate] = xy_df[iterate].cat.as_known()

        # generate resulting grid
        image = gen_grid(xy_df, x_min, x_max, y_min, y_max,
                         c_width=c_width, c_height=c_height, x_name=x.name,
                         y_name=y.name, cat=iterate, cat_vals=cats, ncols=9,
                         color=color, title=title, x_axis_type=x_axis_type,
                         pw=210, ph=100, xlab=xlab, ylab=ylab, x=x,
                         ms_name=ms_name, xds_table_obj=xds_table_obj)

    if colour_axis != None:
        title = title + " Colourised By: {}".format(colour_axis.capitalize())
        if colour_axis in ['corr', 'chan']:
            if colour_axis == 'corr':
                iter_data = xr.DataArray(da.arange(y[colour_axis].size),
                                         name=colour_axis, dims=[colour_axis])
        elif iter_cols[colour_axis] == 'Baseline':
            # get the data array over which to colour_axis and merge it to x
            # and y
            iter_data = create_bl_data_array(xds_table_obj)
        else:
            try:
                iter_data = xds_table_obj[iter_cols[colour_axis]]
            except:
                logger.error("Specified data column not found.")
        logger.info("Creating Dataframe")
        xy_df = create_df(x, y, iter_data=iter_data)[[x.name, y.name,
                                                      iter_cols[colour_axis]]]

        cats = np.unique(iter_data.values)

        # change value of colour_axis to the required data column
        colour_axis = iter_cols[colour_axis]

        xy_df = xy_df.astype({colour_axis: 'category'})
        xy_df[colour_axis] = xy_df[colour_axis].cat.as_known()

        # generate resulting image
        image = gen_image(xy_df, x_min, x_max, y_min, y_max, x=x,
                          c_width=c_width, c_height=c_height, x_name=x.name,
                          y_name=y.name, cat=colour_axis, color=color,
                          title=title, ylab=ylab, xlab=xlab,
                          x_axis_type=x_axis_type)

    if colour_axis is None and iterate is None:
        logger.info("Creating Dataframe")
        xy_df = create_df(x, y, iter_data=None)[[x.name, y.name]]

        # generate resulting image
        image = gen_image(xy_df, x_min, x_max, y_min, y_max, x=x,
                          c_width=c_width, c_height=c_height, x_name=x.name,
                          y_name=y.name, cat=iterate, color=color,
                          title=title, ylab=ylab, xlab=xlab,
                          x_axis_type=x_axis_type)

    elif colour_axis is not None and iterate is not None:
        logger.error(
            """Unable to generate plot. Ensure that NOT both colour-axis and iter-axis have been specified.""")

    return image


def get_ms(ms_name, chunks=None, data_col='DATA', ddid=None, fid=None,
           scan=None, where=None, cbin=None, tbin=None, chan_select=None):
    """Get xarray Dataset objects containing Measurement Set columns of the selected data

    Parameters
    ----------
    ms_name: :obj:`str`
        Name of your MS or path including its name
    chunks: :obj:`str`
        Chunk sizes for the resulting dataset.
    cbin: :obj:`int`
        Number of channels binned together for channel averaging
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
    where: :obj:`str`
        TAQL where clause to be used with the MS.
    chan_select: :obj:`int` or :obj:`slice`
        Channels to be selected

    Returns
    -------
    tab_objs: :obj:`list`
        A list containing the specified table objects as  :obj:`xarray.Dataset`
    """
    sel_cols = ["ANTENNA1", "ANTENNA2",
                "UVW", "TIME", "FIELD_ID",
                "FLAG", "SCAN_NUMBER", "DATA_DESC_ID"]
    sel_cols.append(data_col)

    ms_schema = MS_SCHEMA.copy()
    ms_schema['WEIGHT_SPECTRUM'] = ms_schema['DATA']

    # defining part of the gain table schema
    if data_col not in ['DATA', 'CORRECTED_DATA']:
        ms_schema[data_col] = ms_schema['DATA']

    if chunks is None:
        chunks = dict(row=100000)
    else:
        dims = ['row', 'chan', 'corr']
        chunks = {k: int(v) for k, v in zip(dims, chunks.split(','))}

    # always ensure that where stores something
    if where == None:
        where = []
    else:
        where = [where]

    if ddid is not None:
        where.append("DATA_DESC_ID IN {}".format(ddid))

    if fid is not None:
        where.append("FIELD_ID IN {}".format(fid))

    if scan is not None:
        where.append("SCAN_NUMBER IN {}".format(scan))

    # combine the strings to form the where clause
    where = " && ".join(where)

    try:
        if cbin or tbin:
            logger.info("Averaging active")
            sel_cols.append("INTERVAL")
            # automatically groups data into spectral windows
            tab_objs = average_ms(ms_name, tbin=tbin, cbin=cbin,
                                  chunk_size=None, taql=where,
                                  columns=sel_cols, chan=chan_select)
        else:
            tab_objs = xm.xds_from_ms(ms_name, taql_where=where,
                                      table_schema=ms_schema,
                                      group_cols=[],
                                      chunks=chunks,
                                      columns=sel_cols)
            # select channels
            if chan_select is not None:
                tab_objs = [_.sel(chan=chan_select) for _ in tab_objs]

        # get some info about the data
        chunk_sizes = tab_objs[0].FLAG.data.chunksize
        chunk_p = tab_objs[0].FLAG.data.npartitions

        logger.info("Chunk sizes: {}".format(chunk_sizes))
        logger.info("Number of Partitions: {}".format(chunk_p))

        return tab_objs
    except:
        logger.exception(
            "Invalid DATA_DESC_ID, FIELD_ID, SCAN_NUMBER or TAQL clause")
        sys.exit(-1)


def main(**kwargs):
    """Main function that launches the visibilities plotter"""

    if 'options' in kwargs:
        NB_RENDER = False
        options = kwargs.get('options', None)
        chan = options.chan
        corr = options.corr
        chunks = options.chunks
        data_column = options.data_column
        ddid = options.ddid
        fields = options.fields
        flag = options.flag
        iterate = validate_axis_inputs(options.iterate)
        html_name = options.html_name
        mycmap = options.mycmap
        mytabs = options.mytabs
        mem_limit = options.mem_limit
        n_cores = options.n_cores
        scan = options.scan
        where = options.where
        xaxis = validate_axis_inputs(options.xaxis)
        xmin = options.xmin
        xmax = options.xmax
        ymin = options.ymin
        ymax = options.ymax
        yaxis = validate_axis_inputs(options.yaxis)
        tbin = options.tbin
        cbin = options.cbin
        c_width = options.c_width
        c_height = options.c_height
        colour_axis = validate_axis_inputs(options.colour_axis)

    # number of processes or threads/ cores
    dask_config.set(num_workers=n_cores, memory_limit=mem_limit)

    if len(mytabs) > 0:
        mytabs = [x.rstrip("/") for x in mytabs]
    else:
        logger.error('ragavi exited: No Measurement set specified.')
        sys.exit(-1)

    for mytab in mytabs:
        # for each table

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

        if scan != None:
            scan = vu.resolve_ranges(scan)

        if ddid != None:
            n_ddid = vu.slice_data(ddid)
            ddid = vu.resolve_ranges(ddid)
        else:
            # set data selection to all unless ddid is specified
            n_ddid = slice(0, None)

        # capture channels or correlations to be selected
        chan = vu.slice_data(chan)

        corr = vu.slice_data(corr)

        # open MS perform averaging and select desired fields, scans and spws
        partitions = get_ms(mytab, data_col=data_column, ddid=ddid,
                            fid=fields, scan=scan, where=where, chunks=chunks,
                            tbin=tbin, cbin=cbin, chan_select=chan)

        if mycmap in cc.palette:
            mycmap = cc.palette[mycmap]
        else:
            logging.info(
                """Selected color not found in palette. Reverting to default""")
            mycmap = cc.palette["blues"]

        if colour_axis != None:
            logger.info('Enforcing categorical colours for colour axis')
            mycmap = cc.palette['glasbey_bw']

        oup_a = []

        # iterating over spectral windows (spws)
        for count, chunk in enumerate(partitions):

            # We check current DATA_DESC_ID
            if isinstance(chunk.DATA_DESC_ID, xr.DataArray):
                c_spw = chunk.DATA_DESC_ID[0].values
            else:
                c_spw = chunk.DATA_DESC_ID

            logger.info("Starting data processing.")

            f = DataCoreProcessor(chunk, mytab, xaxis, yaxis, chan=chan,
                                  corr=corr, flag=flag, ddid=c_spw,
                                  datacol=data_column, cbin=cbin)
            ready = f.act()

            logger.info("Starting plotting function.")

            # black box returns an plottable element / composite element
            fig = hv_plotter(x=ready.x, y=ready.y, xaxis=xaxis,
                             xlab=ready.xlabel, yaxis=yaxis,
                             ylab=ready.ylabel, color=mycmap, iterate=iterate,
                             xds_table_obj=chunk, ms_name=mytab, x_min=xmin,
                             x_max=xmax, y_min=ymin, y_max=ymax,
                             c_width=c_width, c_height=c_height,
                             colour_axis=colour_axis)

            logger.info("Plotting complete.")

        if html_name:
            if 'html' not in html_name:
                html_name += '.html'
            fname = html_name
        else:
            fname = "{}_{}_{}.html".format(
                mytab.split('/')[-1], yaxis, xaxis)

        output_file(fname, title=fname)
        save(fig)

        logger.info("Rendered plot to: {}".format(fname))
        logger.info(wrapper.fill(",\n".join(
            ["{}: {}".format(k, v) for k, v in options.__dict__.items() if v != None])))

        logger.info(">" * (len(fname) + 19))
