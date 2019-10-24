# -*- coding: utf-8 -*-
from __future__ import division

import logging
import sys
import warnings

import colorcet as cc
import dask.array as da
import daskms as xm
import datashader as ds
import datashader.transfer_functions as tf

import numpy as np
import xarray as xr


from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
from dask import compute, delayed
from daskms.table_schemas import MS_SCHEMA
from datashader.bokeh_ext import InteractiveImage
from itertools import cycle


from bokeh.plotting import figure
from bokeh.layouts import column, gridplot, row, widgetbox
from bokeh.io import (export_png, output_file, output_notebook,
                      show, save)
from bokeh.models import (BasicTicker, ColumnDataSource,
                          ColorBar, Div, HoverTool, LinearAxis,
                          LinearColorMapper, PrintfTickFormatter, Range1d,
                          Title)
from dask.diagnostics import ProgressBar

from . import utils as vu

logger = vu.logger
excepthook = vu.sys.excepthook
time_wrapper = vu.time_wrapper
wrapper = vu.textwrap.TextWrapper(initial_indent='',
                                  break_long_words=True,
                                  subsequent_indent=''.rjust(50),
                                  width=160)

# vu.__welcome()


class DataCoreProcessor:
    """Process Measurement Set data into forms desirable for visualisation.

    Parameters
    ----------
    chan : :obj:`slice`
        Channels that will be selected. Defaults to all
    corr : :obj:`int`
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
                 chan=slice(0, None), corr=slice(0, None),
                 ddid=slice(0, None), datacol='DATA', flag=True):

        self.xds_table_obj = xds_table_obj
        self.ms_name = ms_name
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.corr = corr
        self.ddid = ddid
        self.chan = chan
        self.datacol = datacol
        self.flag = flag

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

    def get_xaxis_data(self, xds_table_obj, ms_name, xaxis, datacol='DATA'):
        """Get x-axis data. It is dependent on the gaintype.
        This function also returns the relevant x-axis labels for both pairs of plots.

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
            xdata = vu.get_frequencies(ms_name)
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

    def prep_xaxis_data(self, xdata, chan=slice(0, None), freq=None,
                        xaxis='time'):
        """Prepare the x-axis data for plotting.

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
            prepdx = xdata.sel(chan=chan) / 1e9
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
                        chan=slice(0, None), corr=0, flag=True,
                        yaxis='amplitude'):
        """Process data for the y-axis.
        This includes:

            * Correlation selection
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
        ydata = ydata.sel(dict(corr=corr, chan=chan))
        flags = vu.get_flags(xds_table_obj).sel(dict(corr=corr, chan=chan))

        # if flagging enabled return a list of DataArrays otherwise return a
        # single dataarray
        if flag:
            processed = self.process_data(ydata, yaxis=yaxis, wrap=True)
            y = processed.where(flags == False)
        else:
            y = self.process_data(ydata, yaxis=yaxis)

        return y

    def blackbox(self, xds_table_obj, ms_name, xaxis, yaxis,
                 chan=slice(0, None), corr=0, datacol='DATA', ddid=None,
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
                         ddid=ddid, flag=flag, datacol=datacol)
        x_prepd = xs.x
        xlabel = xs.xlabel

        ys = self.y_only(xds_table_obj, ms_name, yaxis, chan=chan, corr=corr,
                         flag=flag, datacol=datacol)
        y_prepd = ys.y
        ylabel = ys.ylabel

        if xaxis == 'channel' or xaxis == 'frequency':
            if y_prepd.ndim == 3:
                y_prepd = y_prepd.transpose('chan', 'row', 'corr')
            else:
                y_prepd = y_prepd.T

            # selected channel numbers
            chan_nos = xds_table_obj[datacol].chan.values[chan]

            # assign chan coordinates to both x and y
            # these coordinates should correspond to the selected channels
            y_prepd = y_prepd.assign_coords(chan=chan_nos)
            x_prepd = x_prepd.assign_coords(chan=chan_nos)

            # delete table row coordinates for channel data because of
            # incompatibility
            del x_prepd.coords['table_row']

        d = Data(x=x_prepd, xlabel=xlabel, y=y_prepd, ylabel=ylabel)

        return d

    def act(self):
        """Activate the :meth:`ragavi.ragavi.DataCoreProcessor.blackbox`
        """
        return self.blackbox(self.xds_table_obj, self.ms_name, self.xaxis,
                             self.yaxis, chan=self.chan, corr=self.corr,
                             datacol=self.datacol, ddid=self.ddid,
                             flag=self.flag)

    def x_only(self, xds_table_obj, ms_name, xaxis, flag=True, corr=0,
               chan=slice(0, None), ddid=None, datacol='DATA'):
        """Return only x-axis data and label

        Returns
        -------
        d : :obj:`collections.namedtuple`
            Named tuple containing x-axis data and x-axis label. Items in the tuple can be accessed by using the dot notation.
        """
        Data = namedtuple('Data', 'x xlabel')
        x_data, xlabel = self.get_xaxis_data(xds_table_obj, ms_name, xaxis,
                                             datacol=datacol)
        if xaxis == 'uvwave':
            # compute uvwave using the available selected frequencies
            freqs = vu.get_frequencies(ms_name, spwid=ddid,
                                       chan=chan).compute()
            x = self.prep_xaxis_data(x_data, xaxis=xaxis, freq=freqs)

        # if we have x-axes corresponding to ydata
        elif xaxis == 'real' or xaxis == 'phase':
            x = self.prep_yaxis_data(xds_table_obj, ms_name, x_data,
                                     yaxis=xaxis, corr=corr, chan=chan,
                                     flag=flag)
        else:
            x = self.prep_xaxis_data(x_data, xaxis=xaxis, chan=chan)
        d = Data(x=x, xlabel=xlabel)
        return d

    def y_only(self, xds_table_obj, ms_name, yaxis, chan=slice(0, None),
               corr=0, datacol='DATA', flag=True):
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
                                 chan=chan, corr=corr, flag=flag)
        d = Data(y=y, ylabel=ylabel)
        return d


def save_png_image(name, disp_layout):
    """Save plots in PNG format
    Note
    ----
    One image will emerge for each figure in `disp_layout`. To save png images, the python package selenium, node package phantomjs are required. More information `Exporting bokeh plots <https://bokeh.pydata.org/en/latest/docs/user_guide/export.html>`_

    Parameters
    ----------
    disp_layout : :obj:`bokeh.layouts`
        Layout object containing the renderers
    name : :obj:`str`
        Name of output image
    """
    export_png(disp_layout, filename=name)


def add_axis(fig, axis_range, ax_label):
    """Add an extra axis to the current figure

    Parameters
    ----------
    fig: :obj:`bokeh.plotting.figure`
        The figure onto which to add extra axis

    axis_range: :obj:`list`, :obj:`tuple`
        A range of sorted values or a tuple with 2 values containing or the order (min, max). This can be any ordered iteraable that can be indexed.

    Returns
    -------
    fig : :obj:`bokeh.plotting.figure`
        Bokeh gigure with an extra axis added
    """
    fig.extra_x_ranges = {"fxtra": Range1d(
        start=axis_range[0], end=axis_range[-1])}
    linaxis = LinearAxis(x_range_name="fxtra", axis_label=ax_label,
                         minor_tick_line_alpha=0,
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

    # not using categorical because we'll need to define our own colors
    cmapper = LinearColorMapper(palette=cmap[:ncats], low=lo, high=hi)
    b_ticker = BasicTicker(desired_num_ticks=ncats)

    cbar = ColorBar(color_mapper=cmapper, label_standoff=12,
                    border_line_color=None, ticker=b_ticker,
                    location=(0, 0), title=category, title_standoff=5,
                    title_text_font_size='15pt')

    return cbar


def hv_plotter(x, y, xaxis, xlab='', yaxis='amplitude', ylab='',
               color='blue', xds_table_obj=None, ms_name=None, iterate=None,
               x_min=None, x_max=None, y_min=None, y_max=None):
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
    ms_nmae : :obj:`str`
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
    iters = {'antenna1': 'ANTENNA1',
             'antenna2': 'ANTENNA2',
             'chan': 'chan',
             'corr': 'corr',
             'field': 'FIELD_ID',
             'scan': 'SCAN_NUMBER',
             'spw': 'DATA_DESC_ID',
             None: None}

    @time_wrapper
    def image_callback(xr, yr, w, h, x=None, y=None, cat=None, cat_ids=None, col=None):
        cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=xr, y_range=yr)
        logger.info("Datashader aggregation starting")
        if cat:
            with ProgressBar():
                agg = cvs.points(xy_df, x, y, ds.count_cat(cat))
            img = tf.shade(agg, color_key=col[:cat_ids.size])
        else:
            with ProgressBar():
                agg = cvs.points(xy_df, x, y, ds.count())
            img = tf.shade(agg, cmap=col)
        logger.info("Aggregation done")
        return img

    # change xaxis name to frequency if xaxis is channel. For df purpose
    xaxis = 'frequency' if xaxis == 'channel' else xaxis

    # changing the Name of the dataArray to the name provided as xaxis
    x_cap = xaxis.capitalize()
    y_cap = yaxis.capitalize()

    x.name = x_cap
    y.name = y_cap

    # set plot title name
    title = "{} vs {}".format(y_cap, x_cap)

    if xaxis == 'time':
        # bounds for x
        bds = np.array([np.nanmin(x), np.nanmax(x)]).astype(datetime)
        # creating the x-xaxis label
        txaxis = "Time {} to {}".format(bds[0].strftime("%Y-%m-%d %H:%M:%S"),
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

    if iterate:
        title = title + " Colorise By: {}".format(iterate.capitalize())
        if iterate in ['corr', 'chan']:
            xy = xr.merge([x, y])
            if iterate == 'corr':
                cats = y.corr.values
            else:
                cats = y.chan.values
        else:
            # get the data array over which to iterate and merge it to x and y
            iter_data = xds_table_obj[iters[iterate]]
            xy = xr.merge([x, y, iter_data])
            cats = np.unique(iter_data.values)

        # change value of iterate to the required data column
        iterate = iters[iterate]
        xy_df = xy.to_dask_dataframe()[[x_cap, y_cap, iterate]]
        xy_df = xy_df.astype({iterate: 'category'})

        # initialise bokeh colorbar
        cbar = make_cbar(cats, iterate, cmap=color)
    else:
        xy = xr.merge([x, y])
        xy_df = xy.to_dask_dataframe()[[x_cap, y_cap]]
        cats = None
        cbar = None

    logger.info('Creating canvas')
    fig = figure(tools='pan,box_zoom,wheel_zoom,reset,save',
                 x_range=(x_min, x_max), y_axis_label=ylab,
                 y_range=(y_min, y_max),
                 x_axis_type=x_axis_type, title=title,
                 plot_width=1920, plot_height=1080,
                 sizing_mode='stretch_both')

    logger.info("Starting datashading")

    im = InteractiveImage(fig, image_callback, x=x_cap,
                          y=y_cap, cat=iterate,
                          col=color, cat_ids=cats)

    h_tool = HoverTool(tooltips=[(x_cap, '$x'),
                                 (y_cap, '$y')])
    fig.add_tools(h_tool)

    fig.xaxis.axis_label = txaxis if xaxis == 'time' else xlab

    x_axis = fig.xaxis[0]
    x_axis.major_label_orientation = 45
    x_axis.ticker.desired_num_ticks = 30

    if cbar:
        fig.add_layout(cbar, 'right')

    if xaxis == 'channel' or xaxis == 'frequency':
        fig = add_axis(fig=fig,
                       axis_range=x.chan.values,
                       ax_label='Channel')
    if yaxis == 'phase':
        fig.yaxis[0].formatter = PrintfTickFormatter(format=u"%f\u00b0")

    fig.axis.axis_label_text_font_style = "normal"
    fig.axis.axis_label_text_font_size = "15px"
    fig.title.text_font = "helvetica"
    fig.title.text_font_size = "25px"
    fig.title.align = 'center'

    return fig


def get_argparser():
    """Create command line arguments for ragavi-gains

    Returns
    -------
    parser : :obj:`ArgumentParser()`
        Argument parser object that contains command line argument's values

    """
    x_choices = ['antenna1', 'antenna2',
                 'channel', 'frequency', 'phase',
                 'real', 'scan', 'time',
                 'uvdistance', 'uvwave']

    y_choices = ['amplitude', 'imaginary', 'phase', 'real']

    iter_choices = ['antenna1', 'antenna2', 'chan', 'corr', 'field', 'scan',
                    'spw']

    parser = ArgumentParser(usage='ragavi-vis [options] <value>')

    required = parser.add_argument_group('Required arguments')
    required.add_argument('--ms', dest='mytabs',
                          nargs='*', type=str, metavar='',
                          help='Table(s) to plot. Default is None',
                          default=[])
    required.add_argument('--xaxis', dest='xaxis', type=str, metavar='',
                          choices=x_choices, help='x-axis to plot',
                          default=None)
    required.add_argument('--yaxis', dest='yaxis', type=str, metavar='',
                          choices=y_choices, help='Y axis variable to plot',
                          default=None)

    parser.add_argument('--corr', dest='corr', type=str, metavar='',
                        help="""Correlation index or subset to plot Can be specified using normal python slicing syntax i.e "0:5" for 0<=corr<5 or "::2" for every 2nd corr or "0" for corr 0  or "0,1,3". Default is all.""",
                        default='0:')
    parser.add_argument('--chan', dest='chan', type=str, metavar='',
                        help="""Channels to select. Can be specified using syntax i.e "0:5" (exclusive range) or "20" for channel 20 or "10~20" (inclusive range) (same as 10:21) "::10" for every 10th channel or "0,1,3" etc. Default is all.""",
                        default=None)
    parser.add_argument('--chunks', dest='chunks', type=str, metavar='',
                        help="""Chunk sizes to be applied to the dataset. Can be an integer e.g "1000", or a comma separated string e.g "1000,100,2" for multiple dimensions. The available dimensions are (row, chan, corr) respectively. If an integer, the specified chunk size will be applied to all dimensions. If comma separated string, these chunk sizes will be applied to each dimension respectively. Default is 100000""",
                        default=None)
    parser.add_argument('--cmap', dest='mycmap', type=str, metavar='',
                        help="""Colour or colour map to use. Default is blue. A list of valid cmap arguments can be found at: 
                        https://colorcet.pyviz.org/user_guide/index.html
                        Note that if the argument "colorize" is supplied, 
                        a categorical colour scheme will be adopted.""",
                        default=None)
    parser.add_argument('--data-column', dest='data_column', type=str,
                        metavar='',
                        help="""Column from MS to use for data. Default is DATA.""", default='DATA')
    parser.add_argument('--ddid', dest='ddid', type=str,
                        metavar='',
                        help="""DATA_DESC_ID(s) to select. Can be specified as e.g. "5", "5,6,7", "5~7" (inclusive range), "5:8" (exclusive range), 5:(from 5 to last). Default is all.""",
                        default=None)
    parser.add_argument('--field', dest='fields', type=str,
                        metavar='',
                        help="""Field ID(s) / NAME(s) to plot. Can be specified as "0", "0,2,4", "0~3" (inclusive range), "0:3" (exclusive range), "3:" (from 3 to last) or using a field name or comma separated field names. Default is all""",
                        default=None)
    parser.add_argument('--htmlname', dest='html_name', type=str, metavar='',
                        help='Output HTMLfile name', default=None)
    parser.add_argument('--image-name', dest='image_name', type=str,
                        metavar='',
                        help="""Output png name. This requires works with phantomJS and selenium installed packages installed.""", default=None)
    parser.add_argument('--colorize', dest='iterate', type=str, metavar='',
                        choices=iter_choices,
                        help="""Select which columnt to colorize by. Default is None.""",
                        default=None)
    # To Do: Changed iterate argument to colorize. Iteration to be added later
    parser.add_argument('--no-flag', dest='flag', action='store_false',
                        help="""Plot both flagged and unflagged data. Default only plot data that is not flagged.""",
                        default=True)
    parser.add_argument('--scan', dest='scan', type=str, metavar='',
                        help='Scan Number to select. Default is all.',
                        default=None)
    parser.add_argument('--taql', dest='where', type=str, metavar='',
                        help='TAQL where', default=None)
    parser.add_argument('--xmin', dest='xmin', type=float, metavar='',
                        help='Minimum x value to plot', default=None)
    parser.add_argument('--xmax', dest='xmax', type=float, metavar='',
                        help='Maximum x value to plot', default=None)
    parser.add_argument('--ymin', dest='ymin', type=float, metavar='',
                        help='Minimum y value to plot', default=None)
    parser.add_argument('--ymax', dest='ymax', type=float, metavar='',
                        help='Maximum y value to plot', default=None)

    return parser


def get_ms(ms_name, chunks=None, data_col='DATA', ddid=None, fid=None, scan=None, where=None):
    """Get xarray Dataset objects containing Measurement Set columns of the selected data

    Parameters
    ----------
    ms_name: :obj:`str`
        Name of your MS or path including its name
    chunks: :obj:`str`
        Chunk sizes for the resulting dataset.
    data_col: :obj:`str`
        Data column to be used. Defaults to 'DATA'
    ddid: :obj:`int`
        DATA_DESC_ID or spectral window to choose. Defaults to all
    fid: :obj:`int`
        Field id to select. Defaults to all
    scan: :obj:`int`
        SCAN_NUMBER to select. Defaults to all
    where: :obj:`str`
        TAQL where clause to be used with the MS.

    Returns
    -------
    tab_objs: :obj:`list`
        A list containing the specified table objects as  :obj:`xarray.Dataset`
    """

    ms_schema = MS_SCHEMA.copy()
    ms_schema['WEIGHT_SPECTRUM'] = ms_schema['DATA']
    # defining part of the gain table schema
    if data_col not in ['DATA', 'CORRECTED_DATA']:
        ms_schema[data_col] = ms_schema['DATA']
    if chunks is None:
        chunks = {'row': 100000}
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
        tab_objs = xm.xds_from_ms(ms_name, taql_where=where,
                                  table_schema=ms_schema, group_cols=[],
                                  chunks=chunks)
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
        iterate = options.iterate
        html_name = options.html_name
        image_name = options.image_name
        mycmap = options.mycmap
        mytabs = options.mytabs
        scan = options.scan
        where = options.where
        xaxis = options.xaxis
        xmin = options.xmin
        xmax = options.xmax
        ymin = options.ymin
        ymax = options.ymax
        yaxis = options.yaxis
        # timebin = options.timebin
        # chanbin = options.chanbin

    if len(mytabs) > 0:
        mytabs = [x.rstrip("/") for x in mytabs]
    else:
        logger.error('ragavi exited: No Measurement set specified.')
        sys.exit(-1)

    for mytab in mytabs:
        # for each pair of table and field

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

        partitions = get_ms(mytab, data_col=data_column, ddid=ddid,
                            fid=fields, scan=scan, where=where, chunks=chunks)

        if mycmap == None:
            mycmap = cc.palette['blues']
        else:
            if mycmap in cc.palette:
                mycmap = cc.palette[mycmap]
            else:
                pass

        oup_a = []

        for count, chunk in enumerate(partitions):

            # black box returns an plottable element / composite element
            if iterate != None:
                logger.info('Setting categorical colors')
                mycmap = cc.palette['glasbey_bw']

            logger.info("Starting data processing.")

            f = DataCoreProcessor(chunk, mytab, xaxis, yaxis, chan=chan,
                                  corr=corr, flag=flag, ddid=n_ddid,
                                  datacol=data_column)
            ready = f.act()

            logger.info("Starting plotting function.")

            fig = hv_plotter(x=ready.x, y=ready.y, xaxis=xaxis,
                             xlab=ready.xlabel, yaxis=yaxis,
                             ylab=ready.ylabel, color=mycmap, iterate=iterate,
                             xds_table_obj=chunk, ms_name=mytab, x_min=xmin,
                             x_max=xmax, y_min=ymin, y_max=ymax)

            logger.info("Plotting complete.")

        if image_name:
            image_name += '.png'
            save_png_image(image_name, fig)
            logger.info("Saved PNG image under name: {}".format(image_name))

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
