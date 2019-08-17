# -*- coding: utf-8 -*-

from __future__ import division

import logging
import glob
import re
import sys
import warnings

import dask.array as da
import datashader as ds
import datashader.transfer_functions as tf

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pyrap.quanta as qa
import xarrayms as xm
import xarray as xa


from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
from dask import compute, delayed
from datashader.bokeh_ext import InteractiveImage
from itertools import cycle
from xarrayms.known_table_schemas import MS_SCHEMA, ColumnSchema


from bokeh.plotting import figure
from bokeh.layouts import column, gridplot, row, widgetbox
from bokeh.io import (export_png, export_svgs, output_file, output_notebook,
                      show, save)
from bokeh.models import (BasicTicker, CheckboxGroup, ColumnDataSource,
                          CustomJS, Div, HoverTool, LinearAxis, Legend,
                          PrintfTickFormatter, Range1d,
                          Select, Text, Toggle, Title)

import vis_utils as vu
from ipdb import set_trace


logger = vu.logger
excepthook = vu.sys.excepthook
wrapper = vu.textwrap.TextWrapper(initial_indent='',
                                  break_long_words=True,
                                  subsequent_indent=''.rjust(50),
                                  width=160)

vu.welcome()


class DataCoreProcessor:

    def __init__(self, xds_table_obj, ms_name, xaxis, yaxis,
                 chan=slice(0, None), corr=0, ddid=0, datacol='DATA',
                 flag=True):

        self.xds_table_obj = xds_table_obj
        self.ms_name = ms_name
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.corr = corr
        self.ddid = ddid
        self.chan = chan
        self.datacol = datacol
        self.flag = flag
        # self.iterate = iterate

    def process_data(self, ydata, yaxis, wrap=True):
        """Abstraction for processing y-data passes it to the processing function.
        Inputs
        ------
        ydata: xarray DataArray
               y-data to process
        yaxis: str
               Selected yaxis

        Outputs
        -------
        y: xarray DataArray
           Processed y-data
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
        """
            Function to get x-axis data. It is dependent on the gaintype.
            This function also returns the relevant x-axis labels for both pairs of plots.

            Inputs
            ------
            xds_table_obj: xarray Dataset
                           MS as xarray dataset from xarrayms
            ms_name: str
                     Name of measurement set
            xaxis: str
                   Name of xaxis

            Outputs
            -------
            xdata: xarray DataArray
                   X-axis data depending  x-axis selected.
            x_label: str
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
        """
            Extract the required column for the y-axis data.


            Inputs
            -----
            xds_table_obj: xarray Dataset
                           MS as xarray dataset from xarrayms
            ms_name: str
                     Name of measurement set.
            yaxis: str
                   yaxis to plot.
            datacol: str
                     Data column to be selected.

            Outputs
            -------
            ydata: xarray DataArray
                   y-axis data depending  y-axis selected.
            y_label: str
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
        """
            Prepare the x-axis data for plotting.

            Inputs
            ------
            xdata: xarray DataArray
                   X-axis data depending  x-axis selected.
            xaxis: str
                   xaxis to plot.

            freq: xarray DataArray or float
                  Frequency(ies) from which corresponding wavelength will be obtained.
                  REQUIRED ONLY when xaxis specified is 'uvwave'.

            Outputs
            -------
            prepdx: xarray DataArray
                    Prepared data for the x-axis.
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
        """
            Process data for the y-axis which includes:
                - Correlation selection
                - Flagging
                - Conversion form complex to the required form
            Data selection and flagging are done by this function itself, however ap and ri conversion are done by specified functions.

            Inputs
            ------
            xds_table_obj: xarray Dataset
                           MS as xarray dataset from xarrayms
            ms_name: str
                     Name of measurement set.
            ydata: xarray DataArray
                   y-axis data to be processed
            yaxis: str
                   selected y-axis
            corr: int
                  Correlation number to select
            flag: bool
                  Option on whether to flag the data or not

            Outputs
            -------
            y: xarray DataArray
               Processed yaxis data.
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
        """
            Takes in raw input and gives out a holomap.

            Inputs
            ------

            chan: slice / numpy array
                  For the selection of the channels. Defaults to all
            corr: int
                  Correlation index to select
            ddid:
                  spectral window(s) to be selected
            datacol: str
                     Column from which data is pulled. Default is 'DATA'
            flag: bool
                         Switch flags on or off.
            ms_name: str
                     Measurement set name
            xaxis: str
                   Selected x-axis
            xds_table_obj: xarray Dataset
                           MS as xarray dataset from xarrayms
            yaxis: str
                   Selected y-axis

            Outputs
            -------
            data:
                  Processed data for x and y
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

            # assign chan coordinates to both x and y
            # these coordinates should correspond to the selected channels
            y_prepd = y_prepd.assign_coords(chan=y_prepd.chan.values[chan])
            x_prepd = x_prepd.assign_coords(chan=y_prepd.chan.values[chan])

            # delete table row coordinates for channel data because of
            # incompatibility
            del x_prepd.coords['table_row']

        d = Data(x=x_prepd, xlabel=xlabel, y=y_prepd, ylabel=ylabel)

        return d

    def act(self):
        return self.blackbox(self.xds_table_obj, self.ms_name, self.xaxis,
                             self.yaxis, chan=self.chan, corr=self.corr,
                             datacol=self.datacol, ddid=self.ddid,
                             flag=self.flag)

    def x_only(self, xds_table_obj, ms_name, xaxis, flag=True, corr=0,
               chan=slice(0, None), ddid=None, datacol='DATA'):
        """Only return xaxis data and label
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
        """Only return yaxis data and label
        """

        Data = namedtuple('Data', 'y ylabel')
        y_data, ylabel = self.get_yaxis_data(xds_table_obj, ms_name, yaxis,
                                             datacol=datacol)
        y = self.prep_yaxis_data(xds_table_obj, ms_name, y_data, yaxis=yaxis,
                                 chan=chan, corr=corr, flag=flag)
        d = Data(y=y, ylabel=ylabel)
        return d


def save_png_image(name, disp_layout):
    """To save plots as png

    Note: One image will emerge

    """
    export_png(disp_layout, filename=name)


def add_axis(fig, axis_range, ax_label):
    """Add an extra axis to the current figure

    Input
    ------
    fig: Bokeh figure
         The figure onto which to add extra axis

    axis_range: ordered iterable
                A range of sorted values or a tuple with 2 values containing or the order (min, max)

    Output
    ------
    fig
    """
    fig.extra_x_ranges = {"fxtra": Range1d(
        start=axis_range[0], end=axis_range[-1])}
    linaxis = LinearAxis(x_range_name="fxtra", axis_label=ax_label,
                         minor_tick_line_alpha=0,
                         major_label_orientation='horizontal',
                         ticker=BasicTicker(desired_num_ticks=30))
    fig.add_layout(linaxis, 'above')
    return fig


def get_errors(xds_table_obj, corr=None):
    # CoONFIRM IF ITWORKS ACTUALLY
    #-------------------------------------------
    """Function to get error data from PARAMERR column.
    Inputs
    ------
    table_obj: pyrap table object.

    Outputs
    errors: ndarray
            Error data.
    """
    errors = xds_table_obj.PARAMERR
    return errors


def hv_plotter(x, y, xaxis, xlab='', yaxis='amplitude', ylab='',
               color='blue', xds_table_obj=None, ms_name=None, iterate=None):
    """
        Plotting with holoviews
    Input
    -----
        x:      xarray.DataArray
                x to plot
        y:      xarray.DataArray
                y to plot
        xaxis:  str
                xaxis to be plotted
        xlab:   str
                Label to appear on xaxis
        yaxis:  str
                yaxis to be plotted
        ylab:   str
                Label to appear i  yaxis
        iterate: str
                Iteration over axis
        ititle: str
                title to appear incasea of iteration
        color:  str, colormap, cycler object

        xds_table_obj: xarray.Dataset object
        ms_nmae: str
                 Path to measurement set

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

    def image_callback(xr, yr, w, h, x=None, y=None, cat=None, col=None):
        cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=xr, y_range=yr)
        if cat:
            agg = cvs.points(xy_df, x, y, ds.count_cat(cat))
            img = tf.shade(agg, color_key=col)
        else:
            agg = cvs.points(xy_df, x, y, ds.count())
            img = tf.shade(agg, cmap=col)
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
        bds = np.array([np.nanmin(x), np.nanmax(x)]).astype(datetime)
        # creating the x-xaxis label
        txaxis = "Time {} to {}".format(bds[0].strftime("%Y-%m-%d %H:%M:%S"),
                                        bds[-1].strftime("%Y-%m-%d %H:%M:%S"))
        # convert to milliseconds from epoch for bokeh
        x = x.astype('float64') * 1000
        x_axis_type = "datetime"
    else:
        x_axis_type = 'linear'

    x_min = np.nanmin(x)
    x_max = np.nanmax(x)

    y_min = np.nanmin(y) - np.nanstd(y)
    y_max = np.nanmax(y) + np.nanstd(y)

    if iterate:
        title = title + " Colorise By: {}".format(iterate.capitalize())
        if iterate == 'corr' or iterate == 'chan':
            xy = xa.merge([x, y])
        else:
            iter_data = xds_table_obj[iters[iterate]]
            xy = xa.merge([x, y, iter_data])
        # change value of iterate to the required data column
        iterate = iters[iterate]
        xy_df = xy.to_dask_dataframe()
        xy_df = xy_df.astype({iterate: 'category'})
    else:
        xy = xa.merge([x, y])
        xy_df = xy.to_dask_dataframe()

    fig = figure(tools='pan,box_zoom,wheel_zoom,reset,save',
                 x_range=(x_min, x_max), y_axis_label=ylab,
                 y_range=(y_min, y_max),
                 x_axis_type=x_axis_type, title=title,
                 plot_width=900, plot_height=700, sizing_mode='stretch_both')

    im = InteractiveImage(fig, image_callback, x=x_cap,
                          y=y_cap, cat=iterate,
                          col=color)

    h_tool = HoverTool(tooltips=[(x_cap, '$x'),
                                 (y_cap, '$y')])
    fig.add_tools(h_tool)

    fig.xaxis.axis_label = txaxis if xaxis == 'time' else xlab

    x_axis = fig.xaxis[0]
    x_axis.major_label_orientation = 45
    x_axis.ticker.desired_num_ticks = 30

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
    """Get argument parser"""

    x_choices = ['antenna1', 'antenna2',
                 'channel', 'frequency', 'phase',
                 'real', 'scan', 'time',
                 'uvdistance', 'uvwave']

    y_choices = ['amplitude', 'imaginary', 'phase', 'real']

    iter_choices = ['antenna1', 'antenna2', 'chan', 'corr', 'field', 'scan',
                    'spw']

    # TODO: make this arg parser inherit from ragavi the common options
    parser = ArgumentParser(usage='prog [options] <value>')

    parser.add_argument('--corr', dest='corr', type=str, metavar='',
                        help="""Correlation index or subset to plot Can be specified using normal python slicing syntax i.e "0:5" for 0<=corr<5 or "::2" for every 2nd corr or "0" for corr 0  or "0,1,3". Default is all.""",
                        default='0:')
    parser.add_argument('--chan', dest='chan', type=str, metavar='',
                        help="""Channels to select. Can be specified using syntax i.e "0:5" (exclusive range) or "20" for channel 20 or "10~20" (inclusive range) (same as 10:21) "::10" for every 10th channel or "0,1,3" etc. Default is all.""",
                        default=None)
    parser.add_argument('--cmap', dest='mycmap', type=str, metavar='',
                        help="""Colour or matplotlib colour map to use. Default is coolwarm.""",
                        default='coolwarm')
    parser.add_argument('--data_column', dest='data_column', type=str,
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
    parser.add_argument('--image_name', dest='image_name', type=str,
                        metavar='',
                        help="""Output png name. This requires works with phantomJS and selenium installed packages installed.""", default=None)
    parser.add_argument('--iterate', dest='iterate', type=str, metavar='',
                        choices=iter_choices,
                        help="""Select which variable to iterate over (defaults to none)""",
                        default=None)
    parser.add_argument('--no-flag', dest='flag', action='store_false',
                        help='Plot both flagged and unflagged data',
                        default=True)
    parser.add_argument('--scan', dest='scan', type=str, metavar='',
                        help='Scan Number to select. (default is all)',
                        default=None)
    parser.add_argument('--table', dest='mytabs',
                        nargs='*', type=str, metavar='',
                        help='Table(s) to plot (default = None)', default=[])
    parser.add_argument('--taql', dest='where', type=str, metavar='',
                        help='TAQL where', default=None)
    parser.add_argument('--xaxis', dest='xaxis', type=str, metavar='',
                        choices=x_choices, help='x-axis to plot',
                        default='time')
    parser.add_argument('--yaxis', dest='yaxis', type=str, metavar='',
                        choices=y_choices, help='Y axis variable to plot',
                        default='amplitude')

    return parser


def get_ms(ms_name, data_col='DATA', ddid=None, fid=None, scan=None, where=None):
    """
        Inputs
        ------
        ms_name: str
                 name of your MS or path including its name
        data_col: str
                  data column to be used
        ddid: int
              DATA_DESC_ID or spectral window to choose
        fid: int
             field id to select
        scan: int
              SCAN_NUMBER to select
        where: str
                TAQL where clause to be used with the MS.
        Outputs
        -------
        tab_objs: list
                  A list containing the specified table objects in xarray
    """

    ms_schema = MS_SCHEMA.copy()

    # defining part of the gain table schema
    if data_col != 'DATA':
        ms_schema[data_col] = ColumnSchema(dims=('chan', 'corr'))

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
        tab_objs = xm.xds_from_table(ms_name, taql_where=where,
                                     table_schema=ms_schema, group_cols=None)
        return tab_objs
    except:
        logger.exception(
            "Invalid DATA_DESC_ID, FIELD_ID, SCAN_NUMBER or TAQL clause")
        sys.exit(-1)


def main(**kwargs):

    if len(kwargs) == 0:
        NB_RENDER = False

        parser = get_argparser()
        options = parser.parse_args()

        chan = options.chan
        corr = options.corr
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

        chan = vu.slice_data(chan)

        corr = vu.slice_data(corr)

        partitions = get_ms(mytab, data_col=data_column, ddid=ddid,
                            fid=fields, scan=scan, where=where)

        mymap = cmx.get_cmap(mycmap)
        oup_a = []

        for count, chunk in enumerate(partitions):

            # black box returns an plottable element / composite element
            if iterate != None:
                mymap = cycle(['red', 'blue', 'green', 'purple'])

            logger.info("Starting data processing.")

            f = DataCoreProcessor(chunk, mytab, xaxis, yaxis, chan=chan,
                                  corr=corr, flag=flag, ddid=n_ddid,
                                  datacol=data_column)
            ready = f.act()

            logger.info("Starting plotting function.")

            fig = hv_plotter(ready.x, ready.y, xaxis=xaxis, xlab=ready.xlabel,             yaxis=yaxis, ylab=ready.ylabel, color=mymap,
                             iterate=iterate, xds_table_obj=chunk,
                             ms_name=mytab)

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
        logger.info(wrapper.fill("With arguments:\n" + str(options.__dict__)))
        logger.info(">" * (len(fname) + 19))

# for demo
if __name__ == '__main__':
    main()
