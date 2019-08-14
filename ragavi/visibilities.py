from __future__ import division

import sys
import glob
import re
import logging
import warnings

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import xarrayms as xm
import xarray as xa
import dask.array as da
import numpy as np
import holoviews as hv
import holoviews.operation.datashader as hd
import datashader as ds
import pyrap.quanta as qa

from holoviews import opts, dim
from africanus.averaging import time_and_channel as ntc
from collections import namedtuple
from dask import compute, delayed
from argparse import ArgumentParser
from datetime import datetime
from itertools import cycle
from xarrayms.known_table_schemas import MS_SCHEMA, ColumnSchema


from bokeh.plotting import figure
from bokeh.models.widgets import Div
from bokeh.layouts import row, column, gridplot, widgetbox
from bokeh.io import (output_file, show, output_notebook, export_svgs,
                      export_png, save)
from bokeh.models import (Range1d, HoverTool, ColumnDataSource, LinearAxis,
                          BasicTicker, Legend, Toggle, CustomJS, Title,
                          CheckboxGroup, Select, Text)

import vis_utils as vu
from ipdb import set_trace


logger = vu.logger
excepthook = vu.sys.excepthook


class DataCoreProcessor:

    def __init__(self, xds_table_obj, ms_name, xaxis, yaxis,
                 chan=slice(0, None), corr=0, ddid=0, datacol='DATA',
                 flag=True, iterate=None):

        self.xds_table_obj = xds_table_obj
        self.ms_name = ms_name
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.corr = corr
        self.ddid = ddid
        self.chan = chan
        self.datacol = datacol
        self.flag = flag
        self.iterate = iterate

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
            if xaxis == 'channel':
                x_label = 'Channel'
            else:
                x_label = 'Frequency GHz'
        elif xaxis == 'phase':
            xdata = xds_table_obj[datacol]
            x_label = 'Phase'
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
            x_label = 'UV Wave [lambda]'
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
            y_label = 'Phase[deg]'
        elif yaxis == 'real':
            y_label = 'Real'

        try:
            ydata = xds_table_obj[datacol]
        except KeyError:
            logger.exception('Column "{}" not Found'.format(datacol))
            return sys.exit(-1)

        return ydata, y_label

    def prep_xaxis_data(self, xdata, freq=None, xaxis='time'):
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
        if xaxis == 'channel':
            prepdx = xdata
        elif xaxis == 'frequency':
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
                        chan=slice(0, None), corr=0, flag=True, iterate=None,
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
            iterate: str
                     Data to iterate over.

            Outputs
            -------
            y: xarray DataArray
               Processed yaxis data.
        """
        if iterate == 'corr':
            # group data by correlations and store it in a list
            # groupby list returns the tuple (group_idx, group_item_DA)
            ydata = list(ydata.groupby('corr'))
            ydata = [x[1].sel(chan=chan) for x in ydata]

            # group flags for that data as well and store in a list
            flags = vu.get_flags(xds_table_obj).groupby('corr')
            flags = [x[1].sel(chan=chan) for x in flags]
        else:
            ydata = ydata.sel(dict(corr=corr, chan=chan))
            flags = vu.get_flags(xds_table_obj).sel(dict(corr=corr, chan=chan))

        # if flagging enabled return a list of DataArrays otherwise return a
        # single dataarray
        if flag:
            if isinstance(ydata, list):
                y = []
                for d, f in zip(ydata, flags):
                    processed = self.process_data(d, yaxis=yaxis, wrap=True)
                    # replace with NaNs where flags is not 1
                    y.append(processed.where(f == False))
            else:
                processed = self.process_data(ydata, yaxis=yaxis, wrap=True)
                y = processed.where(flags == False)
        else:
            if isinstance(ydata, list):
                y = []
                for d in ydata:
                    y.append(self.process_data(d, yaxis=yaxis))
            else:
                y = self.process_data(ydata, yaxis=yaxis)

        return y

    def blackbox(self, xds_table_obj, ms_name, xaxis, yaxis,
                 chan=slice(0, None), corr=0, datacol='DATA', ddid=None,
                 flag=True, iterate=None, ititle=None):
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
            ititle: str
                    Title to use incase of an iteration
            iterate: str
                     Paramater over which to iterate
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
                y_prepd = y_prepd.transpose('chan', 'row', 'corr',
                                            transpose_coords=True)
            else:
                y_prepd = y_prepd.T
        """
        if xaxis == 'uvwave':
            # compute uvwave using the available frequencies
            freqs = vu.get_frequencies(ms_name, spwid=ddid).compute()
            x_prepd = prep_xaxis_data(x_data, xaxis, freq=freqs)
        elif xaxis == 'real' or xaxis == 'phase':
            x_prepd = prep_yaxis_data(xds_table_obj, ms_name, x_data,
                                      yaxis=xaxis, corr=corr, chan=chan,
                                      flag=flag)
        else:
            x_prepd = prep_xaxis_data(x_data, xaxis)
        """

        d = Data(x=x_prepd, xlabel=xlabel, y=y_prepd, ylabel=ylabel)

        return d

    def act(self):
        return self.blackbox(self.xds_table_obj, self.ms_name, self.xaxis,
                             self.yaxis, chan=self.chan, corr=self.corr,
                             datacol=self.datacol, ddid=self.ddid,
                             flag=self.flag, iterate=self.iterate)

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
            x = self.prep_xaxis_data(x_data, xaxis=xaxis)
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


def save_svg_image(img_name, figa, figb, glax1, glax2):
    """To save plots as svg

    Note: Two images will emerge

    Inputs
    ------
    img_name: string
              Desired image name
    figa: figure object
          First figure
    figb: figure object
          Second figure
    glax1: list
           Contains glyph metadata saved during execution
    glax2: list
           Contains glyph metadata saved during execution

    Outputs
    -------
    Returns nothing

    """
    for i in range(len(glax1)):
        glax1[i][1][0].visible = True
        glax2[i][1][0].visible = True

    figa.output_backend = "svg"
    figb.output_backend = "svg"
    export_svgs([figa], filename="{:s}_{:s}".format(img_name, "a.svg"))
    export_svgs([figb], filename="{:s}_{:s}".format(img_name, "b.svg"))


def save_png_image(img_name, disp_layout):
    """To save plots as png

    Note: One image will emerge

    """
    export_png(img_name, disp_layout)


def errorbar(fig, x, y, xerr=None, yerr=None, color='red', point_kwargs={}, error_kwargs={}):
    """Function to plot the error bars for both x and y.
       Takes in 3 compulsory parameters fig, x and y

    Inputs
    ------
    fig: the figure object
    x: numpy.ndarray
        x_axis value
    y: numpy.ndarray
        y_axis value
    xerr: numpy.ndarray
        Errors for x axis, must be an array
    yerr: numpy.ndarray
        Errors for y axis, must be an array
    color: str
        Color for the error bars


    Outputs
    -------
    h: fig.multi_line
        Returns a multiline object for external legend rendering

    """
    # Setting default return value
    h = None

    if xerr is not None:

        x_err_x = []
        x_err_y = []

        for px, py, err in zip(x, y, xerr):
            x_err_x.append((px - err, px + err))
            x_err_y.append((py, py))

        h = fig.multi_line(x_err_x, x_err_y, color=color, line_width=3,
                           level='underlay', visible=False, **error_kwargs)

    if yerr is not None:
        y_err_x = []
        y_err_y = []

        for px, py, err in zip(x, y, yerr):
            y_err_x.append((px, px))
            y_err_y.append((py - err, py + err))

        h = fig.multi_line(y_err_x, y_err_y, color=color, line_width=3,
                           level='underlay', visible=False, **error_kwargs)

    fig.legend.click_policy = 'hide'

    return h


def add_axis(fig, axis_range, ax_label):
    """Add an extra axis to the current figure

    Input
    ------
    fig: Bokeh figure
         The figure onto which to add extra axis

    axis_range: tuple
                Starting and ending point for the range

    Output
    ------
    Nothing
    """
    fig.extra_x_ranges = {"fxtra": Range1d(
        start=axis_range[0], end=axis_range[-1])}
    linaxis = LinearAxis(x_range_name="fxtra", axis_label=ax_label,
                         major_label_orientation='horizontal', ticker=BasicTicker(desired_num_ticks=12))
    return linaxis


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
               iterate=None, ititle=None, color='blue',
               xds_table_obj=None, ms_name=None):
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
    hv.extension('bokeh', logo=False)
    w = 900
    h = 700

    ys = {'amplitude': [('Amplitude', 'Amplitude')],
          'phase': [('Phase', 'Phase (deg)')],
          'real': [('Real', 'Real')],
          'imaginary': [('Imaginary', 'Imaginary')]}

    xs = {'time': [('Time', 'Time')],
          'scan': [('Scan', 'Scan')],
          'antenna1': [('Antenna1', 'Ant1')],
          'antenna2': [('Antenna2', 'Ant2')],
          'uvdistance': [('Uvdistance', 'UVDistance (m)')],
          'uvwave': [('Uvwave', 'UVWave (lambda)')],
          'frequency': [('Frequency', 'Frequency GHz')],
          'channel': [('Channel', 'Channel')],
          'real': [('Real', 'Real')],
          'phase': [('Phase', 'Phase (deg)')]
          }

    # changing the Name of the dataArray to the name provided as xaxis
    x.name = xaxis.capitalize()

    if isinstance(y, list):
        for i in y:
            i.name = yaxis.capitalize()
    else:
        y.name = yaxis.capitalize()

    if xaxis == 'channel' or xaxis == 'frequency':
        y = y.assign_coords(table_row=x.table_row)
        xma = x.max().data.compute()
        xmi = x.min().data.compute()

        def twinx(plot, element):
            # Setting the second y axis range name and range
            start, end = (element.range(1))
            label = element.dimensions()[1].pprint_label
            plot.state.extra_y_ranges = {"foo": Range1d(start=xmi, end=xma)}
            # Adding the second axis to the plot.
            linaxis = LinearAxis(axis_label='% Channel', y_range_name='foo')
            plot.state.add_layout(linaxis, 'above')

    # setting dependent variables in the data
    vdims = ys[yaxis]
    kdims = xs[xaxis]

    # by default, colorize over baseline
    if iterate == None:
        title = "{} vs {}".format(ylab, xlab)
        res_ds = xa.merge([x, y])
        res_df = res_ds.to_dask_dataframe()
        res_hds = hv.Dataset(res_df, kdims, vdims)
        # res_hds = hv.NdOverlay(res_hds)
        ax = hd.datashade(res_hds, dynamic=False,
                          cmap=color).opts(title=title,
                                           width=w,
                                           height=h,
                                           fontsize={'title': 20,
                                                     'labels': 16})

    else:
        title = "Iteration over {}: {} vs {}".format(iterate, ylab, xlab)
        if iterate == 'scan':
            scans = get_xaxis_data(xds_table_obj, ms_name, 'scan')[0]
            kdims = kdims + xs['scan']
            res_ds = xa.merge([x, y, scans])
            res_df = res_ds.to_dask_dataframe()
            res_hds = hv.Dataset(res_df, kdims, vdims).groupby('Scan')
            res_hds = hv.NdOverlay(res_hds)
            ax = hd.datashade(res_hds, dynamic=False, color_key=color,
                              aggregator=ds.count_cat('Scan')
                              ).opts(title=title, width=w,
                                     height=h, fontsize={'title': 20,
                                                         'labels': 16})
        elif iterate == 'spw':
            pass

        elif iterate == 'corr':
            outs = []
            for item in y:
                res_ds = xa.merge([x, item])
                res_df = res_ds.to_dask_dataframe()
                res_hds = hv.Dataset(res_df, kdims, vdims)

                outs.append(hd.datashade(res_hds,
                                         dynamic=False,
                                         cmap=color.next()))

            ax = hv.Overlay(outs)
            ax.opts(title=title, width=w, height=h, fontsize={'title': 20,
                                                              'labels': 16})

    if xaxis == 'time':
        mint = np.nanmin(x)
        mint = mint.astype(datetime).strftime("%Y-%m-%d %H:%M:%S'")
        maxt = np.nanmax(x)
        maxt = maxt.astype(datetime).strftime("%Y-%m-%d %H:%M:%S'")

        xlab = "Time [{} to {}]".format(mint, maxt)
        timeax_opts = {'xrotation': 45,
                       'xticks': 30,
                       'xlabel': xlab}
        ax.opts(**timeax_opts)

    return ax

'''
def stats_display(table_obj, gtype, ptype, corr, field):
    """Function to display some statistics on the plots. These statistics are derived from a specific correlation and a specified field of the data.
    Currently, only the medians of these plots are displayed.

    Inputs
    ------
    table_obj: pyrap table object
    gtype: str
           Type of gain table to be plotted.
    ptype: str
            Type of plot ap / ri
    corr: int
          Correlation number of the data to be displayed
    field: int
            Integer field id of the field being plotted. If a string name was provided, it is converted within the main function by the name2id function.

    Outputs
    -------
    pre: Bokeh widget model object
         Preformatted text containing the medians for both model. The object returned must then be placed within the widget box for display.

    """
    subtable = table_obj.query(query="FIELD_ID=={}".format(field))
    ydata, y1label, y2label = get_yaxis_data(subtable, gtype, ptype)

    flags = vu.get_flags(subtable)[:, :, corr]
    ydata = ydata[:, :, corr]
    m_ydata = np.ma.masked_array(data=ydata, mask=flags)

    if ptype == 'ap':
        y1 = np.ma.abs(m_ydata)
        y2 = np.unwrap(np.ma.angle(m_ydata, deg=True))
        med_y1 = np.ma.median(y1)
        med_y2 = np.ma.median(y2)
        text = "Median Amplitude: {}\nMedian Phase: {} deg".format(
            med_y1, med_y2)
        if gtype == 'K':
            text = "Median Amplitude: {}".format(med_y1)
    else:
        y1 = np.ma.real(m_ydata)
        y2 = np.ma.imag(m_ydata)

        med_y1 = np.ma.median(y1)
        med_y2 = np.ma.median(y2)
        text = "Median Real: {}\nMedian Imaginary: {}".format(med_y1, med_y2)

    pre = PreText(text=text)

    return pre

'''


def get_argparser():
    """Get argument parser"""

    x_choices = ['antenna1', 'antenna2',
                 'channel', 'frequency', 'phase',
                 'real', 'scan', 'time',
                 'uvdistance', 'uvwave']

    y_choices = ['amplitude', 'imaginary', 'phase', 'real']

    iter_choices = ['scan', 'corr', 'spw']

    # TODO: make this arg parser inherit from ragavi the common options
    parser = ArgumentParser(usage='prog [options] <value>')

    parser.add_argument('-c', '--corr', dest='corr', type=str, metavar='',
                        help="""Correlation index or subset to plot Can be specified using normal python slicing syntax i.e 0:5 for 0<=corr<5 or ::2 for every 2nd corr or 0 for corr 0 etc. Default is all""",
                        default='0')
    parser.add_argument('--chan', dest='chan', type=str, metavar='',
                        help="""Channels to select. Can be specified using syntax i.e 0:5  (or 0~4) for 0<=channel<5 or 20 for channel 20 or 10~20 (10<=channel<21) (same as 10:21) ::10 for every 10th channel etc. Default is all""",
                        default=None)
    parser.add_argument('--cmap', dest='mycmap', type=str, metavar='',
                        help="""Colour or matplotlib colour map to use(default=coolwarm )""",
                        default='coolwarm')
    parser.add_argument('--data_column', dest='data_column', type=str,
                        metavar='',
                        help='Column from MS to use for data', default='DATA')
    parser.add_argument('--ddid', dest='ddid', type=str,
                        metavar='',
                        help="""DATA_DESC_ID(s) to select. (default is all) Can be specified as e.g. "5", "5,6,7", "5~7" (inclusive range), "5:8" (exclusive range), 5:(from 5 to last)""",
                        default=None)
    parser.add_argument('-f', '--field', dest='fields', type=str,
                        metavar='',
                        help='Field ID(s) / NAME(s) to plot (Default all)',
                        default=None)
    parser.add_argument('--flag', dest='flag', action='store_true',
                        help='Plot only unflagged data',
                        default=True)
    parser.add_argument('--no-flag', dest='flag', action='store_false',
                        help='Plot both flagged and unflagged data',
                        default=True)
    parser.add_argument('--htmlname', dest='html_name', type=str, metavar='',
                        help='Output HTMLfile name', default='')
    parser.add_argument('--iterate', dest='iterate', type=str, metavar='',
                        choices=iter_choices,
                        help="""Select which variable to iterate over (defaults to none)""",
                        default=None)
    parser.add_argument('--scan', dest='scan', type=str, metavar='',
                        help='Scan Number to select. (default is all)',
                        default=None)
    parser.add_argument('-t', '--table', dest='mytabs',
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
        # image_name = options.image_name
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
            else:
                fields = vu.name_2id(mytab, fields)
                fields = vu.resolve_ranges(fields)

        if scan != None:
            scan = vu.resolve_ranges(scan)

        if ddid != None:
            ddid = vu.resolve_ranges(ddid)

        chan = vu.slice_data(chan)

        corr = vu.slice_data(corr)

        partitions = get_ms(mytab, data_col=data_column, ddid=ddid,
                            fid=fields, scan=scan, where=where)

        mymap = cmx.get_cmap(mycmap)
        oup_a = []

        for count, chunk in enumerate(partitions):

            colour = mymap

            # black box returns an plottable element / composite element
            if iterate != None:
                title = "TEST"
                colour = cycle(['red', 'blue', 'green', 'purple'])

                f = DataCoreProcessor(chunk, mytab, xaxis, yaxis, corr=corr,
                                      chan=chan, ddid=ddid, flag=flag,
                                      iterate=iterate, datacol=data_column)

            else:
                ititle = None
                f = DataCoreProcessor(chunk, mytab, xaxis, yaxis, chan=chan,
                                      corr=corr, flag=flag, iterate=iterate,
                                      datacol=data_column)

            ready = f.act()
            fig = hv_plotter(ready.x, ready.y, xaxis=xaxis, xlab=ready.xlabel, yaxis=yaxis, ylab=ready.ylabel,
                             ititle=ititle, color=colour, iterate=iterate, xds_table_obj=chunk, ms_name=mytab)
            # store resulting dynamicmaps
            oup_a.append(fig)

        try:
            l = hv.Overlay(oup_a).collate().opts(width=900, height=700)
        except TypeError:
            logger.error(
                'Unable to plot; Specified FIELD_IDs or DATA_DESC_ID(s) or SCAN_NUMBER(s) not found.')
            sys.exit(-1)

        layout = hv.Layout([l])

        fname = "{}_{}.html".format(yaxis, xaxis)

        if html_name != None:
            fname = html_name + '_' + fname
        hv.save(layout, fname)

# for demo
if __name__ == '__main__':
    main()
