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

from holoviews import opts, dim
from africanus.averaging import time_and_channel as ntc
from dask import compute, delayed
import pyrap.quanta as qa
from argparse import ArgumentParser
from datetime import datetime
from itertools import cycle


from bokeh.plotting import figure
from bokeh.models.widgets import Div
from bokeh.layouts import row, column, gridplot, widgetbox
from bokeh.io import (output_file, show, output_notebook, export_svgs,
                      export_png, save)
from bokeh.models import (Range1d, HoverTool, ColumnDataSource, LinearAxis,
                          BasicTicker, Legend, Toggle, CustomJS, Title,
                          CheckboxGroup, Select, Text)


def config_logger():
    """This function is used to configure the logger for ragavi and catch
        all warnings output by sys.stdout.
    """
    logfile_name = 'ragavi.log'
    # capture only a single instance of a matching repeated warning
    warnings.filterwarnings('default')

    # setting the format for the logging messages
    start = " (O_o) ".center(80, "=")
    form = '{}\n%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    form = form.format(start)
    formatter = logging.Formatter(form, datefmt='%d.%m.%Y@%H:%M:%S')

    # setup for ragavi logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # capture all stdout warnings
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger('py.warnings')
    warnings_logger.setLevel(logging.DEBUG)

    # setup for logfile handing ragavi
    fh = logging.FileHandler(logfile_name)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    warnings_logger.addHandler(logger)
    return logger


def _handle_uncaught_exceptions(extype, exval, extraceback):
    """Function to Capture all uncaught exceptions into the log file

       Inputs to this function are acquired from sys.excepthook. This
       is because this function overrides sys.excepthook 

       https://docs.python.org/3/library/sys.html#sys.excepthook

    """
    message = "Oops ... !"
    logger.error(message, exc_info=(extype, exval, extraceback))


logger = config_logger()
sys.excepthook = _handle_uncaught_exceptions


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


def name_2id(val, dic):
    """Translate field name to field id

    Inputs
    -----
    val: string
         Field ID name to convert
    dic: dict
         Dictionary containing enumerated source ID names

    Outputs
    -------
    key: int
         Integer field id
    """
    upperfy = lambda x: x.upper()
    values = dic.values()
    values = [upperfy(x) for x in values]
    val = val.upper()

    if val in values:
        val_index = values.index(val)
        keys = [x for x in dic.keys()]

        # get the key to that index from the key values
        key = keys[val_index]
        return int(key)
    else:
        return -1


def get_polarizations(ms_name):

    # Stokes types in this case are 1 based and NOT 0 based.
    stokes_types = ['I', 'Q', 'U', 'V', 'RR', 'RL', 'LR', 'LL', 'XX', 'XY', 'YX', 'YY', 'RX', 'RY', 'LX', 'LY', 'XR', 'XL', 'YR',
                    'YL', 'PP', 'PQ', 'QP', 'QQ', 'RCircular', 'LCircular', 'Linear', 'Ptotal', 'Plinear', 'PFtotal', 'PFlinear', 'Pangle']
    subname = "::".join((ms_name, 'POLARIZATION'))
    pol_subtable = list(xm.xds_from_table(subname, ack=False))
    pol_subtable = pol_subtable[0]
    # ofsetting the acquired corr typeby one to match correctly the stokes type
    corr_types = pol_subtable.CORR_TYPE.sel(row=0).data.compute() - 1
    cor2stokes = []
    for typ in corr_types:
        cor2stokes.append(stokes_types[typ])
    return cor2stokes


def get_frequencies(ms_name, spwid=0):
    """Function to get channel frequencies from the SPECTRAL_WINDOW subtable.
    Inputs
    ------
    table_obj: pyrap table object
    spwid: int
           Spectral window id number. Defaults to 0

    Outputs
    -------
    freqs: xarray.core.dataarray.DataArray
           Channel centre frequencies for specified spectral window.
    """
    subname = "::".join((ms_name, 'SPECTRAL_WINDOW'))
    spw_subtab = list(xm.xds_from_table(subname, group_cols='__row__',
                                        ack=False))
    spw = spw_subtab[spwid]
    freqs = spw.CHAN_FREQ
    # spw_subtab('close')
    return freqs


def get_antennas(ms_name):
    """Function to get antennae names from the ANTENNA subtable.
    Inputs
    ------
    table_obj: pyrap table object

    Outputs
    -------
    ant_names: xarray.core.dataarray.DataArray
               Names for all the antennas available.

    """
    subname = "::".join((ms_name, 'ANTENNA'))
    ant_subtab = list(xm.xds_from_table(subname, ack=False))
    ant_subtab = ant_subtab[0]
    ant_names = ant_subtab.NAME
    # ant_subtab('close')
    return ant_names


def get_flags(xds_table_obj, corr=None):
    """Function to get Flag values from the FLAG column
    Allows the selection of flags for a single correlation. If none is specified the entire data is then selected.
    Inputs
    ------
    table_obj: pyrap table object
    corr: int
          Correlation number to select.

    Outputs
    -------
    flags: xarray.core.dataarray.DataArray
           Array containing selected flag values.

    """
    flags = xds_table_obj.FLAG
    if corr is None:
        return flags
    else:
        flags = flags.sel(corr=corr)
    return flags


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


def get_fields(ms_name):
    """Function to get field names from the FIELD subtable.
    Inputs
    ------
    table_obj: pyrap table object

    Outputs
    -------
    field_names: xarray.core.dataarray.DataArray
                 String names for the available data in the table
    """
    subname = "::".join((ms_name, 'FIELD'))
    field_subtab = list(xm.xds_from_table(subname, ack=False))
    field_subtab = field_subtab[0]
    field_names = field_subtab.NAME
    return field_names


def calc_uvdist(uvw):
    """ Function to Calculate uv distance in metres
    Inputs
    ------
    uvw: xarray.core.dataarray.DataArray

    Outputs
    -------
    uvdist: xarray.core.dataarray.DataArray
            uv distance in meters
    """
    u = uvw.isel(**{'(u,v,w)': 0})
    v = uvw.isel(**{'(u,v,w)': 1})
    uvdist = xa.ufuncs.sqrt(xa.ufuncs.square(u) + xa.ufuncs.square(v))
    return uvdist


def calc_uvwave(uvw, freq):
    """
        Calculate uv distance in wavelength for availed frequency.
        This function also calculates the corresponding wavelength.

    Inputs
    ------
    uvw: xarray.core.dataarray.DataArray
    freq: float
          Frequency from which corresponding wavelength will be obtained.

    Outputs
    -------
    uvwave: xarray.core.dataarray.DataArray
            uv distance in wavelength for specific frequency
    """

    # speed of light
    C = 3e8

    # wavelength = velocity / frequency
    wavelength = (C / freq)
    uvdist = calc_uvdist(uvw)

    uvwave = uvdist / wavelength
    return uvwave


def time_convert(xdata):
    # get first time instance
    init_time = xdata[0].data.compute()

    # get number of seconds from initial time
    time = xdata - init_time

    # convert the initial time to unix time in seconds
    init_time = qa.quantity(init_time, 's').to_unix_time()

    # Add the initial unix time in seconds to get actual time progrssion
    newtime = time + init_time

    newtime = da.array(newtime, dtype='datetime64[s]')
    return newtime


def get_xaxis_data(xds_table_obj, ms_name, xaxis):
    """Function to get x-axis data. It is dependent on the gaintype.
        This function also returns the relevant x-axis labels for both pairs of plots.
    Inputs
    ------
    table_obj: pyrap table object

    gtype:  str
            Type of gain table being plotted.

    Outputs
    -------
    xdata: ndarray
           X-axis data depending on the gain table to be plotted.
    xaxis_label: str
                 Label to appear on the x-axis of the plots. This is shared amongst both plots.
    """
    if xaxis == 'antenna1':
        xdata = xds_table_obj.ANTENNA1
        xaxis_label = 'Antenna1'
    elif xaxis == 'antenna2':
        xdata = xds_table_obj.ANTENNA2
        xaxis_label = 'Antenna2'
    elif xaxis == 'frequency' or xaxis == 'channel':
        xdata = get_frequencies(ms_name)
        if xaxis == 'channel':
            xaxis_label = 'Channel'
        else:
            xaxis_label = 'Frequency GHz'
    elif xaxis == 'phase':
        xdata = xds_table_obj.DATA
        xaxis_label = 'Phase (Deg)'
    elif xaxis == 'scan':
        xdata = xds_table_obj.SCAN_NUMBER
        xaxis_label = 'Scan'
    elif xaxis == 'time':
        xdata = xds_table_obj.TIME
        xaxis_label = 'Time [s]'
    elif xaxis == 'uvdistance':
        xdata = xds_table_obj.UVW
        xaxis_label = 'UV Distance [m]'
    elif xaxis == 'uvwave':
        xdata = xds_table_obj.UVW
        xaxis_label = 'UV Wave [lambda]'
    else:
        print("Invalid xaxis name")
        return

    return xdata, xaxis_label


def prep_xaxis_data(xdata, xaxis, freq=None):
    """Function to Prepare the x-axis data.
    Inputs
    ------
    xdata: 1-D array
           Data for the xaxis to be prepared
    freq: float
          REQUIRED ONLY when xaxis specified is 'uvwave'. In this case.
          this function must be the called within a loop containing all
          the frequencies available in  a spectral window.

    Outputs
    -------
    prepdx: 1-D array
            Data for the x-axid of the plots
    """
    if xaxis == 'channel':
        prepdx = xdata.chan
    elif xaxis == 'frequency':
        prepdx = xdata / 1e9
    elif xaxis == 'phase':
        prepdx = get_phase(xdata, deg=True)
    elif xaxis == 'time':
        prepdx = time_convert(xdata)
    elif xaxis == 'uvdistance':
        prepdx = calc_uvdist(xdata)
    elif xaxis == 'uvwave':
        prepdx = calc_uvwave(xdata, freq)
    elif xaxis == 'antenna1' or xaxis == 'antenna2' or xaxis == 'scan':
        prepdx = xdata
    elif xaxis == 'phase':
        # TODO: add clause for phase. It must be processed
        pass
    return prepdx


def get_yaxis_data(xds_table_obj, ms_name, yaxis):
    """ Function to extract the required column for the y-axis data.
    This column is determined by ptype which can be amplitude vs phase 'ap'
    or real vs imaginary 'ri'.

    Inputs
    -----
    table_obj: python casacore table object
               Table in which to get the data

    gtype: str
           Gain table type B, F, G or K.

    yaxis: str
           Plot type ap / ri

    Outputs
    -------
    Returns np.ndarray dataa as well as the y-axis labels (str) for both plots.
    """
    if yaxis == 'amplitude':
        y_label = 'Amplitude'
    elif yaxis == 'imaginary':
        y_label = 'Imaginary'
    elif yaxis == 'phase':
        y_label = 'Phase[deg]'
    elif yaxis == 'real':
        y_label = 'Real'

    ydata = xds_table_obj.DATA
    return ydata, y_label


def prep_yaxis_data(xds_table_obj, ms_name, ydata, yaxis='amplitude', corr=0, flag=False, iterate=None):
    """Function to process data for the y-axis. Part of the processing includes:
    - Selecting correlation for the data and error
    - Flagging
    - Complex correlation parameter conversion to amplitude, phase, real and
      imaginary for processing
    Data selection and flagging are done by this function itself, however ap and ri conversion are done by specified functions.

    Inputs
    ------
    table_obj: pyrap table object
               table object for an already open table
    ydata: ndarray
           Relevant y-axis data to be processed
    yaxis: str
           Plot type 'ap' / 'ri'
    corr: int
          Correlation number to select
    flag: bool
          Option on whether to flag the data or not

    Outputs
    -------
    y: masked ndarray
        Amplitude / real part of the complex input data.
    y_err: masked ndarray
        Error data for y1.

    """
    if iterate == 'corr':
        ydata = list(ydata.groupby('corr'))
        ydata = [x[1] for x in ydata]
        flags = get_flags(xds_table_obj).groupby('corr')
        flags = [x[1] for x in flags]
    else:
        ydata = ydata.sel(corr=corr)
        flags = get_flags(xds_table_obj).sel(corr=corr)

    # if flagging enabled return a list otherwise return a single dataarray
    if flag:
        if isinstance(ydata, list):
            y = []
            for d, f in zip(ydata, flags):
                masked = da.ma.masked_array(data=d, mask=f)
                y.append(process_data(masked, yaxis=yaxis))
        else:
            masked = da.ma.masked_array(data=ydata, mask=flags)
            y = process_data(masked, yaxis=yaxis)
    else:
        if isinstance(ydata, list):
            y = []
            for d in ydata:
                y.append(process_data(d, yaxis=yaxis))
        else:
            y = process_data(ydata, yaxis=yaxis)

    return y


def get_phase(ydata, unwrap=False):
    phase = xa.ufuncs.angle(ydata, deg=True)
    if unwrap:
        # delay dispatching of unwrapped phase
        phase = delayed(np.unwrap)(phase)
    return phase


def get_amplitude(ydata):
    amplitude = da.absolute(ydata)
    return amplitude


def get_real(ydata):
    real = ydata.real
    return real


def get_imaginary(ydata):
    imag = ydata.imag
    return imag


def process_data(ydata, yaxis):
    if yaxis == 'amplitude':
        y = get_amplitude(ydata)
    elif yaxis == 'imaginary':
        y = get_imaginary(ydata)
    elif yaxis == 'phase':
        y = get_phase(ydata)
    elif yaxis == 'real':
        y = get_real(ydata)
    return y


def blackbox(xds_table_obj, ms_name, xaxis, yaxis, corr, showFlagged=False, ititle=None, color='blue', iterate=None):
    x_data, xlabel = get_xaxis_data(xds_table_obj, ms_name, xaxis)
    y_data, ylabel = get_yaxis_data(xds_table_obj, ms_name, yaxis)
    y_prepd = prep_yaxis_data(xds_table_obj, ms_name, y_data,
                              yaxis=yaxis, corr=corr,
                              flag=showFlagged, iterate=iterate)
    if xaxis == 'channel' or xaxis == 'frequency':
        y_prepd = y_prepd.transpose()
        y_prepd = y_prepd.transpose()
    #fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(40, 20))

    # plt.tight_layout()
    if xaxis == 'uvwave':
        freqs = get_frequencies(ms_name).compute()
        x_prepd = prep_xaxis_data(x_data, xaxis, freq=freqs)
    else:
        x_prepd = prep_xaxis_data(x_data, xaxis)
        # for bokeh
        # for mpl
    #f1, f2 = mpl_plotter(ax1, ax2, x_prepd, y1_prepd, y2_prepd)
    """mpl_plotter(ax1, ax2, x_prepd, y1_prepd, y2_prepd, xaxis, xlab=xlabel,
                            yaxis=yaxis, y1lab=y1label, y2lab=y2label, ititle=ititle, color=color)"""
    fig = hv_plotter(x_prepd, y_prepd, xaxis, xlab=xlabel,
                     yaxis=yaxis, ylab=ylabel, ititle=ititle,
                     color=color, iterate=iterate,
                     xds_table_obj=xds_table_obj, ms_name=ms_name)
    return fig


def mpl_plotter(ax1, x, y1, y2, xaxis, xlab='', yaxis='amplitude',
                ylab='', ititle=None, color='blue'):
    x, y = compute(x, y)
    ax1.plot(x, y1, marker='o', markersize=0.5, linewidth=0, color=color)

    ax1.set_xlabel(xlab)
    ax1.set_ylabel(y1lab)

    if ititle is not None:
        ax1.set_title("{}: {} vs {}".format(ititle, y1lab, xlab))
        plt.savefig(fname="{}_{}_{}".format(ititle, ptype, xaxis))
    else:
        ax1.set_title("{} vs {}".format(y1lab, xlab))
        plt.savefig(fname="{}_{}".format(ptype, xaxis))
    plt.close('all')

    return None
    # return ax1


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
          'uvdistance': [('Uvdist', 'UVDistance (m)')],
          'uvwave': [('Uvwave', 'UVDistance (lambda)')],
          'frequency': [('Frequency', 'Frequency GHz')],
          'channel': [('Channel', 'Channel')],
          'phase': [('Phase', 'Phase (deg)')]
          }

    # iteration groups
    baseline = ['Antenna1', 'Antenna2']
    grps = ['SCAN_NUMBER']

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

    # get grouping data
    scans = get_xaxis_data(xds_table_obj, ms_name, 'scan')[0]
    scans.name = 'Scan'
    ant1 = get_xaxis_data(xds_table_obj, ms_name, 'antenna1')[0]
    ant1.name = 'Antenna1'
    ant2 = get_xaxis_data(xds_table_obj, ms_name, 'antenna2')[0]
    ant2.name = 'Antenna2'

    # by default, colorize over baseline
    if iterate == None:
        title = "{} vs {}".format(ylab, xlab)
        res_ds = xa.merge([x, y, ant1, ant2])
        res_df = res_ds.to_dask_dataframe()
        res_hds = hv.Dataset(res_df, kdims, vdims)
        #res_hds = hv.NdOverlay(res_hds)
        ax = hd.datashade(res_hds, dynamic=False,
                          cmap=color).opts(title=title,
                                           width=w,
                                           height=h,
                                           fontsize={'title': 20,
                                                     'labels': 16})

    else:
        title = "Iteration over {}: {} vs {}".format(iterate, ylab, xlab)
        if iterate == 'scan':
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
            grp = grps[2]
            pass

        elif iterate == 'corr':
            outs = []
            for c, item in enumerate(y):
                res_ds = xa.merge([x, item])
                res_df = res_ds.to_dask_dataframe()
                res_hds = hv.Dataset(res_df, kdims, vdims)

                outs.append(hd.datashade(res_hds,
                                         dynamic=False,
                                         cmap=color.values[c]))

            ax = hv.Overlay(outs)
            ax.opts(title=title, width=w, height=h, fontsize={'title': 20,
                                                              'labels': 16})

    if xaxis == 'time':

        mint = x.min().data.compute()
        mint = mint.astype(datetime).strftime("%Y-%m-%d %H:%M:%S'")
        maxt = x.max().data.compute()
        maxt = maxt.astype(datetime).strftime("%Y-%m-%d %H:%M:%S'")

        xlab = "Time [{} to {}]".format(mint, maxt)
        timeax_opts = {'xrotation': 45,
                       'xticks': 30,
                       'xlabel': xlab}
        ax.opts(**timeax_opts)

    return ax


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

    flags = get_flags(subtable)[:, :, corr]
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


def get_argparser():
    """Get argument parser"""

    x_choices = ['antenna1', 'antenna2',
                 'channel', 'frequency',
                 'phase', 'scan', 'time']

    y_choices = ['amplitude', 'imaginary', 'phase', 'real']

    iter_choices = ['scan', 'corr', 'spw']

    # TODO: make this arg parser inherit from ragavi the common options
    parser = ArgumentParser(usage='prog [options] <value>')
    """
    parser.add_argument('-a', '--ant', dest='plotants', type=str,
                        help='Plot only this antenna, or comma-separated list\
                              of antennas',
                        default=[-1])
    """
    parser.add_argument('-c', '--corr', dest='corr', type=int,
                        help='Correlation index to plot (usually just 0 or 1,\
                              default = 0)',
                        default=0)
    parser.add_argument('--cmap', dest='mycmap', type=str,
                        help='Matplotlib colour map to use for antennas\
                             (default=coolwarm)',
                        default='coolwarm')
    parser.add_argument('--yaxis', dest='yaxis', type=str,
                        choices=y_choices, help='Y axis variable to plot',
                        default='amplitude')
    parser.add_argument('-f', '--field', dest='fields', nargs='*', type=str,
                        help='Field ID(s) / NAME(s) to plot', default=None)
    parser.add_argument('--htmlname', dest='html_name', type=str,
                        help='Output HTMLfile name', default='')

    """
    parser.add_argument('-p', '--plotname', dest='image_name', type=str,
                        help='Output image name', default='')

    """
    parser.add_argument('-t', '--table', dest='mytabs',
                        nargs='*', type=str,
                        help='Table(s) to plot (default = None)', default=[])

    """
    parser.add_argument('--t0', dest='t0', type=float,
                        help='Minimum time to plot (default = full range)',
                        default=-1)
    parser.add_argument('--t1', dest='t1', type=float,
                        help='Maximum time to plot (default = full range)',
                        default=-1)
    parser.add_argument('--yu0', dest='yu0', type=float,
                        help='Minimum y-value to plot for upper panel (default=full range)',
                        default=-1)
    parser.add_argument('--yu1', dest='yu1', type=float,
                        help='Maximum y-value to plot for upper panel (default=full range)',
                        default=-1)
    parser.add_argument('--yl0', dest='yl0', type=float,
                        help='Minimum y-value to plot for lower panel (default=full range)',
                        default=-1)
    parser.add_argument('--yl1', dest='yl1', type=float,
                        help='Maximum y-value to plot for lower panel (default=full range)',
                        default=-1)
    """
    parser.add_argument('--xaxis', dest='xaxis', type=str,
                        choices=x_choices, help='x-axis to plot',
                        default='time')

    parser.add_argument('--iterate', dest='iterate', type=str,
                        choices=iter_choices,
                        help='Select which variable to iterate over \
                              (defaults to none)',
                        default=None)
    """
    parser.add_argument('--timebin', dest='timebin', type=str,
                        help='Number of timestamsp in each bin')
    parser.add_argument('--chanbin', dest='chanbin', type=str,
                        help='Number of channels in each bin')
    """

    return parser


def main(**kwargs):

    if len(kwargs) == 0:
        NB_RENDER = False

        parser = get_argparser()
        options = parser.parse_args()

        corr = int(options.corr)
        yaxis = options.yaxis
        field_ids = options.fields
        html_name = options.html_name
        #image_name = options.image_name
        mycmap = options.mycmap
        mytabs = options.mytabs
        #plotants = options.plotants
        #t0 = options.t0
        #t1 = options.t1
        #yu0 = options.yu0
        #yu1 = options.yu1
        #yl0 = options.yl0
        #yl1 = options.yl1
        xaxis = options.xaxis
        iterate = options.iterate
        #timebin = options.timebin
        #chanbin = options.chanbin

    if len(mytabs) > 0:
        mytabs = [x.rstrip("/") for x in mytabs]
    else:
        logger.error('ragavi exited: No gain table specified.')
        sys.exit(-1)

    if len(field_ids) == 0:
        logger.error('ragavi exited: No field id specified.')
        sys.exit(-1)

    for mytab, field in zip(mytabs, field_ids):
        # for each pair of table and field

        field_names = get_fields(mytab).data.compute()

        # get id: field_name pairs
        field_ids = dict(enumerate(field_names))

        # convert all field names to their field ids
        if field.isdigit():
            field = int(field)
        else:
            field = name_2id(field, field_ids)

        # iterate over spw and field ids by default
        # by default data is grouped in field ids and data description
        partitions = list(xm.xds_from_ms(mytab, ack=False))

        #cNorm = colors.Normalize(vmin=0, vmax=len(partitions) - 1)
        #scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=mymap)

        mymap = cmx.get_cmap(mycmap)
        oup_a = []

        for count, chunk in enumerate(partitions):

            #colour = scalarMap.to_rgba(float(count), bytes=True)[:-1]
            #colour = colors.to_hex(mymap(count)).encode('utf-8')
            #colour = mymap(count)[:-1]
            colour = mymap

            # Only plot the specified field id
            if chunk.FIELD_ID != field:
                continue

            # black box returns an plottable element / composite element
            if iterate != None:
                title = "TEST"
                colour = cycle(['red', 'blue', 'green', 'purple'])
                f = blackbox(chunk, mytab, xaxis, yaxis, corr,
                             showFlagged=False, ititle=title,
                             color=colour, iterate=iterate)
            else:
                ititle = None
                f = blackbox(chunk, mytab, xaxis, yaxis, corr,
                             showFlagged=False, ititle=ititle,
                             color=colour, iterate=iterate)

            # store resulting dynamicmaps
            oup_a.append(f)

        l = hv.Overlay(oup_a).collate().opts(width=900, height=700)

        layout = hv.Layout([l])
        fname = "{}_{}.html".format(yaxis, xaxis)
        hv.save(layout, fname)

# for demo
if __name__ == '__main__':
    main()
