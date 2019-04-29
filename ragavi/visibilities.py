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


def config_logger():
    """This function is used to configure the logger for ragavi and catch
        all warnings output by sys.stdout.
    """
    logfile_name = 'ragavi.log'
    # capture only a single instance of a matching repeated warning
    warnings.filterwarnings('once')
    logging.captureWarnings(True)

    # setting the format for the logging messages
    start = " (O_o) ".center(80, "=")
    form = '{}\n%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    form = form.format(start)
    formatter = logging.Formatter(form, datefmt='%d.%m.%Y@%H:%M:%S')

    # setup for ragavi logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)

    warnings_logger = logging.getLogger('py.warnings')

    xmslog = logging.getLogger('xarrayms')
    xmslog.setLevel(logging.ERROR)

    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(formatter)
    # setup for logfile handing ragavi
    fh = logging.FileHandler(logfile_name)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    xmslog.addHandler(fh)
    warnings_logger.addHandler(fh)
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
    ms_name: str
             Name of measurement set
    spwid: int
           Spectral window id number. Defaults to 0

    Outputs
    -------
    freqs: xarray DataArray
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
    ms_name: str
             Name of measurement set

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
    """ Get Flag values from the FLAG column
    Allows the selection of flags for a single correlation. If none is specified the entire data is then selected.
    Inputs
    ------
    xds_table_obj: xarray Dataset
                   MS as xarray dataset from xarrayms
    corr: int
          Correlation number to select.

    Outputs
    -------
    flags: xarray DataArray
           Data array containing values from FLAG column selected by correlation if index is available.

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
    """Get field names from the FIELD subtable.
    Inputs
    ------
    ms_name: str
             Name of measurement set

    Outputs
    -------
    field_names: xarray DataArray
                 String names for the available data in the table
    """
    subname = "::".join((ms_name, 'FIELD'))
    field_subtab = list(xm.xds_from_table(subname, ack=False))
    field_subtab = field_subtab[0]
    field_names = field_subtab.NAME
    return field_names


def calc_uvdist(uvw):
    """ Calculate uv distance in metres
    Inputs
    ------
    uvw: xarray DataArray
         UVW column from measurement set

    Outputs
    -------
    uvdist: xarray DataArray
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
    uvw: xarray DataArray
         UVW column from the MS dataset
    freq: xarray DataArray or float
          Frequency(ies) from which corresponding wavelength will be obtained.

    Outputs
    -------
    uvwave: xarray DataArray
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
    """ Convert time from MJD to UTC time
    Inputs
    ------
    xdata: xarray DataArray
           TIME column from the MS xarray dataset in MJD format.

    Outputs
    -------
    newtime: xarray DataArray
             TIME column in a more human readable UTC format. Stored as np.datetime type.

    """

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
        xdata = get_frequencies(ms_name)
        if xaxis == 'channel':
            x_label = 'Channel'
        else:
            x_label = 'Frequency GHz'
    elif xaxis == 'real':
        xdata = xds_table_obj.DATA
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


def prep_xaxis_data(xdata, xaxis='time', freq=None):
    """Prepare the x-axis data for plotting.
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
        prepdx = xdata.chan
    elif xaxis == 'frequency':
        prepdx = xdata / 1e9
    elif xaxis == 'phase':
        prepdx = xdata
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


def get_yaxis_data(xds_table_obj, ms_name, yaxis, datacol='DATA'):
    """Extract the required column for the y-axis data.


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


def prep_yaxis_data(xds_table_obj, ms_name, ydata, yaxis='amplitude', corr=0, flag=False, iterate=None):
    """Process data for the y-axis which includes:
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
        ydata = list(ydata.groupby('corr'))
        ydata = [x[1] for x in ydata]

        # group flags for that data as well and store in a list
        flags = get_flags(xds_table_obj).groupby('corr')
        flags = [x[1] for x in flags]
    else:
        ydata = ydata.sel(corr=corr)
        flags = get_flags(xds_table_obj).sel(corr=corr)

    # if flagging enabled return a list of DataArrays otherwise return a
    # single dataarray
    if flag:
        if isinstance(ydata, list):
            y = []
            for d, f in zip(ydata, flags):
                processed = process_data(d, yaxis=yaxis)
                # replace with NaNs where flags is not 1
                y.append(processed.where(f < 1))
        else:
            processed = process_data(ydata, yaxis=yaxis)
            y = processed.where(flags < 1)
    else:
        if isinstance(ydata, list):
            y = []
            for d in ydata:
                y.append(process_data(d, yaxis=yaxis))
        else:
            y = process_data(ydata, yaxis=yaxis)

    return y


def get_phase(ydata, unwrap=False):
    """Convert complex data to angle in degrees
    Inputs
    ------
    ydata: xarray DataArray
           y-axis data to be processed

    Outputs
    -------
    phase: xarray DataArray
           y-axis data converted to degrees
    """
    phase = xa.ufuncs.angle(ydata, deg=True)
    if unwrap:
        # delay dispatching of unwrapped phase
        phase = delayed(np.unwrap)(phase)
    return phase


def get_amplitude(ydata):
    """Convert complex data to amplitude (abs value)
    Inputs
    ------
    ydata: xarray DataArray
           y-axis data to be processed

    Outputs
    -------
    amplitude: xarray DataArray
           y-axis data converted to amplitude
    """
    amplitude = da.absolute(ydata)
    return amplitude


def get_real(ydata):
    """Extract real part from complex data
    Inputs
    ------
    ydata: xarray DataArray
           y-axis data to be processed

    Outputs
    -------
    real: xarray DataArray
           Real part of the y-axis data
    """
    real = ydata.real
    return real


def get_imaginary(ydata):
    """Extract imaginary part from complex data
    Inputs
    ------
    ydata: xarray DataArray
           y-axis data to be processed

    Outputs
    -------
    imag: xarray DataArray
           Imaginary part of the y-axis data
    """
    imag = ydata.imag
    return imag


def process_data(ydata, yaxis):
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
        y = get_amplitude(ydata)
    elif yaxis == 'imaginary':
        y = get_imaginary(ydata)
    elif yaxis == 'phase':
        y = get_phase(ydata)
    elif yaxis == 'real':
        y = get_real(ydata)
    return y


def blackbox(xds_table_obj, ms_name, xaxis, yaxis, corr, datacol='DATA', showFlagged=False, ititle=None, color='blue', iterate=None):
    """Takes in raw input and gives out a holomap.
    Inputs
    ------
    xds_table_obj: xarray Dataset
                   MS as xarray dataset from xarrayms
    ms_name: str
             Measurement set name
    xaxis: str
           Selected x-axis
    yaxis: str
           Selected y-axis
    corr: int
          Correlation index to select
    datacol: str
             Column from which data is pulled. Default is 'DATA'
    showFlagged: bool
                 Switch flags on or off.
    ititle: str
            Title to use incase of an iteration
    color: str / colormap
           Color to use for the plot. May be a color in the form of a string or a colormap.
    iterate: str
             Paramater over which to iterate

    Outputs
    -------
    fig:  
         A plotable metacontainer**
    """

    x_data, xlabel = get_xaxis_data(xds_table_obj, ms_name, xaxis)
    y_data, ylabel = get_yaxis_data(xds_table_obj, ms_name, yaxis,
                                    datacol=datacol)
    y_prepd = prep_yaxis_data(xds_table_obj, ms_name, y_data,
                              yaxis=yaxis, corr=corr,
                              flag=showFlagged, iterate=iterate)
    if xaxis == 'channel' or xaxis == 'frequency':
        y_prepd = y_prepd.transpose()
        y_prepd = y_prepd.transpose()

    if xaxis == 'uvwave':
        freqs = get_frequencies(ms_name).compute()
        x_prepd = prep_xaxis_data(x_data, xaxis, freq=freqs)
    elif xaxis == 'real':
        x_prepd = prep_yaxis_data(xds_table_obj, ms_name, x_data,
                                  yaxis=xaxis, corr=corr,
                                  flag=showFlagged)
    else:
        x_prepd = prep_xaxis_data(x_data, xaxis)

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
          'uvdistance': [('Uvdistance', 'UVDistance (m)')],
          'uvwave': [('Uvwave', 'UVDistance (lambda)')],
          'frequency': [('Frequency', 'Frequency GHz')],
          'channel': [('Channel', 'Channel')],
          'real': [('Real', 'Real')]
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


def resolve_ranges(inp):
    """Parse string range to numpy range.
    Returns a single integer value for the case whereby the input is of the format e.g "7:" returns 7, otherwise, an array will be returned
    e.g
    Inputs
    ------
    inp: str
         String to be resolved into numeric range

    Outputs:
    res: numpy.ndarray
         Numpy array containing a range of values or a single value
    """
    delimiters = ['~', ':', ',']
    if '~' in inp:
        outp = [int(x) for x in inp.split('~')]
        res = np.arange(outp[0], outp[-1] + 1)
    elif ':' in inp:
        split = inp.split(':')
        if split[-1] != '':
            outp = [int(x) for x in split]
            res = np.arange(outp[0], outp[-1])
        else:
            res = int(split[0])
    elif ',' in inp:
        outp = [int(x) for x in inp.split(',')]
        res = np.array(outp)
    else:
        res = np.array(int(inp))
    return res


def get_argparser():
    """Get argument parser"""

    x_choices = ['antenna1', 'antenna2',
                 'channel', 'frequency',
                 'real', 'scan', 'time',
                 'uvdistance', 'uvwave']

    y_choices = ['amplitude', 'imaginary', 'phase', 'real']

    iter_choices = ['scan', 'corr', 'spw']

    # TODO: make this arg parser inherit from ragavi the common options
    parser = ArgumentParser(usage='prog [options] <value>')

    parser.add_argument('-c', '--corr', dest='corr', type=int, metavar='',
                        help='Correlation index to plot (usually just 0 or 1,\
                              default = 0)',
                        default=0)
    parser.add_argument('--cmap', dest='mycmap', type=str, metavar='',
                        help=' Colour or matplotlib colour map to use \
                                (default=coolwarm)',
                        default='coolwarm')
    parser.add_argument('--datacolumn', dest='datacolumn', type=str,
                        metavar='',
                        help='Column from MS to use for data', default='DATA')
    parser.add_argument('--ddid', dest='ddid', type=str,
                        metavar='',
                        help='DATA_DESC_ID(s) to select. (default is all) Can\
                              be specified as e.g. "5", "5,6,7", "5~7"\
                              (inclusive range), "5:8" (exclusive range),"5:"\
                              (from 5 to last)',
                        default=None)
    parser.add_argument('-f', '--field', dest='fields', nargs='+', type=str,
                        metavar='',
                        help='Field ID(s) / NAME(s) to plot', default=None)
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
                        help='Select which variable to iterate over \
                              (defaults to none)',
                        default=None)
    parser.add_argument('--scan_no', dest='scan_no', type=str, metavar='',
                        help='Scan Number to select. (default is all)',
                        default=None)
    parser.add_argument('-t', '--table', dest='mytabs',
                        nargs='*', type=str, metavar='',
                        help='Table(s) to plot (default = None)', default=[])
    parser.add_argument('--xaxis', dest='xaxis', type=str, metavar='',
                        choices=x_choices, help='x-axis to plot',
                        default='time')
    parser.add_argument('--yaxis', dest='yaxis', type=str, metavar='',
                        choices=y_choices, help='Y axis variable to plot',
                        default='amplitude')

    """
    parser.add_argument('-a', '--ant', dest='plotants', type=str,
                        help='Plot only this antenna, or comma-separated list\
                              of antennas',
                        default=[-1])
    parser.add_argument('-p', '--plotname', dest='image_name', type=str,
                        help='Output image name', default='')

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
        datacolumn = options.datacolumn
        ddid = options.ddid
        field_ids = options.fields
        flag = options.flag
        iterate = options.iterate
        html_name = options.html_name
        #image_name = options.image_name
        mycmap = options.mycmap
        mytabs = options.mytabs
        scan_no = options.scan_no
        #plotants = options.plotants
        #t0 = options.t0
        #t1 = options.t1
        #yu0 = options.yu0
        #yu1 = options.yu1
        #yl0 = options.yl0
        #yl1 = options.yl1
        xaxis = options.xaxis
        yaxis = options.yaxis
        #timebin = options.timebin
        #chanbin = options.chanbin

    if len(mytabs) > 0:
        mytabs = [x.rstrip("/") for x in mytabs]
    else:
        logger.error('ragavi exited: No Measurement set specified.')
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

        groupings = ['FIELD_ID', 'DATA_DESC_ID']

        if scan_no != None:
            groupings.append('SCAN_NUMBER')
            scan_no = resolve_ranges(scan_no)
        if ddid != None:
            ddid = resolve_ranges(ddid)

        if datacolumn == 'DATA':
            # iterate over spw and field ids by default
            # by default data is grouped in field ids and data description
            partitions = list(xm.xds_from_ms(mytab, group_cols=groupings,
                                             ack=False))
        else:
            ms_schema = MS_SCHEMA.copy()
            ms_schema[datacolumn] = ColumnSchema(dims=('chan', 'corr'))
            partitions = list(xm.xds_from_table(mytab, ack=False,
                                                group_cols=groupings,
                                                table_schema=ms_schema))

        mymap = cmx.get_cmap(mycmap)
        oup_a = []

        for count, chunk in enumerate(partitions):

            colour = mymap

            # Only plot the specified field id
            if chunk.FIELD_ID != field:
                continue

            # if ddid or scan number has been specified
            if ddid is not None:
                if isinstance(ddid, int):
                    if chunk.DATA_DESC_ID < ddid:
                        continue
                else:
                    if chunk.DATA_DESC_ID not in ddid:
                        continue

            if scan_no is not None:
                if isinstance(scan_no, int):
                    if chunk.SCAN_NUMBER < scan_no:
                        continue
                else:
                    if chunk.SCAN_NUMBER not in scan_no:
                        continue

            # black box returns an plottable element / composite element
            if iterate != None:
                title = "TEST"
                colour = cycle(['red', 'blue', 'green', 'purple'])
                f = blackbox(chunk, mytab, xaxis, yaxis, corr,
                             showFlagged=flag, ititle=title,
                             color=colour, iterate=iterate,
                             datacol=datacolumn)
            else:
                ititle = None
                f = blackbox(chunk, mytab, xaxis, yaxis, corr,
                             showFlagged=flag, ititle=ititle,
                             color=colour, iterate=iterate,
                             datacol=datacolumn)

            # store resulting dynamicmaps
            oup_a.append(f)

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
