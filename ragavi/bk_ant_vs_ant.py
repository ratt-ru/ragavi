from __future__ import division, print_function

import hvplot
import hvplot.xarray
import hvplot.dask
import logging

import astropy.stats as astats
import bokeh as bk
import dask.array as da
import dask.dataframe as dd

import holoviews as hv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import scipy.stats as sstats
import sys
import xarray as xa
import xarrayms as xm
import vis_utils as ut
import warnings

from collections import namedtuple
from dask import compute, delayed
from functools import partial
from holoviews import opts
from holoviews.operation.datashader import aggregate, dynspread, datashade
from ipdb import set_trace
from itertools import count, cycle
from pyrap.tables import table

from bokeh.io import show, output_file, save
from bokeh.models import (Band, BasicTicker, Button,
                          Circle, ColorBar, ColumnDataSource, CustomJS,
                          DataRange1d, Div, Grid, HoverTool, Line, LinearAxis,
                          LinearColorMapper, LogColorMapper, LogScale,
                          Patch, PanTool, Plot, Range1d,
                          ResetTool, WheelZoomTool, Whisker)

from bokeh.layouts import gridplot, row, column
from bokeh.plotting import figure, curdoc
from bokeh.events import Tap
from bokeh.resources import INLINE
from bokeh import events

try:
    import gi
    gi.require_version('Gdk', '3.0')
    from gi.repository import Gdk
except:
    pass


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

warnings.filterwarnings('once')
logging.captureWarnings(True)

logger = config_logger()
sys.excepthook = _handle_uncaught_exceptions


def get_screen_size():
    """You have to import
        : from gi.repository import Gdk
          from collections import namedtuple
    """
    Screen = namedtuple('screen', 'width height')
    try:
        s = Gdk.Screen.get_default()
        height = 1 * s.get_height()
        width = 1 * s.get_width()
    except:
        # Assume a default resolution of 1920 x 1080
        height = 1080
        width = 1920
    return Screen(width, height)


def calc_plot_dims(nrows, ncols):
    ssize = get_screen_size()
    # screen size returned is 80% of the entire available screen
    width = ssize.width
    height = ssize.height

    # Reduce the size further to account for the space in btwn grids
    plot_width = int((width / ncols)) - 10
    plot_height = int((height / nrows)) - 10
    return plot_width, plot_height


def convert_data(data, yaxis='amplitude'):
    if yaxis == 'amplitude':
        out_data = ut.calc_amplitude(data)
    elif yaxis == 'phase':
        out_data = ut.calc_phase(data, unwrap=False)
    elif yaxis == 'real':
        out_data = ut.calc_real(data)
    elif yaxis == 'imaginary':
        out_data = ut.calc_imaginary(data)
    return out_data


def flag_data(data, flags):
    out_data = data.where(flags == False)
    return out_data


def process(chunk, corr=None, data_column='DATA', ptype='amplitude', flag=True):

    selection = ['TIME', 'SCAN_NUMBER', 'FLAG']

    if corr == None:
        data = chunk[data_column]
        flags = chunk['FLAG']
    else:
        data = chunk[data_column].sel(corr=corr)
        flags = chunk['FLAG'].sel(corr=corr)

    # if flagging is enabled
    if flag:
        data = flag_data(data, flags)

    if ptype == 'amplitude':
        selection.append('Amplitude')
        chunk['Amplitude'] = convert_data(data, yaxis=ptype)
    elif ptype == 'phase':
        selection.append('Phase')
        chunk['Phase'] = convert_data(data, yaxis=ptype)
    elif ptype == 'real':
        selection.append('Real')
        chunk['Real'] = convert_data(data, yaxis=ptype)
    elif ptype == 'imaginary':
        selection.append('Imaginary')
        chunk['Imaginary'] = convert_data(data, yaxis=ptype)

    # change time from MJD to UTC
    chunk['TIME'] = ut.time_convert(chunk.TIME)
    chunk['FLAG'] = flags

    return chunk[selection]


def callback(call_chunk, ant1, ant2, yaxis, event):
    global document, ant_names, freqs

    # get the actual antenna names
    ant1_name = ant_names[ant1]
    ant2_name = ant_names[ant2]

    # get range of channels
    nchans = call_chunk.chan.size
    first_chan = freqs[0]
    last_chan = freqs[-1]

    # get the time ranges from the current chunk
    first_time = np.datetime_as_string(call_chunk.TIME.data.compute()[
                                       0],                               timezone='UTC')
    first_time = first_time[:10] + ' ' + first_time[11:-1]

    last_time = np.datetime_as_string(call_chunk.TIME.data.compute(
    )[-1],                               timezone='UTC')
    last_time = last_time[:10] + ' ' + last_time[11:-1]

    new_ch = call_chunk.to_dask_dataframe()

    im = hv.render(new_ch.hvplot.heatmap(x='TIME', y='chan', C=yaxis,
                                         colorbar=True))

    # configuring the new popup plot
    im.width = 1000
    im.height = 1000

    # setting axis specifications
    im.axis.major_label_text_font_size = "0pt"
    im.axis.major_tick_line_alpha = 0.0
    im.axis.axis_label_text_font_size = "12pt"
    im.xaxis.axis_label = "Time {} to {}".format(first_time, last_time)
    im.yaxis.axis_label = "Channel 0: {:.4f} GHz to Channel {}: {:.4f} GHz".format(
        first_chan, nchans - 1, last_chan)

    im.title.text = "{} per Frequency vs Time: A1 {}, A2 {}".format(
        yaxis, ant1_name, ant2_name)
    im.title.align = 'center'
    im.title.text_font_size = '14pt'

    avail_roots = document.roots[0]

    if len(avail_roots.children) == 1:
        avail_roots.children.append(im)
    else:
        avail_roots.children[-1] = im

    print("Plot rendered")


def make_plot(df, tsource, fsource, yaxis):

    global mygrid, ncols, nrows, width, height, document, ant_names

    # get the current row antenna number
    ant1 = df.ANTENNA1
    ant2 = df.ANTENNA2

    ant1_name = ant_names[ant1]
    ant2_name = ant_names[ant2]

    dis = Div(text='A1 {}'.format(ant1_name),
              background="#3FBF9D")

    if ant1 == 0:
        div_top_row = Div(text='A2 {}'.format(ant2_name),
                          background="#3FBF9D")
        mygrid[0, ant2 + 1] = div_top_row

    # add div labels to the last col of the grid
    if ant2 == ncols - 1:
        div_right_col = Div(text='A1 {}'.format(ant1_name),
                            background="#3FBF9D")
        mygrid[ant1 + 1, -1] = div_right_col

    # add divs to the last row of the grid
    div_bot_row = Div(text='A1 {}'.format(ant1_name),
                           background="#BACA35")
    mygrid[-1, ant1 + 1] = div_bot_row

    # add div labels to the first col of the grid
    div_left_col = Div(text='A2 {}'.format(ant2_name),
                            background="#BACA35")
    mygrid[ant2 + 1, 0] = div_left_col

    # define some aesthetic options for the area markers
    # for the points
    circle_opts = {'fill_alpha': 1,
                   'fill_color': '#E1341E',
                   'size': 3,
                   'line_width': None,
                   'line_color': None,
                   }

    # for the margin of error
    whisker_opts = {'line_alpha': 0.4,
                    'line_width': 1,
                    'line_color': '#1ECBE1',
                    'line_cap': 'round',
                    'level': 'underlay'
                    }

    plot_opts = {'background_fill_alpha': 0.95,
                 'plot_width': height * 2,
                 'plot_height': height * 2,
                 'background_fill_color': '#efe8f2',
                 'output_backend': 'webgl',
                 'sizing_mode': 'scale_both'}

    hover = HoverTool(tooltips=[('Antenna1', '{}'.format(ant1_name)),
                                ('Antenna2', '{}'.format(ant2_name))])

    drange = Range1d(start=np.nanmin(fsource.data['flower']) - 5,
                     end=np.nanmax(fsource.data['fupper']) + 5)

    p = Plot(y_range=drange, **plot_opts)

    circle = Circle(x='time', y='mean', **circle_opts)

    whisker = Whisker(base='time', lower='tlower',
                      upper='tupper', source=tsource, **whisker_opts)
    p.add_glyph(tsource, circle)
    p.add_layout(whisker)
    p.on_event(Tap, partial(callback, df, ant1, ant2, yaxis))
    p.add_tools(WheelZoomTool(), ResetTool(), hover)

    f = Plot(y_range=drange, **plot_opts)
    fcircle = Circle(x='chans', y='fmean', **circle_opts)

    fwhisker = Whisker(base='chans', lower='flower',
                       upper='fupper', source=fsource, **whisker_opts)
    f.add_glyph(fsource, fcircle)
    f.add_layout(fwhisker)
    f.js_on_event(Tap, CustomJS(
        code="alert('A2: {}, A1: {}')".format(ant2_name, ant1_name)))
    f.on_event(Tap, partial(callback, df, ant1, ant2, yaxis))
    f.add_tools(WheelZoomTool(), ResetTool(), hover)

    mygrid[ant1 + 1, ant2 + 1] = p
    mygrid[ant2 + 1, ant1 + 1] = f


##############################################################
############## Main function begins here #####################

# To Do: Get these through the commandline

ms_name = "/home/andati/measurement_sets/1491291289.1ghz.1.1ghz.4hrs.ms"
#ms_name = "/home/andati/measurement_sets/ngc1399_20MHz.ms"

fid = 0
ddid = 0
data_col = 'DATA'
corr = 0

# Quantity to plot
yaxis = 'imaginary'
upper_yaxis = yaxis.capitalize()


# group the data per baseline, field id and spectral window
groups = list(xm.xds_from_ms(ms_name, group_cols=['DATA_DESC_ID',
                                                  'FIELD_ID',
                                                  'ANTENNA1',
                                                  'ANTENNA2']))


################################################################
###############  some global variables #########################

# antenna names and number of antennas
ant_names = ut.get_antennas(ms_name).data.compute()
nants = len(ant_names)
ncols = nants
nrows = nants

# initialise output layout grid using numpy array
# increase both columns and rows by 2 in order to add labels on all edges
mygrid = np.full(shape=(nants + 2, nants + 2), fill_value=None)

# get the associated frequencies for the specified ddid
freqs = (ut.get_frequencies(ms_name, spwid=ddid) / 1e9).data.compute()

# calculate the plot dimensions
width, height = calc_plot_dims(nrows, ncols)

# get the current document object for this bokeh script
document = curdoc()


###############################################################
####### data selection, processing and plotting  ##############

# find the field id that matches fid and did for all baselines
selections = []
for chunk in groups:
    if chunk.FIELD_ID == fid and chunk.DATA_DESC_ID == ddid:
        selections.append(chunk)


# process the data, convert to amplitudes and phases and select correlations
# also change the time format to datetime64
for sel_chunk in selections:
    sel_chunk = process(sel_chunk, corr=corr, data_column=data_col,
                        ptype=yaxis)

    # To Do: phase wrapped means?

    # find means in frequencies over a arnge of time
    mean = sel_chunk[upper_yaxis].mean(dim='chan').data
    std = sel_chunk[upper_yaxis].std(dim='chan').data

    # find in time over a range of frequencies
    fmean = sel_chunk[upper_yaxis].mean(dim='row').data
    fstd = sel_chunk[upper_yaxis].std(dim='row').data

    # get the available channel numbers
    chans = sel_chunk.chan.data
    tupper = mean + std
    tlower = mean - std

    fupper = fmean + fstd
    flower = fmean - fstd

    time = sel_chunk.TIME.data - sel_chunk.TIME[0].data

    # ignore division by nan warning because nanmean is executed
    np.seterr(divide='ignore', invalid='ignore')
    time, mean, tupper, tlower, fupper, flower, fmean = compute(
        time, mean, tupper, tlower, fupper, flower, fmean)

    ant1 = sel_chunk.ANTENNA1
    ant2 = sel_chunk.ANTENNA2

    # create a data source for the time component
    tsource = ColumnDataSource(data={'time': time,
                                     'mean': mean,
                                     'tupper': tupper,
                                     'tlower': tlower})

    # create a data source for the frequency component
    fsource = ColumnDataSource(data={'chans': chans,
                                     'fmean': fmean,
                                     'fupper': fupper,
                                     'flower': flower})

    make_plot(sel_chunk, tsource, fsource, upper_yaxis)


###############################################################
############ laying and rendering ############################

print("Done")
mygrid = mygrid.tolist()
gp = gridplot(mygrid)
col = column(gp)

document.add_root(col)
