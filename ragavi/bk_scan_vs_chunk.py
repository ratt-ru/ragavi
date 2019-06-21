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
import xarray as xa
import xarrayms as xm
import vis_utils as ut

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
                          LogAxis, Patch, PanTool, Plot, Range1d,
                          ResetTool, WheelZoomTool, Whisker)

from bokeh.layouts import gridplot, row, column
from bokeh.plotting import figure, curdoc
from bokeh.events import Tap
from bokeh.resources import INLINE
from bokeh.models import Button, AjaxDataSource
from bokeh import events


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
    out_data = data.where(flags != 1)
    return out_data


def process(chunk, corr=None, data_column='DATA', ptype='ap', flag=True, bin_size=100):
    """
        1. Select the correlation if an
        2. Chunk data and the flags along channels into bin_size sizes
        3. Flag the data
        4. Convert process data into ap or ri
        5. Convert time to human readable format
        6. Return the processed data and selected FLAG data
    """

    selection = ['TIME', 'SCAN_NUMBER', 'FLAG']

    if corr == None:
        data = chunk[data_column]
        flags = chunk['FLAG']
    else:
        data = chunk[data_column].sel(corr=corr)
        flags = chunk['FLAG'].sel(corr=corr)

    # chunk the data into bin sizes before processing
    data = data.chunk({'chan': bin_size})
    flags = flags.chunk({'chan': bin_size})

    # Data is flagged before processing
    # To Do: explore processing before flagging?

    # if flagging is enabled
    if flag:
        data = flag_data(data, flags)

    if ptype == 'ap':
        selection.extend(['Amplitude', 'Phase'])
        chunk['Amplitude'] = convert_data(data, yaxis='amplitude')
        chunk['Phase'] = convert_data(data, yaxis='phase')
    elif ptype == 'ri':
        selection.extend(['Real', 'Imaginary'])
        chunk['Real'] = convert_data(data, yaxis='real')
        chunk['Imaginary'] = convert_data(data, yaxis='imaginary')
    else:
        logging.exception("Invalid doplot value")
        sys.exit(-1)

    # change time from MJD to UTC
    chunk['TIME'] = ut.time_convert(chunk.TIME)
    chunk['FLAG'] = flags

    return chunk[selection]


def compute_idx(df, bw):
    """Compute category number for all the channel partitions available
    df: dask df
    bw: Number of channels in a particular bin
    """
    nchans = bw
    first = df.chan.min()
    index = int((nchans + first) / nchans) - 1
    return df.assign(chan_bin=index)


def creat_mv(df, doplot):
    if doplot == 'ap':
        df = df.assign(adatmean=df['Amplitude'].mean())
        df = df.assign(adatvar=df['Amplitude'].var())
        df = df.assign(pdatmean=astats.circmean(df['Phase']))
        df = df.assign(pdatvar=astats.circvar(df['Phase']))

    elif doplot == 'ri':
        df = df.assign(rdatmean=df['Real'].mean())
        df = df.assign(rdatvar=df['Real'].var())
        df = df.assign(idatmean=astats.circmean(df['Imaginary']))
        df = df.assign(idatvar=astats.circvar(df['Imaginary']))

    df = df.assign(flagged_pc=percentage_flagged(df))
    return df


def split_freqs(fs, bin_width):
    """
        Splits frequency arrays into specified chunks and returns the beginning and ending frequency for each chunk
    """
    fs_size = fs.size
    num_bins = int(np.ceil(fs_size / float(bin_width)))
    splits = np.array_split(fs, num_bins)

    ends = []
    for each in splits:
        start = each[0]
        end = each[-1]
        ends.append((start, end))

    return ends


def percentage_flagged(df):
    """
        Find the percentage of data flagged in each dask dataframe partition
    """
    true_flag = df.FLAG.where(df['FLAG'] == 1).count()
    nrows = df.FLAG.count()
    percentage = (true_flag / nrows) * 100
    return percentage


def on_click_callback(inp_df, scn, chan, doplot, event):

    global f_edges
    chan = int(chan)
    scn = int(scn)

    logging.info("Callback Active")
    if doplot == 'ap':
        xy = ['Phase', 'Amplitude']
    else:
        xy = ['Real', 'Imaginary']

    selection = inp_df[(inp_df['SCAN_NUMBER'] == scn)
                       & (inp_df['chan_bin'] == chan)]

    im = hv.render(selection.hvplot(*xy, kind='scatter', datashade=True))

    im.width = 1000
    im.height = 1000

    im.axis.axis_label_text_font_size = "12pt"

    im.title.text = "Scan {} Chunk {}: {:.4f} to {:.4f} GHz".format(
        scn, chan, *f_edges[chan])
    im.title.align = "center"
    im.title.text_font_size = '14pt'

    avail_roots = document.roots[0]

    if len(avail_roots.children) == 1:
        avail_roots.children.append(im)
    else:
        avail_roots.children[-1] = im

    logging.info('Callback plot added to root document.')


def make_plot(inp_df, doplot):
    # get the grouped dataframe with the mean and variance columns
    # drop the duplicate values based on the mean column
    # because a single entire partition contains the same values for both mean
    # # and variance

    global document, freqs, ncols, nrows, plot_width, plot_height

    logging.info("Starting plotting function")

    inp_subdf = inp_df.drop_duplicates(subset=['SCAN_NUMBER', 'flagged_pc'])
    logging.info('Dropped duplicate values')

    # To Do: conform to numpy matrix rather than sorting
    # rearrange the values in ascending so that plotting is easier
    inp_subdf = inp_subdf.compute()
    inp_subdf.index = inp_subdf.droplevel([0, 1])
    inp_subdf = inp_subdf.sort_values(['SCAN_NUMBER', 'chan_bin']).set_index(
        pd.RangeIndex(0, ncols * nrows))

    #*_col_a is for cols related to amplitude or real
    #*_col_b is for cols related to phase or imaginary

    if doplot == 'ap':
        mean_col_a = 'adatmean'
        mean_col_b = 'pdatmean'
        var_col_a = 'adatvar'
        var_col_b = 'pdatvar'
    else:
        mean_col_a = 'rdatmean'
        mean_col_b = 'idatmean'
        var_col_a = 'rdatvar'
        var_col_b = 'idatvar'

    lowest_mean = inp_subdf[mean_col_a].min()
    highest_mean = inp_subdf[mean_col_a].max()

    # get this mean in order to center the plot
    mid_mean = inp_subdf[mean_col_a].median()

    highest_var = inp_subdf[var_col_a].max()

    # calculate the lowest and highest channel number
    min_chan = inp_subdf.chan_bin.min()
    max_chan = inp_subdf.chan_bin.max()

    # calculate the lowest and highest scan number
    min_scan = inp_subdf.SCAN_NUMBER.min()
    max_scan = inp_subdf.SCAN_NUMBER.max()

    # initialise the layout matrix
    mygrid = np.full((nrows + 2, ncols + 1), None)

    logging.info('Calculations of maximums and minimums done')

    # set the axis bounds
    xdr = DataRange1d(bounds=None, start=lowest_mean -
                      highest_var, end=highest_mean + highest_var)
    ydr = DataRange1d(bounds=None, start=lowest_mean -
                      highest_var, end=highest_mean + highest_var)

    col_mapper = LinearColorMapper(
        palette="Viridis256",
        low=((lowest_mean - mid_mean) / mid_mean) * 100,
        high=((highest_mean - mid_mean) / mid_mean) * 100)

    col_bar = ColorBar(color_mapper=col_mapper,
                       location=(0, 0))

    color_bar_plot = figure(title="%Deviation from mean",
                            title_location="right",
                            height=int(plot_width * (nrows / 2)), width=200,
                            toolbar_location=None, min_border=0,
                            outline_line_color=None)

    color_bar_plot.add_layout(col_bar, 'right')
    color_bar_plot.title.align = "center"
    color_bar_plot.title.text_font_size = '12pt'

    logging.info('Starting iterations by row')

    # cyclic object for iteratively repetition over chan_bins
    cyc = cycle(np.arange(ncols))

    # get index number of row with each iteration
    ctr = count(0)
    row_idx = next(ctr)

    for rs, cols in inp_subdf.iterrows():

        col_idx = next(cyc)

        curr_scan_no = int(cols.SCAN_NUMBER)
        curr_cbin_no = int(cols.chan_bin)
        flagged_pc = cols.flagged_pc

        dis = Div(text='S{}'.format(curr_scan_no),
                  background="#3FBF9D")

        if col_idx == 0:
            mygrid[row_idx + 1, 0] = dis

        if curr_scan_no == max_scan:
            dis = Div(text='{}'.format('{:.4} - {:.4} GHz'.format(*f_edges[curr_cbin_no])),
                      background="#3FBF9D")
            mygrid[-1, col_idx + 1] = dis

        pc_dev_mean = ((cols[mean_col_a] - mid_mean) / mid_mean) * 100
        source = ColumnDataSource(data={'mean': [cols[mean_col_a]],
                                        'mid_mean': [mid_mean],
                                        'mean_anorm': [pc_dev_mean],
                                        'var': [cols[var_col_a]],
                                        'pcf': [cols.flagged_pc]})

        p = Plot(background_fill_alpha=0.95, x_range=xdr, y_range=ydr, plot_width=int(plot_width / 2), plot_height=int(plot_width / 2), y_scale=LogScale(),
                 x_scale=LogScale(), background_fill_color="#efe8f2")
        circle = Circle(x='mean', y='mean', radius='var',
                        fill_alpha=0.8, line_color=None,
                        fill_color={'field': 'mean_anorm',
                                    'transform': col_mapper})
        flag_circle = Circle(x='mean', y='mean',
                             radius=((source.data['pcf'][0] *
                                      source.data['var'][0]) * 0.01)**2,
                             fill_alpha=0.8, line_color=None,
                             fill_color='white')
        p.add_glyph(source, circle)
        p.add_glyph(source, flag_circle)

        # setting the plot ticker
        xticker = BasicTicker()
        yticker = BasicTicker()

        if curr_scan_no == max_scan:
            xaxis = LogAxis(major_tick_out=1,
                            minor_tick_out=1,
                            axis_line_width=1)

            xticker = xaxis.ticker

        if curr_cbin_no == min_chan:
            yaxis = LogAxis(major_tick_out=1,
                            minor_tick_out=1,
                            axis_line_width=1)

            yticker = yaxis.ticker

        hover = bk.models.HoverTool(tooltips=[('mean', '@mean'),
                                              ('var', '@var'),
                                              ('Flagged %', '@pcf'),
                                              ("%Deviation from midmean",
                                               '@mean_anorm')
                                              ])

        p.add_tools(PanTool(), hover, WheelZoomTool(), ResetTool())
        p.on_event(Tap, partial(on_click_callback, inp_df,
                                curr_scan_no, curr_cbin_no, doplot))

        # add the divs from the 2nd element to the 2nd last element of the grid
        mygrid[row_idx + 1, curr_cbin_no + 1] = p
        if col_idx == ncols - 1:
            row_idx = next(ctr)

    gp = gridplot(mygrid.tolist())
    ro = row(gp, color_bar_plot)

    col = column(ro)

    print("Adding plot to root")

    document.add_root(col)


#####################################################################
################# Main function starts here ##########################

ms_name = "/home/andati/measurement_sets/1491291289.1ghz.1.1ghz.4hrs.ms"
bin_width = 50
fid = 1
corr = 0
ddid = 0

data_col = 'DATA'
doplot = 'ap'

a = list(xm.xds_from_ms(ms_name, columns=['DATA', 'TIME', 'FLAG',
                                          'SCAN_NUMBER', 'ANTENNA1',
                                          'ANTENNA2'],
                        group_cols=['DATA_DESC_ID', 'FIELD_ID']))

ch = a[fid]

sel_chunk = process(ch, corr=corr, data_column=data_col, ptype=doplot,
                    flag=True, bin_size=bin_width)


###################################################################
################# some global variables ##########################
document = curdoc()
freqs = (ut.get_frequencies(ms_name, spwid=ddid) / 1e9).data.compute()
f_edges = split_freqs(freqs, bin_width)
nrows = np.unique(sel_chunk.SCAN_NUMBER).size
# checking the flag column because it is chunked like the data
ncols = sel_chunk.FLAG.data.npartitions
plot_width, plot_height = calc_plot_dims(nrows, ncols)


# converting to data frame to allow for leveraging of dask data frame #partitions
# Since data is already chunked in xarray, the number of partitions
# corresponds to the number of chunks which makes it easier for now.

sel_df = sel_chunk.to_dask_dataframe()

# add chunk index numbers to the partitions
nddf = sel_df.map_partitions(compute_idx, bin_width)

sel_cols = ['SCAN_NUMBER', 'FLAG', 'chan_bin']

# additional columns
additional_cols = []

if doplot == 'ap':
    sel_cols.extend(['Amplitude', 'Phase'])
    additional_cols.extend(['adatmean', 'adatvar',
                            'pdatmean', 'pdatvar', 'flagged_pc'])
    f64 = sel_df.dtypes.to_dict()['Amplitude']
elif doplot == 'ri':
    sel_cols.extend(['Real', 'Imaginary'])
    additional_cols.extend(['rdatmean', 'rdatvar',
                            'idatmean', 'idatvar', 'flagged_pc'])
    f64 = sel_df.dtypes.to_dict()['Real']

# get colnames from dataframes and add the new col names
fdf_col_names = sel_cols + additional_cols

fdf_dtypes = nddf[sel_cols].dtypes.to_list() + [f64] * len(additional_cols)

# groupby meta

meta = [(key, value) for key, value in zip(fdf_col_names, fdf_dtypes)]

# grouping by chan_bin and scan_number and select
res = nddf.groupby(['SCAN_NUMBER', 'chan_bin'])[sel_cols].apply(creat_mv,
                                                                doplot,
                                                                meta=meta)

make_plot(res, doplot)
