from __future__ import division, print_function

import gi
gi.require_version('Gdk', '3.0')

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

from collections import namedtuple
from dask import compute, delayed
from gi.repository import Gdk
from functools import partial
from holoviews import opts
from holoviews.operation.datashader import dynspread, datashade
from ipdb import set_trace
from itertools import count, cycle
from pyrap.tables import table

from bokeh.io import show, output_file, save
from bokeh.models import (Band, BasicTicker,
                          Circle, ColorBar, ColumnDataSource, CustomJS,
                          DataRange1d, Div, Grid, HoverTool, Line, LinearAxis,
                          LinearColorMapper, LogColorMapper, PanTool, Plot,
                          ResetTool, WheelZoomTool)

from bokeh.layouts import gridplot, row, column
from bokeh.plotting import figure, curdoc
from bokeh.events import Tap
from bokeh.resources import INLINE


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
        out_data = da.absolute(data)
    elif yaxis == 'phase':
        out_data = xa.ufuncs.angle(data)
    elif yaxis == 'real':
        out_data = data.real
    elif yaxis == 'imaginary':
        out_data = data.imag
    return out_data


def flag_data(data, flags):
    out_data = data.where(flags != 1)
    return out_data


def process(mchunk, corr, yaxis):
    data = mchunk.DATA.sel(corr=corr)
    flag = mchunk.FLAG.sel(corr=corr)

    ind = convert_data(data, yaxis)
    out = flag_data(ind, flag)

    return out


def compute_idx(df, bw):
    """Compute category number for all the channel partitions available
    df: dask df
    bw: Number of channels in a particular bin
    """
    nchans = bw
    first = df.chan.min()
    index = int((nchans + first) / nchans) - 1
    return df.assign(chan_bin=index)


def creat_mv(df, ap):
    if ap == 'amplitude':
        df['datmean'] = np.mean(df['Amplitude'])
        df['datvar'] = np.var(df['Amplitude'])
    else:
        df['datmean'] = astats.circmean(df['Phase'])
        df['datvar'] = astats.circvar(df['Phase'])

    df['flagged_pc'] = percentage_flagged(df)
    return df


def get_field(ms_name):
    ftab = list(xmsm.xds_from_table("::".join((ms_name, 'ANTENNA'))))[0]
    fields = ftab.Name.data.compute()
    return fields[fnum]


def get_channels(ms_name):
    spw = list(xm.xds_from_table("::".join((ms_name, 'SPECTRAL_WINDOW'))))[0]
    freqs = (spw.CHAN_FREQ / 1e9).data.compute()[0]
    return freqs


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
    true_flag = df.FLAG.where(df['FLAG'] == 1).count()
    nrows = df.FLAG.count()
    percentage = (true_flag / nrows) * 100
    return percentage


def make_plot(inp_df):
    # get the grouped dataframe with the mean and variance columns
    # drop the duplicate values based on the mean column
    # because a single entire partition contains the same values for both mean
    # # and variance
    document = curdoc()

    def on_click_callback(scn, chan, event):

        print("Registering callback")

        #inp_df = inp_df.head(npartitions=-1)

        selection = inp_df[(inp_df['SCAN_NUMBER'] == scn)
                           & (inp_df['chan_bin'] == chan)]
        im = hv.render(selection.hvplot('Phase', 'Amplitude',
                                        kind='scatter', datashade=True))
        im.title.text = "Scan {} Chunk {}".format(int(scn), int(chan))
        # im.title.text_align = "center"

        avail_roots = document.roots[0]

        if len(avail_roots.children) == 1:
            avail_roots.children.append(im)
        else:
            avail_roots.children[-1] = im

    print("In plotting")

    inp_subdf = inp_df.drop_duplicates(subset=['datmean'])
    print('Dups droped, proceed')
    # rearrange the values in ascending so that plotting is easier
    inp_subdf = inp_subdf.compute().sort_values(['SCAN_NUMBER', 'chan_bin'])
    nrows = inp_subdf.SCAN_NUMBER.unique().size
    ncols = inp_subdf.chan_bin.unique().size

    plot_width, plot_height = calc_plot_dims(nrows, ncols)

    lowest_mean = inp_subdf.datmean.min()
    highest_mean = inp_subdf.datmean.max()

    # get this mean in order to center the plot
    mid_mean = inp_subdf.datmean.median()

    highest_var = inp_subdf.datvar.max()

    print('Stats calculated proceed')

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

    min_chan = inp_subdf.chan_bin.min()
    max_chan = inp_subdf.chan_bin.max()

    min_scan = inp_subdf.SCAN_NUMBER.min()
    max_scan = inp_subdf.SCAN_NUMBER.max()

    print("Min max gottend")
    gcols = []
    grows = []

    print("starting row by row iter")

    mygrid = np.full((nrows + 2, ncols + 1), None)

    cyc = cycle(np.arange(ncols))
    ctr = count(0)
    row_idx = ctr.next()

    for rs, cols in inp_subdf.iterrows():

        col_idx = cyc.next()

        dis = Div(text='S{}'.format(int(cols.SCAN_NUMBER)),
                  background="#3FBF9D")

        if col_idx == 0:
            mygrid[row_idx + 1, 0] = dis

        if int(cols.SCAN_NUMBER) == max_scan:
            dis = Div(text='{}'.format('{:.4} - {:.4} GHz'.format(*f_edges[cols.chan_bin])),
                      background="#3FBF9D")
            mygrid[-1, col_idx + 1] = dis

        # set_trace()
        pc_dev_mean = ((cols.datmean - mid_mean) / mid_mean) * 100
        source = ColumnDataSource(data={'mean': [cols.datmean],
                                        'mid_mean': [mid_mean],
                                        'mean_anorm': [pc_dev_mean],
                                        'var': [cols.datvar],
                                        'pcf': [cols.flagged_pc]})

        # set_trace()

        p = Plot(background_fill_alpha=0.95, x_range=xdr, y_range=ydr, plot_width=int(plot_width / 2), plot_height=int(plot_width / 2), y_scale=LogScale(),
                 x_scale=LogScale(), background_fill_color="#efe8f2")
        circle = Circle(x='mean', y='mean', radius='var',
                        fill_alpha=0.8, line_color=None,
                        fill_color={'field': 'mean_anorm',
                                    'transform': col_mapper})
        flag_circle = Circle(x='mean', y='mean',
                             radius=(source.data['pcf'][0] *
                                     source.data['var'][0]) * 0.01,
                             fill_alpha=0.8, line_color=None,
                             fill_color='white')
        p.add_glyph(source, circle)
        p.add_glyph(source, flag_circle)
        # xdr.renderers.append(renderer)
        # ydr.renderers.append(renderer)

        # setting the plot ticker
        xticker = BasicTicker()
        yticker = BasicTicker()

        if int(cols.SCAN_NUMBER) == min_scan and int(cols.chan_bin) == max_chan:
            pass
            #p.add_layout(col_bar, 'right')

        if int(cols.SCAN_NUMBER) == max_scan:
            xaxis = LogAxis(major_tick_out=1,
                            minor_tick_out=1,
                            axis_line_width=1)
            cbin = int(cols.chan_bin)
            #xaxis.axis_label = '{:.4} - {:.4} GHz'.format(*f_edges[cbin])
            p.add_layout(xaxis, 'below')
            xticker = xaxis.ticker

        if int(cols.chan_bin) == min_chan:
            yaxis = LogAxis(bounds="auto", major_tick_out=1,
                            minor_tick_out=1,
                            axis_line_width=1)
            #yaxis.axis_label = 'Scan {}'.format(int(cols.SCAN_NUMBER))
            p.add_layout(yaxis, 'left')
            yticker = yaxis.ticker

        hover = bk.models.HoverTool(
            tooltips=[('mean', '@mean'), ('var', '@var'),
                      ('Flagged %', '@pcf'),
                      ("%Deviation from midmean", '@mean_anorm')])

        # p.add_layout(Grid(dimension=0, ticker=xticker))
        # p.add_layout(Grid(dimension=1, ticker=yticker))
        p.add_tools(PanTool(), hover, WheelZoomTool(), ResetTool())
        # p.js_on_event(Tap, CustomJS(
        # code="alert('scan_no: {} chunk: {}')".format(cols.SCAN_NUMBER,
        # cols.chan_bin)))

        p.on_event(Tap, partial(on_click_callback,
                                cols.SCAN_NUMBER, cols.chan_bin))
        # p.callback = CustomJS(code=jscode, args=dict(plot=p, dims=dims))

        # add the divs from the 2nd element to the 2nd last element of the grid
        mygrid[row_idx + 1, int(cols.chan_bin) + 1] = p
        if col_idx == ncols - 1:
            row_idx = ctr.next()

        # gcols.append(p)
    # set_trace()
    #grows = np.split(np.array(gcols), len(gcols) / ncols)
    gp = gridplot(mygrid.tolist())
    ro = row(gp, color_bar_plot)

    col = column(ro)

    print("Adding plot to root")

    curdoc().add_root(col)

# gb_scans = ['DATA_DESC_ID', 'FIELD_ID', 'SCAN_NUMBER']
# gb_baselines = ['DATA_DESC_ID', 'FIELD_ID', 'ANTENNA1', 'ANTENNA2']
# default = ['DATA_DESC_ID', 'FIELD_ID']


print("Lifting off")

ms_name = "/home/andati/measurement_sets/1491291289.1ghz.1.1ghz.4hrs.ms"

a = list(xm.xds_from_ms(ms_name, columns=['DATA', 'TIME', 'FLAG',
                                          'SCAN_NUMBER', 'ANTENNA1',
                                          'ANTENNA2'],
                        group_cols=['DATA_DESC_ID', 'FIELD_ID']))

field_id = 0
corr = 0
ch = a[field_id]

amp = process(ch, corr, 'amplitude')


phase = process(ch, corr, 'phase')

bin_width = 50
fs = get_channels(ms_name)
f_edges = split_freqs(fs, bin_width)

ch['Amplitude'] = amp.chunk({'chan': bin_width})
ch['Phase'] = phase.chunk({'chan': bin_width})
ch['FLAG'] = ch.FLAG.sel(corr=0).chunk({'chan': bin_width})

selection = ['Amplitude', 'Phase', 'SCAN_NUMBER', 'FLAG']
ddf = ch[selection].to_dask_dataframe()
nddf = ddf.map_partitions(compute_idx, bin_width)
print("partiitons mapped")

# grouping by channelbin and san_number and select
res = nddf.groupby(['SCAN_NUMBER', 'chan_bin'])[['SCAN_NUMBER', 'FLAG',
                                                 'chan_bin', 'Amplitude',
                                                 'Phase']].apply(creat_mv, 'amplitude')

print("Starting plotting function")
make_plot(res)
