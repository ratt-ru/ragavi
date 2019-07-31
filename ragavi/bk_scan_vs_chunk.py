from __future__ import division, print_function

import hvplot
import hvplot.xarray
import hvplot.dask
import logging
import sys

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
from functools import partial
from holoviews import opts
from holoviews.operation.datashader import aggregate, dynspread, datashade
from ipdb import set_trace
from itertools import count, cycle
from pyrap.tables import table
from screeninfo import get_monitors
from xarrayms.known_table_schemas import MS_SCHEMA, ColumnSchema

from bokeh.io import show, output_file, save
from bokeh.models import (Band, BasicTicker, Button,
                          Circle, ColorBar, ColumnDataSource, CustomJS,
                          DataRange1d, Div, Grid, HoverTool, Line, LinearAxis,
                          LinearColorMapper, LogColorMapper, LogScale,
                          LogAxis, Patch, PanTool, Plot, Range1d,
                          ResetTool, WheelZoomTool, Whisker, LinearScale)

from bokeh.layouts import gridplot, row, column, grid, layout
from bokeh.plotting import figure, curdoc
from bokeh.events import Tap
from bokeh.resources import INLINE
from bokeh.models import Button, AjaxDataSource
from bokeh import events

import vis_utils as ut
from ipdb import set_trace


def get_ms(ms_name, data_col='DATA', ddid=None, fid=None, where=None):
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
        where: str
                TAQL where clause to be used with the MS. Takes precidence over all the other optional arguments

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
        if ddid != None:
            where.append("DATA_DESC_ID=={}".format(ddid))
        if fid != None:
            where.append("FIELD_ID=={}".format(fid))

        where = "&&".join(where)

    try:
        tab_objs = xm.xds_from_table(ms_name, taql_where=where,
                                     table_schema=ms_schema)
        return tab_objs
    except:
        logging.exception("Invalid DATA_DESC_ID, FIELD_ID or TAQL clause")
        sys.exit(-1)


def calc_plot_dims(nrows, ncols):

    try:
        ssize = get_monitors()[0]
        # screen size returned is 80% of the entire available screen
        width = ssize.width
        height = ssize.height
    except IndexError:
        width = 1980
        height = 1080

    # Reduce the size further to account for the space in btwn grids
    # nrows, number of frequency chunks present
    # ncols, number of scans present
    plot_width = (width // ncols) * 0.8

    plot_height = (width // ncols) * 0.8

    if nrows > ncols:
        plot_height = height // nrows

    return int(plot_width), int(plot_height)


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
    # we are chunking in the chan dimension
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

    im.width = 800
    im.height = 800

    im.axis.axis_label_text_font_size = "12pt"

    im.title.text = "Scan {} Chunk {}: {:.4f} to {:.4f} GHz".format(
        scn, chan, *f_edges[chan])
    im.title.align = "center"
    im.title.text_font_size = '14pt'

    avail_roots = document.roots[0]

    if not isinstance(avail_roots.children[-1], type(im)):
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

    # dropping duplicate values base on scan_number and flagged_pc
    # cause each chunk has the same scan number, flagged_pc, mean and variance

    inp_subdf = inp_df.drop_duplicates(subset=['SCAN_NUMBER', 'adatmean',
                                               'flagged_pc'])
    logging.info('Dropped duplicate values')

    # To Do: conform to numpy matrix rather than sorting
    # rearrange the values in ascending so that plotting is easier

    inp_subdf = (inp_subdf.reset_index(drop=True)
                 .compute().sort_values(['SCAN_NUMBER', 'chan_bin'])
                 .reset_index(drop=True))

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
    lowest_var = inp_subdf[var_col_a].min()

    # calculate the lowest and highest channel number
    min_chan = inp_subdf.chan_bin.min()
    max_chan = inp_subdf.chan_bin.max()

    # calculate the lowest and highest scan number
    min_scan = inp_subdf.SCAN_NUMBER.min()
    max_scan = inp_subdf.SCAN_NUMBER.max()

    none_text = Div(text='All data Flagged')
    # initialise the layout matrix
    mygrid = np.full((nrows + 1, ncols + 1), None)

    logging.info('Calculations of maximums and minimums done')

    # set the axis bounds
    xmin = lowest_mean - 2 * np.abs(np.log(np.sqrt(lowest_var)))
    xmax = highest_mean + 2 * np.abs(np.log(np.sqrt(lowest_var)))

    xdr = Range1d(start=xmin, end=xmax)
    ydr = Range1d(start=xmin, end=xmax)

    col_mapper = LinearColorMapper(
        palette="Viridis256",
        low=((lowest_mean - mid_mean) / mid_mean) * 100,
        high=((highest_mean - mid_mean) / mid_mean) * 100)

    col_bar = ColorBar(color_mapper=col_mapper,
                       location=(0, 0))

    color_bar_plot = figure(title="%Deviation from mean",
                            title_location="right",
                            height=int(plot_height * nrows), width=150,
                            toolbar_location=None, min_border=0,
                            outline_line_color=None)

    color_bar_plot.add_layout(col_bar, 'right')
    color_bar_plot.title.align = "center"
    color_bar_plot.title.text_font_size = '12pt'

    logging.info('Starting iterations by row')

    # cyclic object for iteratively repetition over scans
    cols_cycler = cycle(range(ncols))
    rows_cycler = cycle(range(nrows))

    for rs, cols in inp_subdf.iterrows():

        row_idx = next(rows_cycler)
        if row_idx == min_chan:
            col_idx = next(cols_cycler)

        curr_scan_no = int(cols.SCAN_NUMBER)
        curr_cbin_no = int(cols.chan_bin)

        flagged_pc = cols.flagged_pc

        # setting the position of the scan label
        # fchunk label should be at row [:+1, 0]
        # scan number should be at column -1,or 0 thus [-1, :]
        dis = Div(text='S{}'.format(curr_scan_no),
                  width=plot_width - 10)
        mygrid[0, col_idx + 1] = dis

        # if curr_scan_no == min_scan:

        cfreq = np.mean(f_edges[curr_cbin_no])
        dis = Div(text='{:.3f}'.format(cfreq),
                  width=plot_width - 10,
                  style={'font-size': '100%'})
        mygrid[row_idx + 1, 0] = dis

        # percentage deviation mean
        pc_dev_mean = ((cols[mean_col_a] - mid_mean) / mid_mean) * 100
        source = ColumnDataSource(data={'mean': [cols[mean_col_a]],
                                        'mid_mean': [mid_mean],
                                        'mean_anorm': [pc_dev_mean],
                                        'var': [cols[var_col_a]],
                                        'std': [np.abs(np.log(np.sqrt(cols[var_col_a])))],
                                        'pcf': [flagged_pc]})

        p = Plot(background_fill_alpha=0.95,
                 x_range=xdr, y_range=xdr,
                 plot_width=plot_width, plot_height=plot_height,
                 y_scale=LinearScale(), x_scale=LinearScale(),
                 background_fill_color="#efe8f2")
        circle = Circle(x='mean', y='mean',
                        radius='std',
                        fill_alpha=0.8,
                        line_color=None,
                        fill_color={'field': 'mean_anorm',
                                    'transform': col_mapper})
        flag_circle = Circle(x='mean', y='mean',
                             radius=((source.data['pcf'][0] *
                                      source.data['std'][0]) * 0.01),
                             fill_alpha=0.8, line_color=None,
                             fill_color='white')

        p.add_glyph(source, circle)
        p.add_glyph(source, flag_circle)

        # setting the plot ticker
        xticker = BasicTicker()
        yticker = BasicTicker()

        if curr_scan_no == max_scan:
            xaxis = LinearAxis(major_tick_out=1,
                               minor_tick_out=1,
                               axis_line_width=1)

            xticker = xaxis.ticker

        if curr_cbin_no == min_chan:
            yaxis = LinearAxis(major_tick_out=1,
                               minor_tick_out=1,
                               axis_line_width=1)

            yticker = yaxis.ticker

        hover = bk.models.HoverTool(tooltips=[('μ', '@mean'),
                                              ('variance', '@var'),
                                              ('flagged %', '@pcf'),
                                              ("μ-mid_μ%",
                                               '@mean_anorm'),
                                              ('mid_μ', '@mid_mean')
                                              ])

        p.add_tools(PanTool(), hover, WheelZoomTool(), ResetTool())
        p.on_event(Tap, partial(on_click_callback, inp_df,
                                curr_scan_no, curr_cbin_no, doplot))

        # add the divs from the 2nd element to the 2nd last element of the
        mygrid[row_idx + 1, col_idx + 1] = p

    grid_plots = grid(children=mygrid.tolist())
    final_plot = gridplot([[grid_plots, color_bar_plot]])
    final_plot.name = 'final_plot'
    logging.info("Adding plot to root")

    document.add_root(final_plot)


#####################################################################
################# Main function starts here ##########################

#ms_name = "/home/andati/measurement_sets/1491291289.1ghz.1.1ghz.4hrs.ms"
ms_name = "/home/andati/tutorials/vla_continuum_tut/3c391_ctm_mosaic_10s_spw0.ms"
bin_width = 64
fid = 1
corr = 0
ddid = 0

data_col = 'CORRECTED_DATA'
doplot = 'ap'


# desired columns from the data
columns = [data_col, 'TIME', 'FLAG', 'SCAN_NUMBER', 'ANTENNA1', 'ANTENNA2']
ms_obj = get_ms(ms_name, fid=fid, ddid=ddid, data_col=data_col, where=None)[0]

# condensed ms with the selected columns
sub_ms = ms_obj[columns]

# convert the data into amp/ real/ phase/ imag and return on the relevant
# data that is already chunked

ready_ms_obj = process(sub_ms, corr=corr, data_column=data_col, ptype=doplot,
                       flag=True, bin_size=bin_width)


###################################################################
################# some global variables ##########################
# document object
document = curdoc()

# frequencies in the specified spw
freqs = (ut.get_frequencies(ms_name, spwid=ddid) / 1e9).data.compute()

# frequency edges depending on the bin size
f_edges = split_freqs(freqs, bin_width)

# ncols correspond to the number of scans
ncols = np.unique(ready_ms_obj.SCAN_NUMBER).size

"""
nrows corresponds to the number of chunks in the chan dimension.
chunksizes for each dimension returned as a tuple of the form: 
        ((chunk, size, dim, 1), (chunk, size, dim,2))
checking the flag column because it is chunked like the data
"""
nrows = len(ready_ms_obj.FLAG.data.chunks[-1])

# get the size of the current screen
plot_width, plot_height = calc_plot_dims(nrows, ncols)


# converting to data frame to allow for leveraging of dask data frame #partitions
# Since data is already chunked in xarray, the number of partitions
# corresponds to the number of chunks which makes it easier for now.
sel_df = ready_ms_obj.to_dask_dataframe()

# add chunk index numbers to the partitions
nddf = sel_df.map_partitions(compute_idx, bin_width)

sel_cols = ['SCAN_NUMBER', 'FLAG', 'chan_bin']

# additional columns that will be selected
additional_cols = []

if doplot == 'ap':
    sel_cols.extend(['Amplitude', 'Phase'])
    additional_cols.extend(['adatmean', 'adatvar',
                            'pdatmean', 'pdatvar', 'flagged_pc'])
    # save the data types of amplitude
    f64 = sel_df.dtypes.to_dict()['Amplitude']
elif doplot == 'ri':
    sel_cols.extend(['Real', 'Imaginary'])
    additional_cols.extend(['rdatmean', 'rdatvar',
                            'idatmean', 'idatvar', 'flagged_pc'])
    f64 = sel_df.dtypes.to_dict()['Real']

# column names forming the meta model=> expected output column names
fdf_col_names = sel_cols + additional_cols

# data types for all those output columns in the meta info
# dask needs to know what kind of data is expected after groupby apply
fdf_dtypes = nddf[sel_cols].dtypes.to_list() + [f64] * len(additional_cols)

# groupby meta; without this the code breaks :()
meta = [(key, value) for key, value in zip(fdf_col_names, fdf_dtypes)]

# grouping by chan_bin and scan_number and select
# create_mv calculates the mu & variances of the data for each of the chunks
res = nddf.groupby(['SCAN_NUMBER', 'chan_bin'])[sel_cols].apply(creat_mv,
                                                                doplot,
                                                                meta=meta)
make_plot(res, doplot)
