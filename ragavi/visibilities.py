import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import glob
import pylab
#import sys
from dask import compute, delayed
from pyrap.tables import table, tablecolumn, tableiter
from optparse import OptionParser

"""
This is the visibilty part of the RAGaVI tool.
It is used to plot visibilities but can only do so in an averaged mode
because while processing the data might take a shorter time due
to its optimised nature, plotting large quantities of data can be tasking to machine because of the graphics aspect.

CAVEATS
========
1. Works for all the available x-axes except phase and imaginary x-axis because of dimensionality issues. i.e. these are 3 dimensional while the plotter only accepts maximum of 2D for the x-axis.

2. Plots with uvwave for x-xaxis maybe inaccurate and will need to be revised due to a glitch in the code

3. Iterations may be done through calling the iteration functions defined, which include iterating over scans, correlations and spectral windows

4. Current plotter used is matplotlib and this was done for demonstration purposes. It is intended that the plots will be rendered using bokeh plotting library
"""


def get_antnames(table_obj):
    """
        Pull antenna  names from ANTENNA subtable
    """
    ant_subtable = table(table_obj.getkeyword('ANTENNA'), ack=False)
    ant_names = ant_subtable.getcol('NAME')
    ant_subtable.close()
    return ant_names


def get_polarizations(table_obj):

    # Stokes types in this case are 1 based and NOT 0 based.
    stokes_types = ['I', 'Q', 'U', 'V', 'RR', 'RL', 'LR', 'LL', 'XX', 'XY', 'YX', 'YY', 'RX', 'RY', 'LX', 'LY', 'XR', 'XL', 'YR',
                    'YL', 'PP', 'PQ', 'QP', 'QQ', 'RCircular', 'LCircular', 'Linear', 'Ptotal', 'Plinear', 'PFtotal', 'PFlinear', 'Pangle']
    pol_subtable = table(table_obj.getkeyword('POLARIZATION'), ack=False)

    # ofsetting the acquired corr typeby one to match correctly the stokes type
    corr_types = pol_subtable.getcell('CORR_TYPE', 0) - 1
    cor2stokes = []
    for typ in corr_types:
        cor2stokes.append(stokes_types[typ])
    return cor2stokes


def get_flagdata(table_obj):
    flags = table_obj.getcol('FLAG')
    return flags


def get_frequencies(table_obj):
    """
        Pull channel frequencies from SPECTRAL WINDOW subtable
    """
    freq_subtable = table(table_obj.getkeyword(
                          'SPECTRAL_WINDOW'), ack=False)
    frequencies = freq_subtable.getcell('CHAN_FREQ', 0)
    freq_subtable.close()
    return frequencies


def select_correlation(data, corr):
    # Assuming a 3 dimensional input data.
    return data[:, :, corr]


def calc_uvdist(uvw):
    """
        Calculate uv distance in metres
    """
    u = uvw[:, 0]
    v = uvw[:, 1]
    uvdist = np.sqrt(np.square(u) + np.square(v))
    return uvdist


def calc_uvwave(uvw, freqs):
    """
        Generate uv distance in wavelength for each wavelength
        This function also calculates the wavelengths corresponding
        to the frequency of each channel
    """
    # speed of light
    C = 3e8

    #wavelength = velocity / frequency
    wavelengths = C / freqs
    uvdist = calc_uvdist(uvw)

    for wavelength in wavelengths:
        yield uvdist / wavelength


def get_xaxis_data(table_obj, xaxis):
    #global ms_name

    xaxis_names = ['antenna1', 'antenna2', 'scan', 'frequency', 'phase', 'imaginary',
                   'uvdistance', 'uvwave', 'channel', 'time']

    # confirm validity of axis name
    if xaxis not in xaxis_names:
        print "Invalid x axis name.\nMust be: ", xaxis_names
        return

    # get data from freq subtable
    if xaxis == 'frequency' or xaxis == 'channel':
        frequencies = get_frequencies(table_obj)
        xdata = frequencies

    elif xaxis == 'scan':
        xdata = table_obj.getcol('SCAN_NUMBER')
    elif xaxis == 'time':
        xdata = table_obj.getcol('TIME')
    elif xaxis == "phase" or xaxis == 'imaginary':
        xdata = table_obj.getcol('DATA')

    elif xaxis == 'antenna1':
        ant1 = table_obj.getcol('ANTENNA1')
        xdata = ant1
    elif xaxis == 'antenna2':
        ant2 = table_obj.getcol('ANTENNA2')
        xdata = ant2
    elif xaxis == 'uvdistance' or xaxis == 'uvwave':
        xdata = table_obj.getcol('UVW')

    return xdata


def get_yaxis_data(table_obj, yaxis):
    global ms_name

    yaxis_names = ["amplitude", "phase", "real", "imaginary"]
    if yaxis not in yaxis_names:
        print "Invalid x axis name.\nMust be: ", yaxis_names
        return
    else:
        ydata = table_obj.getcol('DATA')
        return ydata


def prep_xdata(table_obj, xdata, xaxis):
    if xaxis == 'antenna1' or xaxis == 'antenna2' or xaxis == 'scan':
        prepdx = xdata
    elif xaxis == 'phase':
        prepdx = np.unwrap(np.angle(xdata, deg=True))
    elif xaxis == 'time':
        prepdx = xdata - xdata[0]
    elif xaxis == 'frequency':
        prepdx = xdata
    elif xaxis == 'channel':
        prepdx = np.arange(xdata.size)
    elif xaxis == 'imaginary':
        prepdx = np.imag(xdata)
    elif xaxis == 'uvdistance':
        prepdx = calc_uvdist(xdata)
    elif xaxis == 'uvwave':
        freqs = get_frequencies(table_obj)

        prepdx = next(uvdist)

    return prepdx


def prep_ydata(table_obj, ydata, yaxis, xaxis=None, corr=0, showFlagged=False):

    num_corrs = len(get_polarizations(table_obj))
    if corr > num_corrs - 1:
        print "Invalid correlation"
        return

    flag_mask = get_flagdata(table_obj)
    if corr is not None:
        ydata = ydata = ydata[:, :, corr]
        flag_mask = flag_mask[:, :, corr]

    if yaxis == 'amplitude':
        prepdy = np.abs(ydata)
    elif yaxis == 'phase':
        prepdy = np.unwrap(np.angle(ydata, deg=True))
    elif yaxis == 'imaginary':
        prepdy = np.imag(ydata)
    elif yaxis == 'real':
        prepdy = np.real(ydata)

    if showFlagged:
        prepdy = np.ma.masked_array(prepdy, flag_mask)
    if xaxis == 'frequency' or xaxis == 'channel':
        prepdy = prepdy.transpose((1, 0))

    return prepdy


def chunk_data(data, bin_size=0, xdata=None, dim='row'):
    """
        data: numpy.ndarray or dask.array

        Input
        -----
            xdata: array
                   Optional, must only be used if chunking yaxisdata for purposes of ensuring the chunks for the x component are thesame.   


        Output
        ------
        dask array

        # row corresponds to baselnine
        # chan/channel to frequency
        # time is self explanatory
        !!!need to determin the dimension coord for time!!!!

        bin_size must be a default value passed into the function 
    """
    dims = {'row': 0, 'chan': 1, 'time': 2}

    # Handle numpy and dask arrays differently. Dask arrays need to maintain
    # their chunk structure
    if type(data) is np.ndarray or type(data) is np.ma.core.MaskedArray:
        data_shape = list(data.shape)
    else:
        data_shape = list(data.chunks)

    if bin_size == 0:
        chunkd_array = da.from_array(data, chunks=data_shape)

    else:
        if data.ndim < 2:
            chunkd_array = da.from_array(data, chunks=bin_size)
        else:
            data_shape[dims[dim]] = bin_size
            if xdata is not None:
                x_chunks = xdata.chunks[0]
                data_shape[0] = x_chunks

            chunkd_array = da.from_array(data, chunks=data_shape)

    return chunkd_array


def average_xdata(chunkd_x, dim=None):
    """
        Input
        -----
        chunkd_x: dask.array
                Must be a 1d array
        # row corresponds to baselnine
        # chan/channel to frequency
        # time is self explanatory
    """
    dims = {'row': 0, 'chan': 1, 'time': 2}

    # get the number of partitions in each dimension
    rows = chunkd_x.numblocks[0]
    intermediates = []

    for row in range(rows):
        if dim is None:
            intermediates.append(chunkd_x.blocks[row])
        else:
            avgd_x = chunkd_x.blocks[row].mean(dims[dim])
            intermediates.append(avgd_x)

    return intermediates


def average_ydata(chunkd_y, dim='row'):
    """
        # row corresponds to baselnine
        # chan/channel to frequency
        # time is self explanatory
        This data must be averaged for everyone's computer's safety :)
    """
    dims = {'row': 0, 'chan': 1, 'time': 2}

    # getting the number of partions in each dimension
    rows, cols = chunkd_y.numblocks

    intermediates = []

    for row in range(rows):
        for col in range(cols):
            avgd_y = chunkd_y.blocks[row, col].mean(axis=dims[dim])
            intermediates.append(avgd_y)
    return intermediates


def iter_scan(table_obj, xaxis, yaxis, scans=-1):

    for sub in table_obj.iter(columnnames='SCAN_NUMBER'):
        # get the appropriate data for both x and y and plot

        p = integrator(sub, xaxis, yaxis, xbinsize=X_BINSIZE,
                       ybinsize=Y_BINSIZE, xavd=XAVG_DIM, yavd=YAVG_DIM)
        scan_no = sub.getcell('SCAN_NUMBER', 0)
        title = "Scan {}: {} vs {}".format(
            scan_no, yaxis.capitalize(), xaxis.capitalize())
        p.title(title)
        p.show()


def counter(x):
    """A generator function to keep count of the items being generated
    """
    for i in range(x):
        yield i


def iter_specwin(table_obj, xaxis, yaxis):

    for sub in table_obj.iter(columnnames='DATA_DESC_ID'):
        # get the appropriate data for both x and y and plot

        p = integrator(sub, xaxis, yaxis, xbinsize=X_BINSIZE,
                       ybinsize=Y_BINSIZE, xavd=XAVG_DIM, yavd=YAVG_DIM)
        specwin_no = sub.getcell('DATA_DESC_ID', 0)
        title = "Spectral Window {}: {} vs {}".format(
            specwin_no, yaxis.capitalize(), xaxis.capitalize())
        p.title(title)
        p.show()


def iter_correlation(table_obj, xaxis, yaxis):
    data = table_obj.getcell('DATA', 0)
    chans, corrs = data.shape
    corr_types = get_polarizations(ms)
    for corr in range(corrs):
        p = integrator(ms, xaxis, yaxis, xbinsize=X_BINSIZE,
                       ybinsize=Y_BINSIZE, xavd=XAVG_DIM, yavd=YAVG_DIM,
                       corr=corr)
        title = "Correlation {}: {} vs {}".format(
            corr_types[corr], yaxis.capitalize(), xaxis.capitalize())
        p.title(title)
        p.show()


def plot(plt_obj, xdata, ydata, xaxis, yaxis, y_nblocks, y_npartitions, iter=None):
    yaxis = yaxis.capitalize()
    xaxis = xaxis.capitalize()
    title = yaxis + " vs " + xaxis

    plt.ylabel(yaxis)
    plt.xlabel(xaxis)
    plt.title(title)

    # getting the total number of partitions in yaxis and blocks in each
    # partition
    count = counter(y_npartitions)
    if len(y_nblocks) == 2:
        rows, cols = y_nblocks
        for row in range(rows):
            for col in range(cols):
                c = next(count)
                plt_obj.plot(xdata[row].compute(), ydata[c].compute(),
                             markersize=0.9, marker='o', linewidth=0)
    elif len(y_nblocks) == 1:
        # checking for number of dimensions in the data
        rows = y_nblocks
        for row in range(rows):
            plt_obj.plot(xdata[row].compute(), ydata[row].compute(),
                         markersize=0.9, marker='o', linewidth=0)
    return plt_obj


def integrator(table_obj, xaxis, yaxis, xbinsize=0, ybinsize=0, xavd=None, yavd='row', average=True, corr=None):
    """A 'message passing' function to integrate all functions
    order

    """

    x_data = get_xaxis_data(table_obj, xaxis)
    y_data = get_yaxis_data(table_obj, yaxis)

    # declaring uvdist generator object as a global resource o allow for
    # proper formation of the uv waves
    global uvdist
    freqs = get_frequencies(table_obj)
    uvdist = calc_uvwave(x_data, freqs)

    if corr is not None:
        y_data = select_correlation(y_data, corr)

    if xaxis == 'uvwave':
        # calculating uvwave for each wavelength in the bandwidth
        for freq in range(len(get_frequencies(table_obj))):
            prepd_x = prep_xdata(table_obj, x_data, xaxis)
            prepd_y = prep_ydata(table_obj, y_data, yaxis, xaxis=xaxis)
            chunkd_x = chunk_data(prepd_x, bin_size=xbinsize, dim=xavd)
            chunkd_y = chunk_data(prepd_y, bin_size=ybinsize, dim=yavd,
                                  xdata=chunkd_x)

            # Store the number of blocks in each dimension of the data
            Y_NBLOCKS = chunkd_y.numblocks
            # Store the total number of partions in the data
            Y_NPARTITIONS = chunkd_y.npartitions

            avgd_x = average_xdata(chunkd_x, xavd)
            avgd_y = average_ydata(chunkd_y, yavd)
            plots = plot(plt, avgd_x, avgd_y, xaxis,
                         yaxis, Y_NBLOCKS, Y_NPARTITIONS)

    else:
        prepd_x = prep_xdata(table_obj, x_data, xaxis)
        prepd_y = prep_ydata(table_obj, y_data, yaxis, xaxis=xaxis)
        chunkd_x = chunk_data(prepd_x, bin_size=xbinsize, dim=xavd)
        chunkd_y = chunk_data(prepd_y, bin_size=ybinsize, dim=yavd,
                              xdata=chunkd_x)

        # Store the number of blocks in each dimension of the data
        Y_NBLOCKS = chunkd_y.numblocks
        # Store the total number of partions in the data
        Y_NPARTITIONS = chunkd_y.npartitions

        avgd_x = average_xdata(chunkd_x, xavd)
        avgd_y = average_ydata(chunkd_y, yavd)
        plots = plot(plt, avgd_x, avgd_y, xaxis,
                     yaxis, Y_NBLOCKS, Y_NPARTITIONS)
    return plots


parser = OptionParser(usage='%prog [options] tablename')
parser.add_option('-f', '--field', dest='field',
                  help='Field ID to plot (default = 0)', default=0)
parser.add_option('-d', '--doplot', dest='doplot',
                  help='Plot complex values as amp and phase (ap) or real and imag (ri) (default = ap)', default='ap')
parser.add_option('-a', '--ant', dest='plotants',
                  help='Plot only this antenna, or comma-separated list of antennas', default=[-1])
parser.add_option('-c', '--corr', dest='corr',
                  help='Correlation index to plot (usually just 0 or 1, default = 0)', default=0)
parser.add_option('--t0', dest='t0',
                  help='Minimum time to plot (default = full range)', default=-1)
parser.add_option('--t1', dest='t1',
                  help='Maximum time to plot (default = full range)', default=-1)
parser.add_option('--yu0', dest='yu0',
                  help='Minimum y-value to plot for upper panel (default = full range)', default=-1)
parser.add_option('--yu1', dest='yu1',
                  help='Maximum y-value to plot for upper panel (default = full range)', default=-1)
parser.add_option('--yl0', dest='yl0',
                  help='Minimum y-value to plot for lower panel (default = full range)', default=-1)
parser.add_option('--yl1', dest='yl1',
                  help='Maximum y-value to plot for lower panel (default = full range)', default=-1)
parser.add_option('--cmap', dest='mycmap',
                  help='Matplotlib colour map to use for antennas (default = coolwarm)', default='coolwarm')
parser.add_option('--ms', dest='myms',
                  help='Measurement Set to consult for proper antenna names', default='')
parser.add_option('-p', '--plotname', dest='pngname',
                  help='Output PNG name (default = something sensible)', default='')
parser.add_option('-i', '--inputms', dest='ms_name')

(options, args) = parser.parse_args()


field = int(options.field)
doplot = options.doplot
plotants = options.plotants
corr = int(options.corr)
t0 = float(options.t0)
t1 = float(options.t1)
yu0 = float(options.yu0)
yu1 = float(options.yu1)
yl0 = float(options.yl0)
yl1 = float(options.yl1)
mycmap = options.mycmap
myms = options.myms
pngname = options.pngname
ms_name = options.ms_name
#showFlagged = options.showFlagged

# avgeraging constants
global X_BINSIZE, Y_BINSIZE, XAVG_DIM, YAVG_DIM, XAXIS, YAXIS
X_BINSIZE = 0
Y_BINSIZE = 100
XAVG_DIM = None
YAVG_DIM = 'chan'
XAXIS = 'time'
YAXIS = 'phase'

#['antenna1', 'antenna2', 'scan', 'frequency', 'phase', 'imaginary', 'uvdistance', 'uvwave', 'channel', 'time']
#ms_name = './ms_tab;/1491291289.1ghz.1.1ghz.4hrs.ms'
ms = table(ms_name, ack=False)

p = integrator(ms, XAXIS, YAXIS, xbinsize=X_BINSIZE,
               ybinsize=Y_BINSIZE, xavd=XAVG_DIM, yavd=YAVG_DIM)
p.show()

#iter_scan(ms, XAXIS, YAXIS)
#iter_specwin(ms, XAXIS, YAXIS)
#iter_correlation(ms, XAXIS, YAXIS)
