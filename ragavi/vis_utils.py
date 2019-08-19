import logging
import os
import sys
import textwrap
import warnings

import dask.array as da
import numpy as np
import pyrap.quanta as qa
import xarray as xa
import xarrayms as xm

from dask import delayed, compute
from datetime import datetime
from pyfiglet import Figlet


########################################################################
####################### Computation Functions ##########################


def calc_amplitude(ydata):
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


def calc_imaginary(ydata):
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


def calc_real(ydata):
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


def calc_phase(ydata, wrap=True):
    """Convert complex data to angle in degrees
    Inputs
    ------
    ydata: xarray DataArray
           y-axis data to be processed
    wrap: bool
          whether to wrap angles between 0 and 2pi
    Outputs
    -------
    phase: xarray DataArray
           y-axis data converted to degrees
    """
    phase = xa.apply_ufunc(da.angle, ydata,
                           dask='allowed', kwargs=dict(deg=True))
    if wrap:
        # using an alternative method to avoid warnings
        try:
            phase = phase.reduce(np.unwrap)
        except TypeError:
            # this is for python2 compat
            phase = xa.apply_ufunc(np.unwrap, phase, dask='allowed')
    return phase


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
    u = uvw.isel(uvw=0)
    v = uvw.isel(uvw=1)
    uvdist = da.sqrt(da.square(u) + da.square(v))
    return uvdist


def calc_uvwave(uvw, freq):
    """
        Calculate uv distance in wavelength for availed frequency.
        This function also calculates the corresponding wavelength. Uses output from *calc_uvdist*

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


########################################################################
####################### Get subtables ##################################


def get_antennas(ms_name):
    """Function to get antennae names from the ANTENNA subtable.
    Inputs
    ------
    ms_name: str
             Name of measurement set

    Outputs
    -------
    ant_names: xarray DataArray
               An xarray Data array containing names for all the antennas available.

    """
    subname = "::".join((ms_name, 'ANTENNA'))
    ant_subtab = list(xm.xds_from_table(subname, ack=False))
    ant_subtab = ant_subtab[0]
    ant_names = ant_subtab.NAME
    # ant_subtab('close')
    return ant_names


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


def get_frequencies(ms_name, spwid=slice(0, None), chan=slice(0, None)):
    """Function to get channel frequencies from the SPECTRAL_WINDOW subtable.
    Inputs
    ------
    ms_name: str
             Name of measurement set
    spwid: int
           Spectral window id number. Defaults to 0
    chan: slice / numpy array
          A slice object or numpy array to select some or all of the channels
          Default is all the channels
    Outputs
    -------
    freqs: xarray DataArray
           Channel centre frequencies for specified spectral window.
    """
    subname = "::".join((ms_name, 'SPECTRAL_WINDOW'))
    spw_subtab = list(xm.xds_from_table(subname, group_cols='__row__',
                                        ack=False))

    if len(spw_subtab) == 1:
        # select the desired spectral windows using slicer
        spw = spw_subtab[spwid.start]
    else:
        # if multiple SPWs,concatenate all the items in the list of SPWs to
        # form a single dataset and then extract the channels
        spw = xa.concat(spw_subtab, 'row')
        spw = spw.sel(row=spwid)

    if spw.CHAN_FREQ.size == 1:
        freqs = spw.CHAN_FREQ
    else:
        freqs = spw.CHAN_FREQ[chan]

    return freqs


def get_polarizations(ms_name):
    """
        Get the type of correlation in the measurement set

    Inputs
    ------
    ms_name: str
             Name of MS containing its path

    Outputs
    -------
    cor2stokes: list
                Returns a list containing the types of correlation
    """
    # Stokes types in this case are 1 based and NOT 0 based.
    stokes_types = ['I', 'Q', 'U', 'V', 'RR', 'RL', 'LR', 'LL', 'XX', 'XY',
                    'YX', 'YY', 'RX', 'RY', 'LX', 'LY', 'XR', 'XL', 'YR',
                    'YL', 'PP', 'PQ', 'QP', 'QQ', 'RCircular', 'LCircular',
                    'Linear', 'Ptotal', 'Plinear', 'PFtotal', 'PFlinear',
                    'Pangle']

    subname = "::".join((ms_name, 'POLARIZATION'))
    pol_subtable = list(xm.xds_from_table(subname, ack=False))[0]

    # offset the acquired corr type by 1 to match correctly the stokes type
    corr_types = pol_subtable.CORR_TYPE.sel(row=0).data.compute() - 1
    cor2stokes = []

    # Select corr_type name from the stokes types
    cor2stokes = [stokes_types[typ] for typ in corr_types]

    return cor2stokes


# some common rows


def get_errors(xds_table_obj, corr=0, chan=slice(0, None)):
    """Function to get error data from PARAMERR column.
    Inputs
    ------
    table_obj: pyrap table object.

    Outputs
    errors: ndarray
            Error data.
    """
    errors = xds_table_obj.PARAMERR.sel(dict(corr=corr, chan=chan))
    return errors


def get_flags(xds_table_obj, corr=None, chan=slice(0, None)):
    """ Get Flag values from the FLAG column
        Allow for selections in the channel dimension or the correlation dimension
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
        return flags.sel(chan=chan)
    else:
        flags = flags.sel(dict(corr=corr, chan=chan))
    return flags


########################################################################
####################### Some utils for use #############################

def name_2id(tab_name, field_name):
    """Translate field name to field id
    Inputs
    -----
    tab_name: str
              Table name
    field_name: string
         Field ID name to convert
    Outputs
    -------
    field_id: int
              Integer field id
    """
    field_names = get_fields(tab_name).data.compute()

    # make the sup field name uppercase
    field_name = field_name.upper()

    if field_name in field_names:
        field_id = np.where(field_names == field_name)[0][0]
        return int(field_id)
    else:
        return -1


def resolve_ranges(inp):
    """Create a TAQL string that can be parsed given a range of values
    Inputs
    ------
    inp: str
         A range of values to be constructed. Can be in the form of:
         "5", "5,6,7", "5~7" (inclusive range), "5:8" (exclusive range),
         "5:" (from 5 to last)

    Outputs
    -------
    res: str
         Interval string conforming to TAQL sets and intervals as shown in
        https://casa.nrao.edu/aips2_docs/notes/199/node5.html#TAQL:EXPRESSIONS
    """

    if '~' in inp:
        # create expression fro inclusive range
        # to form a curly bracket string we need {{}}
        res = "[{{{}}}]".format(inp.replace('~', ','))
    else:

        # takes care of 5:8 or 5,6,7 or 5:
        res = '[{}]'.format(inp)
    return res


def slice_data(inp):
    """Creates a slicer for an array. To be used to get a data subset.

    Inputs
    ------
    inp: str
         This can be of the form "5", "10~20" (10 to 20 inclusive), "10:21" (same), "10:" (from 10 to end), ":10:2" (0 to 9 inclusive, stepped by 2), "~9:2" (same)

    Outputs
    -------
    sl : slice object
         slicer for an iterable object

    """
    if inp is None:
        start = 0
        stop = None
        sl = slice(start, stop)
        return sl

    if inp.isdigit():
        sl = int(inp)
        return sl
    # check where the string is a comma separated list
    if ',' not in inp:
        if '~' in inp:
            splits = inp.replace('~', ':').split(':')
        else:
            splits = inp.split(':')
        splits = [None if x == '' else int(x) for x in splits]

        len_splits = len(splits)
        start = None
        stop = None
        step = 1

        if len_splits == 1:
            start = int(splits[0])
        elif len_splits == 2:
            start, stop = splits
        elif len_splits == 3:
            start, stop, step = splits

        # since ~ only alters the end value, we can then only change the stop
        # value
        if '~' in inp:
            stop += 1

        sl = slice(start, stop, step)
    else:
        # assuming string is a comma separated list
        inp = inp.replace(' ', '')
        splits = [int(x) for x in inp.split(',')]
        sl = np.array(splits)

    return sl

########################################################################
######################## conversions ###################################


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

    # difference between MJD and unix time i.e. munix = MJD - unix_time
    # so unix_time = MJD - munix
    # munix = 3506716800.0

    # get first time instance
    init_time = xdata[0].data.compute()

    # get number of seconds from initial time
    # not using int_time coz its a single element rather than array
    # using a single element throws warnings
    time_diff = xdata - xdata[0]

    # convert the initial time to unix time in seconds
    init_time = qa.quantity(init_time, 's').to_unix_time()

    # Add the initial unix time in seconds to get actual time progrssion
    unix_time = time_diff + init_time

    # unix_time = da.array(unix_time, dtype='datetime64[s]')
    unix_time = unix_time.astype('datetime64[s]')

    return unix_time


########################################################################
####################### Logger #########################################


def wrap_warning_text(message, category, filename, lineno, file=None,
                      line=None):
    wrapper = textwrap.TextWrapper(initial_indent=''.rjust(51),
                                   break_long_words=True,
                                   subsequent_indent=''.rjust(49),
                                   width=160)
    message = wrapper.fill(str(message))
    return "%s:%s:\n%s:\n%s" % (filename, lineno,
                                category.__name__.rjust(64), message)


warnings.formatwarning = wrap_warning_text


def config_logger():
    """This function is used to configure the logger for ragavi and catch
        all warnings output by sys.stdout.
    """
    logfile_name = 'ragavi.log'

    # capture only a single instance of a matching repeated warning
    warnings.filterwarnings('module')

    # capture warnings from all modules
    logging.captureWarnings(True)

    try:
        cols, rows = os.get_terminal_size(0)
    except:
        # for python2
        cols, rows = (100, 100)

    # create logger named ragavi
    logger = logging.getLogger('ragavi')
    logger.setLevel(logging.INFO)

    # warnings logger
    w_logger = logging.getLogger('py.warnings')
    w_logger.setLevel(logging.INFO)

    # console handler
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(logfile_name)

    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # setting the format for the logging messages
    start = " (O_o) ".center(cols, "=")

    c_formatter = logging.Formatter(
        """%(asctime)s - %(name)-12s - %(levelname)-10s - %(message)s""", datefmt='%d.%m.%Y@%H:%M:%S')
    f_formatter = logging.Formatter(
        """%(asctime)s - %(name)-12s - %(levelname)-10s - %(message)s""" , datefmt='%d.%m.%Y@%H:%M:%S')

    c_handler.setFormatter(c_formatter)
    f_handler.setFormatter(f_formatter)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    w_logger.addHandler(f_handler)

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


########################################################################
############################# Welcome ##################################
def welcome():
    """Welcome to ragavi"""
    print('\n\n')
    #print("_*+_" * 23)
    #print("Welcome to ")
    print(Figlet(font='nvscript').renderText('ragavi'))
    #print("_*+_" * 23)
    print('\n\n')
