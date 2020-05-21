# -*- coding: utf-8 -*-
import logging
import os
import re
import sys
import textwrap
import warnings

import dask.array as da
import numpy as np

import bokeh.palettes as pl
import colorcet as cc
import xarray as xr
import daskms as xm

from datetime import datetime
from time import time

########################################################################
####################### Computation Functions ##########################


def calc_amplitude(ydata):
    """Convert complex data to amplitude (absolute value)

    Parameters
    ----------
    ydata : :obj:`xarray.DataArray`
        y axis data to be processed

    Returns
    -------
    amplitude : :obj:`xarray.DataArray`
        :attr:`ydata` converted to an amplitude
    """
    logger.debug("Setting up amplitude")

    amplitude = da.absolute(ydata)
    return amplitude


def calc_imaginary(ydata):
    """Extract imaginary part from complex data

    Parameters
    ----------
    ydata : :obj:`xarray.DataArray`
        y-axis data to be processed

    Returns
    -------
    imag : :obj:`xarray.DataArray`
        Imaginary part of :attr:`ydata`
    """
    logger.debug("Setting up imaginary")

    imag = ydata.imag
    return imag


def calc_real(ydata):
    """Extract real part from complex data

    Parameters
    ----------
    ydata : :obj:`xarray.DataArray`
        y-axis data to be processed

    Returns
    -------
    real : :obj:`xarray.DataArray`
        Real part of :attr:`ydata`
    """
    logger.debug("Setting up real")

    real = ydata.real
    return real


def calc_phase(ydata, unwrap=False):
    """Convert complex data to angle in degrees

    Parameters
    ----------
    wrap : :obj:`bool`
        whether to wrap angles between 0 and 2pi
    ydata : :obj:`xarray.DataArray`
        y-axis data to be processed

    Returns
    -------
    phase: `xarray.DataArray`
        :attr:`ydata` data converted to degrees
    """
    logger.debug("Setting up wrapped phase")

    # np.angle already returns a phase wrapped between (-pi and pi]
    # https://numpy.org/doc/1.18/reference/generated/numpy.angle.html
    phase = xr.apply_ufunc(da.angle, ydata,
                           dask="allowed", kwargs=dict(deg=True))

    if unwrap:
        # using an alternative method to avoid warnings
        logger.debug("Unwrapping enabled. Unwrapping angles")

        phase = phase.reduce(np.unwrap)
    return phase


def calc_uvdist(uvw):
    """ Calculate uv distance in metres

    Parameters
    ----------
    uvw : :obj:`xarray.DataArray`
        UVW column from measurement set

    Returns
    -------
    uvdist : :obj:`xarray.DataArray`
        uv distance in meters
    """
    logger.debug("Setting up UV Distance (metres)")
    u = uvw.isel(uvw=0)
    v = uvw.isel(uvw=1)
    uvdist = da.sqrt(da.square(u) + da.square(v))
    return uvdist


def calc_uvwave(uvw, freq):
    """Calculate uv distance in wavelength for availed frequency. This function also calculates the corresponding wavelength. Uses output from :func:`ragavi.vis_utils.calc_uvdist`

    Parameters
    ----------
    freq : :obj:`xarray.DataArray or :obj:`float`
        Frequency(ies) from which corresponding wavelength will be obtained.
    uvw : :obj:`xarray.DataArray`
        UVW column from the MS dataset

    Returns
    -------
    uvwave : :obj:`xarray.DataArray`
        uv distance in wavelength for specific frequency
    """

    logger.debug("Setting up UV Wavelengths (lambdas)")

    # speed of light
    C = 3e8

    # wavelength = velocity / frequency
    wavelength = (C / freq)

    # add extra dimension
    uvdist = calc_uvdist(uvw)
    uvdist = uvdist.expand_dims({"chan": 1}, axis=1)
    uvwave = uvdist / wavelength
    return uvwave


def calc_unique_bls(n_ants=None):
    """Calculate number of unique baselines
    Parameters
    ----------
    n_ants : :obj:`int`
        Available antennas
    Returns
    -------
    pq: :obj:`int`
        Number of unique baselines

    """

    pq = int(0.5 * n_ants * (n_ants - 1))
    logger.debug(f"Number of unique baselines: {str(pq)}.")
    return pq


########################################################################
####################### Get subtables ##################################


def get_antennas(ms_name):
    """Function to get antennae names from the ANTENNA subtable.

    Parameters
    ----------
    ms_name : :obj:`str`
        Name of MS or table

    Returns
    -------
    ant_names : :obj:`xarray.DataArray`
        A :obj:`xarray.DataArray` containing names for all the antennas available.

    """
    logger.debug("Getting antenna names")

    subname = "::".join((ms_name, "ANTENNA"))
    ant_subtab = list(xm.xds_from_table(subname))
    ant_subtab = ant_subtab[0]
    ant_names = ant_subtab.NAME
    # ant_subtab("close")

    logger.debug(f"Antennas found: {str(ant_names.values)}")
    return ant_names


def get_fields(ms_name):
    """Get field names from the FIELD subtable.

    Parameters
    ----------
    ms_name : :obj:`str`
        Name of MS or table

    Returns
    -------
    field_names : :obj:`xarray.DataArray`
        String names for the available fields
    """
    logger.debug("Getting antenna names")

    subname = "::".join((ms_name, "FIELD"))
    field_subtab = list(xm.xds_from_table(subname))
    field_subtab = field_subtab[0]
    field_names = field_subtab.NAME

    logger.debug(f"Fields found: {str(field_names.values)}")
    return field_names


def get_frequencies(ms_name, spwid=None, chan=None, cbin=None):
    """Function to get channel frequencies from the SPECTRAL_WINDOW subtable

    Parameters
    ----------
    chan : :obj:`slice` or :obj:`numpy.ndarray`
        A slice object or numpy array to select some or all of the channels. Default is all the channels
    cbin: :obj:`int`
        Number of channels to be binned together. If a value is provided, averaging is assumed to be turned on
    ms_name : :obj:`str`
        Name of MS or table
    spwid : :obj:`int` of :obj:`slice`
        Spectral window id number. Defaults to 0. If slicer is specified, frequencies from a range of spectral windows will be returned.

    Returns
    -------
    frequencies : :obj:`xarray.DataArray`
        Channel centre frequencies for specified spectral window or all the frequencies for all spectral windows if one is not specified
    """

    logger.debug("Gettting Frequencies for selected SPWS and channels")

    subname = "::".join((ms_name, "SPECTRAL_WINDOW"))

    # if averaging is true, it shall be done before selection
    if cbin is None:
        spw_subtab = xm.xds_from_table(
            subname, group_cols="__row__",
            columns=["CHAN_FREQ", "CHAN_WIDTH", "EFFECTIVE_BW",
                     "REF_FREQUENCY", "RESOLUTION"])
        if chan is not None:
            spw_subtab = [_.sel(chan=chan) for _ in spw_subtab]
    else:
        from ragavi.averaging import get_averaged_spws
        # averages done per spectral window, scan and field
        logger.info("Channel averaging active")
        spw_subtab = get_averaged_spws(subname, cbin, chan_select=chan)

    # concat all spws into a single data array
    frequencies = []
    for s in spw_subtab:
        frequencies.append(s.CHAN_FREQ)
        s.close()
    frequencies = xr.concat(frequencies, dim="row")

    if spwid is not None:
        # if multiple SPWs due to slicer, select the desired one(S)
        frequencies = frequencies.sel(row=spwid)

    logger.debug(
        f"Frequency table shape (spws, chans): {str(frequencies.shape)}")
    return frequencies


def get_polarizations(ms_name):
    """Get the type of polarizations available in the measurement set

    Parameters
    ----------
    ms_name: :obj:`str`
             Name of MS / table

    Returns
    -------
    cor2stokes: :obj:`list`
                Returns a list containing the types of correlation
    """

    logger.debug("Getting Stokes' types")
    # Stokes types in this case are 1 based and NOT 0 based.
    stokes_types = ["I", "Q", "U", "V", "RR", "RL", "LR", "LL", "XX", "XY",
                    "YX", "YY", "RX", "RY", "LX", "LY", "XR", "XL", "YR",
                    "YL", "PP", "PQ", "QP", "QQ", "RCircular", "LCircular",
                    "Linear", "Ptotal", "Plinear", "PFtotal", "PFlinear",
                    "Pangle"]

    subname = "::".join((ms_name, "POLARIZATION"))
    pol_subtable = list(xm.xds_from_table(subname))[0]

    # offset the acquired corr type by 1 to match correctly the stokes type
    corr_types = pol_subtable.CORR_TYPE.sel(row=0).data.compute() - 1
    cor2stokes = []

    # Select corr_type name from the stokes types
    cor2stokes = [stokes_types[typ] for typ in corr_types]

    logger.debug(f"Found stokes: {str(cor2stokes)}")

    return cor2stokes


def get_flags(xds_table_obj, corr=None, chan=slice(0, None)):
    """ Get Flag values from the FLAG column. Allow for selections in the channel dimension or the correlation dimension

    Parameters
    ----------
    corr : :obj:`int`
        Correlation number to select.
    xds_table_obj : :obj:`xarray.Dataset`
        MS as xarray dataset from xarrayms

    Returns
    -------
    flags : :obj:`xarray.DataArray`
        Data array containing values from FLAG column selected by correlation if index is available.

    """
    logger.debug("Getting flags")
    flags = xds_table_obj.FLAG

    if np.all(flags.values):
        logger.debug(f"All flags are active for {xds_table_obj.attrs}")

    if corr is None:
        return flags.sel(chan=chan)
    else:
        flags = flags.sel(dict(corr=corr, chan=chan))
    return flags


########################################################################
####################### Some utils for use #############################

def name_2id(tab_name, field_name):
    """Translate field name to field id

    Parameters
    ---------
    tab_name : :obj:str`
        MS or Table name
    field_name : :obj:`str`
        Field name to convert to field ID

    Returns
    -------
    field_id : :obj:`int`
        Integer field id
    """
    logger.debug("Converting field names to FIELD_IDs")

    field_names = get_fields(tab_name).data.compute()

    # make the sup field name uppercase
    field_name = field_name.upper()

    if field_name in field_names:
        field_id = np.where(field_names == field_name)[0][0]

        logger.debug(f"Field name {field_name} --> found in ID {field_id}")

        return int(field_id)
    else:
        logger.debug(f"FIELD_ID of {field_name} not found")
        return -1


def resolve_ranges(inp):
    """Create a TAQL string that can be parsed given a range of values

    Parameters
    ----------
    inp : :obj:`str`
        A range of values to be constructed. Can be in the form of: "5", "5,6,7", "5~7" (inclusive range), "5:8" (exclusive range), "5:" (from 5 to last)

    Returns
    -------
    res : :obj:`str`
        Interval string conforming to TAQL sets and intervals as shown in `Casa TAQL Notes <https://casa.nrao.edu/aips2_docs/notes/199/node5.html#TAQL:EXPRESSIONS>`_
    """

    if '~' in inp:
        # create expression fro inclusive range
        # to form a curly bracket string we need {{}}
        res = "[{{{}}}]".format(inp.replace('~', ','))
    else:

        # takes care of 5:8 or 5,6,7 or 5:
        res = "[{}]".format(inp)

    logger.debug(f"Resolved range {inp} --> {res} for TAQL selection")
    return res


def slice_data(inp):
    """Creates a slicer for an array. To be used to get a data subset such as correlation or channel subsets.

    Parameters
    ----------
    inp : :obj:`str`
        This can be of the form "5", "10~20" (10 to 20 inclusive), "10:21" (same), "10:" (from 10 to end), ":10:2" (0 to 9 inclusive, stepped by 2), "~9:2" (same)

    Returns
    -------
    sl : :obj:`slice`
        slicer for an iterable object

    """
    if inp is None:
        sl = slice(0, None)
    elif inp.isdigit():
        sl = int(inp)
    elif ',' in inp:
        # assume a comma separated list
        sl = np.array(inp.split(","), dtype=int)
    elif '~' in inp or ':' in inp:
        splits = re.split(r"(~|:)", inp)
        n_splits = []
        for x in splits:
            if x.isdigit():
                x = int(x)
            elif x == '':
                x = None
            n_splits.append(x)

        if '~' in n_splits:
            n_splits[n_splits.index('~') + 1] += 1
            n_splits = [':' if _ == '~' else _ for _ in n_splits]

        # get every second item on list
        sl = slice(*n_splits[::2])

    logger.debug(f"Created slicer / (fancy) index {str(sl)} from input {inp}")

    return sl

########################################################################
######################## conversions ###################################


def time_convert(xdata):
    """ Convert time from MJD to UTC time

    Parameters
    ----------
    xdata : :obj:`xarray.DataArray`
        TIME column from the MS or table xarray dataset in MJD format.

    Returns
    -------
    newtime : :obj:`xarray.DataArray`
        TIME column in a more human readable UTC format. Stored as :obj:`numpy.datetime` type.

    """
    import pyrap.quanta as qa

    """
    The difference between MJD and unix time i.e. munix = MJD - unix_time
    so unix_time = MJD - munix
    munix = 3506716800.0 = (40857 * 86400)

    The value 40587 is the number of days between the MJD epoch (1858-11-17) and the Unix epoch (1970-01-01), and 86400 is the number of seconds in a day
    """

    # get first time instance
    init_time = xdata[0].data.compute()

    # get number of seconds from initial time
    # not using int_time coz its a single element rather than array
    # using a single element throws warnings
    time_diff = xdata - xdata[0]

    # convert the initial time to unix time in seconds
    init_time = qa.quantity(init_time, 's').to_unix_time()

    # Add the initial unix time in seconds to get actual time progression
    unix_time = time_diff + init_time

    # unix_time = da.array(unix_time, dtype="datetime64[s]")
    unix_time = unix_time.astype("datetime64[s]")

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


def __config_logger(level="info"):
    """Configure the logger for ragavi and catch all warnings output by sys.stdout.
    """
    logfile_name = "ragavi.log"

    # numeric value of logging level
    level_num = getattr(logging, level.upper(), None)

    # capture only a single instance of a matching repeated warning
    warnings.filterwarnings("module")

    # capture warnings from all modules
    logging.captureWarnings(True)

    try:
        cols, rows = os.get_terminal_size(0)
    except:
        # for python2
        cols, rows = (100, 100)

    # create logger named ragavi
    logger = logging.getLogger("ragavi")
    logger.setLevel(level_num)

    # warnings logger
    w_logger = logging.getLogger("py.warnings")
    w_logger.setLevel(level_num)

    # console handler
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(logfile_name)

    c_handler.setLevel(level_num)
    f_handler.setLevel(level_num)

    # setting the format for the logging messages
    start = " (O_o) ".center(cols, "=")

    c_formatter = logging.Formatter(
        """%(asctime)s - %(name)-12s - %(levelname)-10s - %(message)s""", datefmt="%d.%m.%Y@%H:%M:%S")
    f_formatter = logging.Formatter(
        """%(asctime)s - %(name)-12s - %(levelname)-10s - %(message)s""" , datefmt="%d.%m.%Y@%H:%M:%S")

    c_handler.setFormatter(c_formatter)
    f_handler.setFormatter(f_formatter)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    w_logger.addHandler(f_handler)

    return logger


def __handle_uncaught_exceptions(extype, exval, extraceback):
    """Function to Capture all uncaught exceptions into the log file

       Parameters to this function are acquired from sys.excepthook. This
       is because this function overrides :obj:`sys.excepthook`
      `Sys module excepthook <https://docs.python.org/3/library/sys.html#sys.excepthook>`_

    """
    message = "Oops ... !"
    logger.error(message, exc_info=(extype, exval, extraceback))


########################################################################
########################### Some useful functions ######################

def __welcome():
    """Welcome to ragavi"""
    """
    from pyfiglet import Figlet

    print("\n\n")
    print("_*+_" * 23)
    print("Welcome to ")
    print(Figlet(font="nvscript").renderText("ragavi"))
    print("_*+_" * 23)
    print("\n\n")
    """
    pass


def time_wrapper(func):
    """A decorator function to compute the execution time of a function
    """
    def timer(*args, **kwargs):
        start = time()
        ans = func(*args, **kwargs)
        end = time()
        time_taken = end - start
        logger.info("{} executed in: {:.4} sec.".format(
            func.__name__, time_taken))
        return ans
    return timer


########################################################################
#################### Return colours functions ##########################

"""
Some categories of colormaps and their names

############################ Colorcet ############################
eg. cc.palette[categorical_colours[0]]

categorical_colours = ["glasbey", "glasbey_light", "glasbey_dark",
                        'glasbey_warm', 'glasbey_cool']
diverging_colours = ['bkr', 'bky', 'bwy', 'cwr', 'coolwarm', 'gwv', 'bjy']

misc_colours = ['colorwheel', 'isolum', 'rainbow']

linear_colours = ['bgy', 'bgyw', 'kbc', 'blues', 'bmw', 'bmy', 'kgy', 'gray',
                  'dimgray', 'fire', 'kb', 'kg', 'kr']


"""


def get_cmap(cmap, fall_back="coolwarm", src="bokeh"):
    """Get Hex colors that form a certain cmap. 
    This function checks for the requested cmap in bokeh.palettes 
    and colorcet.palettes.
    List of valid names can be found at:
    https: // colorcet.holoviz.org / user_guide / index.html
    https: // docs.bokeh.org / en / latest / docs / reference / palettes.html

    """
    if src == "bokeh":
        if cmap in pl.all_palettes:
            max_colours = max(pl.all_palettes[cmap].keys())
            colors = pl.all_palettes[cmap][max_colours]
        elif cmap.capitalize() in pl.all_palettes:
            cmap = cmap.capitalize()
            max_colours = max(pl.all_palettes[cmap].keys())
            colors = pl.all_palettes[cmap][max_colours]
        else:
            colors = None
    elif src == "colorcet":
        if cmap in cc.palette:
            colors = cc.palette[cmap]
        else:
            colors = None
    if colors is None:
        if cc.palette[cmap]:
            colors = cc.palette[cmap]
        else:
            colors = cc.palette[fall_back]
            logger.info("Selected colourmap not found. Reverting to default")

    return colors


def get_linear_cmap(cmap, n, fall_back="coolwarm"):
    """Produce n that differ linearly from a given colormap
    This function depends on pl.linear_palettes whose doc be found at:
    https: // docs.bokeh.org / en / latest / docs / reference / palettes.html
    cmap: : obj: `str`
        The colourmap chosen
    n: : obj: `int`
        Number of colours to generate

    Returns
    -------
    colors: : obj: `list`
        A list of size n containing the linear colours
    """
    colors = get_cmap(cmap, fall_back=fall_back)
    if len(colors) < n:
        logger.info("Requested {}, available: {}.".format(n, len(colors)))
        logger.info("Reverting back to default.")
        colors = get_cmap(fall_back)
    colors = pl.linear_palette(colors, n)

    return colors


def get_diverging_cmap(n_colors, cmap1=None, cmap2=None):
    """Produce n_colors that diverge given two different colourmaps.
    This function depends on pl.diverging_palettes whose doc can be found at:
    https: // docs.bokeh.org / en / latest / docs / reference / palettes.html
    cmap1: : obj: `str`
        Name of the first colourmap to use
    cmap2: : obj: `str`
        Name of the second colourmap to use
    n_colors: : obj: `int`
        Number of colours to generate.

    Returns
    -------
    colors: : obj: `list`
        A list of size n_colors containing the diverging colours
    """
    colors1 = get_cmap(cmap1)
    colors2 = get_cmap(cmap2)
    colors = pl.diverging_palette(colors1, colors2, n_colors)

    return colors


########################################################################
#################### define the log ####################################
logger = __config_logger()
sys.excepthook = __handle_uncaught_exceptions
