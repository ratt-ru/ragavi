from __future__ import division

import glob
import numpy as np
import re
import sys

from builtins import map
from collections import OrderedDict, namedtuple
from datetime import datetime
from future.utils import listitems, listvalues

import dask.array as da
import daskms as xm
import matplotlib.cm as cmx
import matplotlib.colors as colors
import xarray as xr

from dask import delayed, compute
from bokeh.io import (export_png, export_svgs, output_file, output_notebook,
                      save, show)
from bokeh.layouts import row, column, gridplot, widgetbox, grid, layout
from bokeh.models import (BasicTicker, CheckboxGroup, ColumnDataSource,
                          CustomJS, HoverTool, Range1d, Legend, LinearAxis,
                          PrintfTickFormatter, Button, Slider, Text, Title,
                          Toggle, Whisker)

from bokeh.models.widgets import Div, PreText
from bokeh.plotting import figure

from ragavi import utils as vu

# defining some constants
# default plot dimensions
PLOT_WIDTH = 900
PLOT_HEIGHT = 750

# gain types supported
GAIN_TYPES = ['B', 'D', 'F', 'G', 'K']

# valueof 1 gigahertz
GHZ = 1e9

# Switch for rendering in notebook
NB_RENDER = None

# Number of antennas to be grouped together
BATCH_SIZE = 16


logger = vu.logger
excepthook = vu.__handle_uncaught_exceptions

#######################################################################
#################### Define some data Processing class ################


class DataCoreProcessor:
    """Process gain table data into forms desirable for visualisation.

    Parameters
    ----------
    antenna : :obj:`str`
    corr : :obj:`int`
        Correlation index to plot
    doplot : :obj:`str`
        Kind of plot to do. Default is ap
    flag : :obj:`bool`
        To flag or not. Default is True
    fid : :obj:`int`
        Field id
    gtype : :obj:`str`
        Gain table type
    ms_name : :obj:`str`
        Name / path to the table
    xds_table_obj : :obj:`xarray.Dataset`
        Table object from which data will be extracted
    """

    def __init__(self, xds_table_obj, ms_name, gtype, fid=None, doplot='ap',
                 corr=0, flag=True, kx=None, ddid=None):

        self.xds_table_obj = xds_table_obj
        self.ms_name = ms_name
        self.gtype = gtype
        self.fid = fid
        self.doplot = doplot
        self.corr = corr
        self.flag = flag
        self.kx = kx
        self.ddid = ddid

    def compute_ydata(self, ydata, yaxis):
        """Abstraction for processing y-data passes it to the processing function.

        Parameters
        ----------
        ydata: :obj:`xarray.DataArray`
               y-data to process
        yaxis: :obj:`str`
               Selected yaxis

        Returns
        -------
        y: :obj:`xarray.DataArray`
           Processed :attr:`ydata`
        """
        if yaxis == 'amplitude':
            y = vu.calc_amplitude(ydata)
        elif yaxis == 'imaginary':
            y = vu.calc_imaginary(ydata)
        elif yaxis == 'phase':
            y = vu.calc_phase(ydata, wrap=False)
        elif yaxis == 'real':
            y = vu.calc_real(ydata)
        elif yaxis == 'delay' or yaxis == 'error':
            y = ydata
        return y

    def flatten_bandpass_data(self, y, x=None):
        """ Massage B table with multiple unique times into a shape
        that works with BokehJS. Returns on x if x input is present
        otherwise returns y.

        Parameters
        ----------
        x : :obj:`np.array`
            1-D Array containing data for the x-axis
        y : :obj:`np.array`
            2-D Array containing y-axis data.

        Returns
        -------
        x : :obj:`np.array`
            1-D Array containing repeated data for the x-axis
        y : :obj:`np.array`
            1-D Array containing y-axis flattened data.


        """
        if y.ndim > 1:
            # no of unique times available in the bandpass
            # last index because y is transposed to (chan, time)
            nrows = y.shape[1]

        if x is not None:
            x = np.tile(x, (nrows))
            return x
        else:
            y = y.flatten(order='F')
            return y

    def get_errors(self, xds_table_obj):
        """Get error data from PARAMERR column.

        Parameters
        ----------
        xds_table_obj : :obj:`xarray.Dataset`
            An xarray Dataset containing all the rows available in a table. The dataset is constructed by :obj:`xarrayms.xds_from_table`

        Returns
        -------
        errors: :obj:`xarray.DataArray`
            Data from the PARAMERR column
        """
        errors = xds_table_obj.PARAMERR
        return errors

    def get_xaxis_data(self, xds_table_obj, ms_name, gtype):
        """Get x-axis data. It is dependent on `gtype`. This function also returns the relevant x-axis labels for both pairs of plots.

        Parameters
        ----------
        gtype: :obj:`str`
            Type of the gain table
        ms_name: :obj:`str`
                 Name of gain table
        xds_table_obj: :obj:`xarray.Dataset`
                       Table as xarray dataset from :obj:`xarrayms`

        Returns
        -------
        (xdata, x_label) : :obj:`xarray.DataArray`, :obj:`str`
                         Tuple containing `xdata` x-axis data depending x-axis selected and a string label to appear on the x-axis of the plots.
        """

        if gtype == 'B' or gtype == 'D':
            xdata = vu.get_frequencies(ms_name, spwid=self.ddid)
            x_label = 'Channel'

        elif gtype == 'F' or gtype == 'G':
            xdata = xds_table_obj.TIME
            x_label = 'Time'
        elif gtype == 'K':
            if self.kx == 'time':
                xdata = xds_table_obj.TIME
                x_label = 'Time'
            else:
                xdata = xds_table_obj.ANTENNA1
                x_label = 'Antenna'
        else:
            logger.error("Invalid xaxis name")
            return
        return xdata, x_label

    def prep_xaxis_data(self, xdata, gtype='G'):
        """Prepare the x-axis data for plotting.

        Parameters
        ----------
        freq: :obj:`xarray.DataArray` or :obj:`float`
            Frequency(ies) from which corresponding wavelength will be obtained. **Only required** when :obj:`xaxis` specified is "uvwave"`
        gtype: :obj:`str`
            Gain table type.
        xdata: :obj:`xarray.DataArray`
            x-axis data depending  xaxis selected.

        Returns
        -------
        prepdx: :obj:`xarray.DataArray`
                Prepared :attr:`xdata`
        """
        if gtype == 'B' or gtype == 'D':
            prepdx = xdata.chan
        elif gtype == 'G' or gtype == 'F':
            prepdx = vu.time_convert(xdata)
        elif gtype == 'K':
            if self.kx == "time":
                prepdx = vu.time_convert(xdata)
            else:
                prepdx = xdata

        return prepdx

    def get_yaxis_data(self, xds_table_obj, ms_name, yaxis):
        """Extract the required column for the y-axis data.

        Parameters
        ----------
        ms_name : :obj:`str`
            Name / path to table.
        xds_table_obj : :obj:`xarray.Dataset`
            Table as xarray dataset from xarrayms
        yaxis : :obj:`str`
            yaxis to plot.

        Returns
        -------
        ydata, y_label : :obj:`xarray.DataArray`, :obj:`str`
                         :attr:`ydata` containing y-axis data depending  y-axis selected and `y_label` which is the label to appear on the y-axis of the plots.
        """

        # default data column
        datacol = 'CPARAM'

        if yaxis == 'amplitude':
            y_label = 'Amplitude'
        elif yaxis == 'imaginary':
            y_label = 'Imaginary'
        elif yaxis == 'phase':
            y_label = 'Phase [deg]'
        elif yaxis == 'real':
            y_label = 'Real'
        elif yaxis == 'delay':
            datacol = 'FPARAM'
            y_label = 'Delay[ns]'

        # attempt to get the specified column from the table
        try:
            ydata = xds_table_obj[datacol]
        except KeyError:
            logger.exception('Column "{}" not Found'.format(datacol))
            return sys.exit(-1)

        return ydata, y_label

    def prep_yaxis_data(self, xds_table_obj, ms_name, ydata, yaxis=None, corr=0, flag=True):
        """Process data for the y-axis.

        This function is responsible for :

            * Correlation selection
            * Flagging
            * Conversion form complex to the required form
            * Data transposition if `yaxis` is frequency or channel

        Data selection and flagging are done by this function itself, however ap and ri conversion are handled by :meth:`ragavi.ragavi.DataCoreProcessor.compute_ydata`

        Parameters
        ----------
        xds_table_obj : :obj:`xarray.Dataset`
            Table as xarray dataset from xarrayms
        ms_name : :obj:`str`
            Name of gain table.
        ydata : :obj:`xarray.DataArray`
            y-axis data to be processed
        yaxis : :obj:`str`
            selected y-axis
        corr : :obj:`int`
            Correlation number to select
        flag : :obj:`bool`
            Option on whether to flag the data or not. Default is True

        Returns
        -------
        y : :obj:`xarray.DataArray`
            Processed :attr:`ydata`
        """
        if corr != None:
            ydata = ydata.sel(corr=corr)
            flags = vu.get_flags(xds_table_obj, corr=corr)
        else:
            ydata = ydata
            flags = vu.get_flags(xds_table_obj)

        # if flagging enabled return a list of DataArrays otherwise return a
        # single dataarray
        if flag:
            # if flagging is activated select data only where flag mask is 0
            processed = self.compute_ydata(ydata, yaxis=yaxis)
            y = processed.where(flags == False)
        else:
            y = self.compute_ydata(ydata, yaxis=yaxis)

        # if B table transpose table to be in shape (chans, solns) rather than
        #(solns, chans)
        if self.gtype == 'B' or self.gtype == 'D':
            y = y.T

        return y

    def blackbox(self, xds_table_obj, ms_name, gtype, fid=None, doplot='ap',
                 corr=0, flag=True):
        """ Get raw input data and churn out processed data. Takes in all inputs from the instance initialising object

        This function incorporates all function in the class to get the desired result. It performs:

            - xaxis data and error data acquisition
            - xaxis data and error preparation and processing
            - yaxis data and error data acquisition
            - yaxis data and error preparation and processing

        Returns
        -------
        d : :obj:`collections.namedtuple`
            A named tuple containing all processed x-axis data, errors and label, as well as both pairs of y-axis data, their error margins and labels. Items from this tuple can be gotten by using the dot notation.

        """

        Data = namedtuple('Data',
                          'x x_label y1 y1_label y1_err y2 y2_label y2_err')

        # this to be ran once
        xdata, xlabel = self.get_xaxis_data(xds_table_obj, ms_name, gtype)
        prepd_x = self.prep_xaxis_data(xdata, gtype=gtype)

        # get errors from the plot
        err_data = self.get_errors(xds_table_obj)
        yerr = self.prep_yaxis_data(xds_table_obj, ms_name, err_data,
                                    yaxis='error', corr=corr, flag=flag)

        ##################################################################
        ##### confirm K table is only plotted in ap mode #################
        ##################################################################

        if gtype == 'K':
            # because only one plot should be generated
            y1data, y1_label = self.get_yaxis_data(xds_table_obj, ms_name,
                                                   'delay')

            # adding the error for preparations
            hi = y1data + yerr
            lo = y1data - yerr

            y1 = self.prep_yaxis_data(xds_table_obj, ms_name, y1data,
                                      yaxis='delay', corr=corr, flag=flag)
            hi_err = self.prep_yaxis_data(xds_table_obj, ms_name, hi,
                                          yaxis='delay', corr=corr, flag=flag)
            lo_err = self.prep_yaxis_data(xds_table_obj, ms_name, lo,
                                          yaxis='delay', corr=corr, flag=flag)

            y2 = 0
            y2_err = 0
            y2_label = 0

            prepd_x, y1, hi_err, lo_err = compute(
                prepd_x.data, y1.data, hi_err.data, lo_err.data)

            # shorting y2 to y1 to avoid problems during plotted
            # y2 does not exist for this table
            d = Data(x=prepd_x, x_label=xlabel, y1=y1,
                     y1_label=y1_label, y1_err=(hi_err, lo_err),
                     y2=y1, y2_label=y1_label,
                     y2_err=(hi_err, lo_err))

            return d

        ##################################################################
        ####### The rest of the tables can be plotted in ap or ri mode ###
        ##################################################################

        if doplot == 'ap':
            y1data, y1_label = self.get_yaxis_data(xds_table_obj, ms_name,
                                                   'amplitude')
            y2data, y2_label = self.get_yaxis_data(xds_table_obj, ms_name,
                                                   'phase')

            hi_y1 = y1data + yerr
            lo_y1 = y1data - yerr
            hi_y2 = y2data + yerr
            lo_y2 = y2data - yerr

            y1 = self.prep_yaxis_data(xds_table_obj, ms_name, y1data,
                                      yaxis='amplitude', corr=corr, flag=flag)
            y2 = self.prep_yaxis_data(xds_table_obj, ms_name, y2data,
                                      yaxis='phase', corr=corr, flag=flag)

            # upper and lower limits for y1
            hi_y1_err = self.prep_yaxis_data(xds_table_obj, ms_name, hi_y1,
                                             yaxis='amplitude', corr=corr,
                                             flag=flag)
            lo_y1_err = self.prep_yaxis_data(xds_table_obj, ms_name, lo_y1,
                                             yaxis='amplitude', corr=corr,
                                             flag=flag)

            # upper and lower limits for y2
            hi_y2_err = self.prep_yaxis_data(xds_table_obj, ms_name, hi_y2,
                                             yaxis='phase', corr=corr,
                                             flag=flag)
            lo_y2_err = self.prep_yaxis_data(xds_table_obj, ms_name, lo_y2,
                                             yaxis='phase', corr=corr,
                                             flag=flag)

        elif doplot == 'ri':
            y1data, y1_label = self.get_yaxis_data(xds_table_obj, ms_name,
                                                   'real')
            y2data, y2_label = self.get_yaxis_data(xds_table_obj, ms_name,
                                                   'imaginary')

            hi_y1 = y1data + yerr
            lo_y1 = y1data - yerr
            hi_y2 = y2data + yerr
            lo_y2 = y2data - yerr

            y1 = self.prep_yaxis_data(xds_table_obj, ms_name, y1data,
                                      yaxis='real', corr=corr, flag=flag)
            y2 = self.prep_yaxis_data(xds_table_obj, ms_name, y2data,
                                      yaxis='imaginary', corr=corr, flag=flag)

            # upper and lower limits for y1
            hi_y1_err = self.prep_yaxis_data(xds_table_obj, ms_name, hi_y1,
                                             yaxis='real', corr=corr,
                                             flag=flag)
            lo_y1_err = self.prep_yaxis_data(xds_table_obj, ms_name, lo_y1,
                                             yaxis='real', corr=corr,
                                             flag=flag)

            # upper and lower limits for y2
            hi_y2_err = self.prep_yaxis_data(xds_table_obj, ms_name, hi_y2,
                                             yaxis='imaginary', corr=corr,
                                             flag=flag)
            lo_y2_err = self.prep_yaxis_data(xds_table_obj, ms_name, lo_y2,
                                             yaxis='imaginary', corr=corr,
                                             flag=flag)

        prepd_x, y1, hi_y1_err, lo_y1_err, y2, hi_y2_err, lo_y2_err =\
            compute(prepd_x.data, y1.data, hi_y1_err.data, lo_y1_err.data,
                    y2.data, hi_y2_err.data, lo_y2_err.data)

        if gtype == 'B' or gtype == 'D':
            prepd_x = self.flatten_bandpass_data(y1, x=prepd_x)
            y1 = self.flatten_bandpass_data(y1)
            hi_y1_err = self.flatten_bandpass_data(hi_y1_err)
            lo_y1_err = self.flatten_bandpass_data(lo_y1_err)
            y2 = self.flatten_bandpass_data(y2)
            hi_y2_err = self.flatten_bandpass_data(hi_y2_err)
            lo_y2_err = self.flatten_bandpass_data(lo_y2_err)

        d = Data(x=prepd_x, x_label=xlabel, y1=y1,
                 y1_label=y1_label, y1_err=(hi_y1_err, lo_y1_err),
                 y2=y2,
                 y2_label=y2_label, y2_err=(hi_y2_err, lo_y2_err))

        return d

    def act(self):
        """Activate the :meth:`ragavi.ragavi.DataCoreProcessor.blackbox`
        """
        return self.blackbox(self.xds_table_obj, self.ms_name, self.gtype,
                             self.fid, self.doplot, self.corr, self.flag)

    def x_only(self):
        """Return only x-axis data and label

        Returns
        -------
        d : :obj:`collections.namedtuple`
            Named tuple containing x-axis data and x-axis label. Items in the tuple can be accessed by using the dot notation.
        """
        Data = namedtuple('Data', 'x x_label')
        xdata, xlabel = self.get_xaxis_data(self.xds_table_obj, self.ms_name,
                                            self.gtype)
        prepd_x = self.prep_xaxis_data(xdata, gtype=self.gtype)
        prepd_x = prepd_x.data

        d = Data(x=prepd_x, x_label=xlabel)
        return d

    def y_only(self, yaxis=None):
        """Return only y-axis data and label

        Returns
        -------
        d : :obj:`collections.namedtuple`
            Named tuple containing x-axis data and x-axis label. Items in the tuple can be accessed by using the dot notation.
        """

        Data = namedtuple('Data', 'y y_label')
        ydata, y_label = self.get_yaxis_data(self.xds_table_obj, self.ms_name,
                                             yaxis)
        y = self.prep_yaxis_data(self.xds_table_obj, self.ms_name, ydata,
                                 yaxis=yaxis, corr=self.corr,
                                 flag=self.flag)
        y = y.data
        d = Data(y=y, y_label=y_label)
        return d


def get_table(tab_name, antenna=None, fid=None, spwid=None, where=None):
    """ Get xarray Dataset objects containing gain table columns of the selected data

    Parameters
    ----------
    antenna: :obj:`str`, optional
        A string containing antennas whose data will be selected
    fid: :obj:`int`, optional
        FIELD_ID whose data will be selected
    spwid: :obj:`int`, optional
        DATA_DESC_ID or spectral window whose data will be selected
    tab_name: :obj:`str`
        name of your table or path including its name
    where: :obj:`str`, optional
        TAQL where clause to be used with the MS.

    Returns
    -------
    tab_objs: :obj:`list`
        A list containing :obj:`xarray.Dataset` objects where each item on the list is determined by how the data is grouped

    """

    # defining part of the gain table schema
    tab_schema = {'CPARAM': {'dims': ('chan', 'corr')},
                  'FLAG': {'dims': ('chan', 'corr')},
                  'FPARAM': {'dims': ('chan', 'corr')},
                  'PARAMERR': {'dims': ('chan', 'corr')},
                  'SNR': {'dims': ('chan', 'corr')},
                  }

    if where == None:
        where = []
    else:
        where = [where]

    if antenna != None:
        where.append("ANTENNA1 IN {}".format(antenna))
    if fid != None:
        where.append("FIELD_ID IN {}".format(fid))
    if spwid != None:
        if spwid.isnumeric():
            spwid = int(spwid)
        else:
            spwid = vu.resolve_ranges(spwid)
        where.append("SPECTRAL_WINDOW_ID IN {}".format(spwid))

    where = "&&".join(where)

    try:
        tab_objs = xm.xds_from_table(tab_name, taql_where=where,
                                     table_schema=tab_schema,
                                     group_cols=None)
        return tab_objs
    except:
        logger.exception("""Invalid ANTENNA id, FIELD_ID,
                            SPECTRAL_WINDOW_ID or TAQL clause""")
        sys.exit(-1)


def save_svg_image(img_name, figa, figb, glax1, glax2):
    """Save plots in SVG format

    Note
    ----
        Two images will emerge. the python package selenium, node package phantomjs are required. More information `Exporting bokeh plots <https://bokeh.pydata.org/en/latest/docs/user_guide/export.html>`_

    Parameters
    ----------
    img_name : :obj:`str`
        Desired image name
    figa : :obj:`bokeh.plotting.figure`
        First figure
    figb: :obj:`bokeh.plotting.figure`
        Second figure
    glax1: :obj:`list`
        Contains glyph metadata saved during execution
    glax2: :obj:`list`
        Contains glyph metadata saved during execution

    """
    for i in range(len(glax1)):
        glax1[i][1][0].visible = True
        glax2[i][1][0].visible = True

    figa.output_backend = "svg"
    figb.output_backend = "svg"
    export_svgs([figa], filename="{:s}_{:s}".format(img_name, "a.svg"))
    export_svgs([figb], filename="{:s}_{:s}".format(img_name, "b.svg"))


def save_png_image(img_name, disp_layout):
    """Save plots in PNG format

    Note
    ----
        One image will emerge for each figure in `disp_layout`. To save png images, the python package selenium, node package phantomjs are required. More information `Exporting bokeh plots <https://bokeh.pydata.org/en/latest/docs/user_guide/export.html>`_

    Parameters
    ----------
    disp_layout : :obj:`bokeh.layouts`
        Layout object containing the renderers
    img_name : :obj:`str`
        Name of output image
    """
    export_png(img_name, disp_layout)


def determine_table(table_name):
    """Find pattern at end of string to determine table to be plotted. The search is not case sensitive

    Parameters
    ----------
    table_name : :obj:`str`
        Name of table /  gain type to be plotted

    Returns
    -------
    result : :obj:`str`
        Table type of (if valid) of the input :attr:`table_name`

    """
    pattern = re.compile(r'\.(G|K|B|F)\d*$', re.I)
    found = pattern.search(table_name)
    try:
        result = found.group()[:2]
        return result.upper()
    except AttributeError:
        return -1


def errorbar(fig, x, y, yerr=None, color='red'):
    """Add errorbars to Figure object based on :attr:`x`, :attr:`y` and attr:`yerr`

    Parameters
    ----------
    color : :obj:`str`
        Color for the error bars
    fig : :obj:`bokeh.plotting.figure`
        Figure onto which the error-bars will be added
    x : :obj:`numpy.ndarray`
        x_axis data
    y : :obj:`numpy.ndarray`
        y_axis data
    yerr : :obj:`numpy.ndarray`, :obj:`numpy.ndarray`
        Tuple with high and low limits for :attr:`y`

    Returns
    -------
    ebars : :obj:`bokeh.models.Whisker`
        Return the object containing error bars
    """
    # Setting default return value
    ebars = None

    if yerr is not None:

        src = ColumnDataSource(data=dict(upper=yerr[0],
                                         lower=yerr[1], base=x))

        ebars = Whisker(source=src, base='base', upper='upper',
                        lower='lower', line_color=color, visible=False)
        ebars.upper_head.line_color = color
        ebars.lower_head.line_color = color
        fig.add_layout(ebars)
    return ebars


def make_plots(source, ax1, ax2, fid=0, color='red', y1err=None, y2err=None):
    """Generate a pair of plots

    Parameters
    ----------
    ax1 : :obj:`bokeh.plotting.figure`
        First figure
    ax2 : :obj:`bokeh.plotting.figure`
        Second Figure
    color : :obj:`str`
        Glyph color[s]
    fid : :obj:`int`
        field id number to set the line width
    source : :obj:`bokeh.models.ColumnDataSource`
            Data source for the plot
    y1err : :obj:`numpy.ndarray`, optional
        y1 Error margins for figure `ax1` data
    y2err: :obj:`numpy.ndarray`, optional
        y2 error margins for figure `ax2` data

    Returns
    -------
    (p1, p1_err, p2, p2_err ): (:obj:`bokeh.models.renderers.GlyphRenderer`, :obj:`bokeh.models.Whisker`, :obj:`bokeh.models.renderers.GlyphRenderer`, :obj:`bokeh.models.Whisker`)
        Tuple of containing bokeh.models.renderers.GlyphRenderer with the data glyphs as well as errorbars.
        `p1` and `p2`: renderers containing data for :obj:`ax1` and :obj:`ax2` respectively. :obj:`p1_err`, :obj:`p2_err` outputs from :func:`ragavi.ragavi.errorbar` for :obj:`ax1` and :obj:`ax2` respectively.

    """
    markers = ['circle', 'diamond', 'square', 'triangle',
               'hex']
    fmarker = 'inverted_triangle'
    glyph_opts = {'size': 4,
                  'fill_alpha': 1,
                  'fill_color': color,
                  'line_color': 'black',
                  'nonselection_fill_color': '#7D7D7D',
                  'nonselection_fill_alpha': 0.3}

    # if there is any flagged data enforce fmarker where flag is active
    fmarkers = None
    if np.any(np.isnan(source.data['y1'])):
        fmarkers = gen_flag_data_markers(source.data['y1'], fid=fid,
                                         markers=markers, fmarker=fmarker)
        # update the data source with markers
        source.add(fmarkers, name='fmarkers')

        p1 = ax1.scatter(x='x', y='y1', marker='fmarkers', source=source,
                         line_width=0, angle=0.7, **glyph_opts)

        p2 = ax2.scatter(x='x', y='y2', marker='fmarkers', source=source,
                         line_width=0, angle=0.7, **glyph_opts)
    else:
        p1 = ax1.scatter(x='x', y='y1', marker=markers[fid], source=source,
                         line_width=0, **glyph_opts)

        p2 = ax2.scatter(x='x', y='y2', marker=markers[fid], source=source,
                         line_width=0, **glyph_opts)

    # add a check for whether all the in y data were NaNs
    # this causes the errorbars to fail if they all are
    # they must be checked
    if np.all(np.isnan(source.data['y1'])):
        p1_err = errorbar(fig=ax1, x=source.data['x'], y=source.data['y1'],
                          color=color, yerr=None)
        p2_err = errorbar(fig=ax2, x=source.data['y2'], y=source.data['y2'],
                          color=color, yerr=None)
    else:
        p1_err = errorbar(fig=ax1, x=source.data['x'], y=source.data['y1'],
                          color=color, yerr=y1err)
        p2_err = errorbar(fig=ax2, x=source.data['x'], y=source.data['y2'],
                          color=color, yerr=y2err)

    # link the visible properties of the two plots
    p1.js_link('visible', p2, 'visible')
    p2.js_link('visible', p1, 'visible')

    if p1_err:
        p1.glyph.js_link('fill_alpha', p1_err, 'line_alpha')
        p2.glyph.js_link('fill_alpha', p2_err, 'line_alpha')

    return p1, p1_err, p2, p2_err


def ant_select_callback():
    """JS callback for the selection and de-selection of antennas

    Returns
    -------
    code : :obj:`str`
    """

    code = """
            /*p1 and p2: both lists containing the plot models
              bsel: batch selection group button
              nbatches: Number of batches available
              fsel: field_selector widget,
              csel: corr_select widget,
              ssel: spw_select widget,
              nfields: Number of available fields,
              ncorrs: Number of avaiblable corrs,
              nspws: Number of spectral windows,
            */
            let nplots = p1.length;
            if (cb_obj.active.includes(0))
                {
                    for(let i=0; i<nplots; i++){
                        p1[i].visible = true;
                        p2[i].visible = true;
                    }
                    //activate all the checkboxes whose antennas are active
                    bsel.active = [...Array(nbatches).keys()]
                    csel.active = [...Array(ncorrs).keys()]
                    fsel.active = [...Array(nfields).keys()]
                    ssel.active = [...Array(nspws).keys()]

                }
            else{
                    for(let i=0; i<nplots; i++){
                       p1[i].visible = false;
                       p2[i].visible = false;
                    }

                    bsel.active = []
                    csel.active = []
                    fsel.active = []
                    ssel.active = []
                }
            """

    return code


def toggle_err_callback():
    """JS callback for Error toggle Toggle button

    Returns
    -------
    code : :obj:`str`
    """
    code = """
            let i;
             //if toggle button active
            if (cb_obj.active.includes(0))
                {
                    for(i=0; i<err1.length; i++){
                        //only switch on if corresponding plot is on
                        if(ax1s[i].visible){
                            //check if the error bars are present first
                            if (err1[i]){
                                err1[i].visible = true;
                                err2[i].visible = true;
                            }

                        }

                    }
                }
            else{
                    for(i=0; i<err1.length; i++){
                        if (err1[i]){
                            err1[i].visible = false;
                            err2[i].visible = false;
                        }
                    }
                }
            """
    return code


def batch_select_callback():
    """JS callback for batch selection Checkboxes

    Returns
    -------
    code : :obj:`str`
    """
    code = """
        /* antsel: Select or deselect all antennas button
           bsize: total number of items in a batch
           csel: the corr selector array
           fsel: the field selector button
           ssel: spw selector button
           nants: Number of antennas
           ncorrs: number of available correlations
           nfields: number of available fields
           nbatches: total number of available batches
           p1 and p2: List containing plots for all antennas, fields and correlations
           count: keeping a cumulative sum of the traverse number
        */

        let count = 0;
        let new_bsize;
        for (let sp=0; sp<nspws; sp++){
            for (let c=0; c<ncorrs; c++){
                for (let f=0; f<nfields; f++){
                    //re-initialise bsize
                    new_bsize = bsize;
                    for(let n=0; n<nbatches; n++){
                        /* Reduce batch size to the size of the last batch
                        and number of antennas is not the same as bsize
                        */
                        if (n == nbatches-1 && nants!=bsize){
                            new_bsize = nants % bsize;
                        }
                        for(let b=0; b<new_bsize; b++){
                            if (cb_obj.active.includes(n)){
                                p1[count].visible = true;
                                p2[count].visible = true;
                                }
                            else{
                                p1[count].visible = false;
                                p2[count].visible = false;
                            }
                            count = count + 1;
                        }
                    }
                }
            }
        }

        if (cb_obj.active.length == nbatches){
            antsel.active = [0];
        }
        else if (cb_obj.active.length==0){
            antsel.active = [];
        }
       """
    return code


def legend_toggle_callback():
    """JS callback for legend toggle Dropdown menu

    Returns
    -------
    code : :obj:`str`
    """
    code = """
                let len = loax1.length;
                if (cb_obj.active.includes(0)){
                    for(let i=0; i<len; i++){
                        loax1[i].visible = true;
                    }
                }

                else{
                    for(let i=0; i<len; i++){
                        loax1[i].visible = false;
                    }
                }

           """
    return code


def size_slider_callback():
    """JS callback to select size of glyphs

    Returns
    -------
    code : :obj:`str`
    """

    code = """

            let pos, i, numplots;

            numplots = p1.length;
            pos = slide.value;

            for (i=0; i<numplots; i++){
                p1[i].glyph.size = pos;
                p2[i].glyph.size = pos;
            }
           """
    return code


def alpha_slider_callback():
    """JS callback to alter alpha of glyphs

    Returns
    -------
    code : :obj:`str`
    """

    code = """

            let pos, i, numplots;
            numplots = p1.length;
            pos = alpha.value;


            for (i=0; i<numplots; i++){
                p1[i].glyph.fill_alpha = pos;
                p2[i].glyph.fill_alpha = pos;
            }
           """
    return code


def field_selector_callback():
    """Return JS callback for field selection checkboxes

    Returns
    -------
    code : :obj:`str`
    """
    code = """
        /*bsize: total number of items in a batch
         bsel: the batch selector group buttons
         csel: corr selector group buttons
         nants: Number of antennas
         ncorrs: number of available correlations
         nfields: number of available fields
         nbatches: total number of available batches
         p1 and p2: List containing plots for all antennas, fields and  correlations
         count: keeping a cumulative sum of the traverse number
        */
        let count = 0;
        let new_bsize;
        for (let sp=0; sp<nspws; sp++){
            for (let c=0; c<ncorrs; c++){
                for (let f=0; f<nfields; f++){
                    //re-initialise new batch size
                    new_bsize = bsize;
                    for(let n=0; n<nbatches; n++){
                        // Reduce new batch size to the size of the last batch
                        if (n == nbatches-1 && nants!=bsize){
                            new_bsize = nants % bsize;
                        }
                        for(let b=0; b<new_bsize; b++){
                            if (cb_obj.active.includes(f) && csel.active.includes(c) &&
                                ssel.active.includes(sp)){
                                p1[count].visible = true;
                                p2[count].visible = true;
                                }
                            else{
                                p1[count].visible = false;
                                p2[count].visible = false;
                            }
                            count = count + 1;
                        }
                    }
                }
            }
        }
       """
    return code


def axis_fs_callback():
    """JS callback to alter axis label font sizes

    Returns
    -------
    code : :obj:`str`
    """
    code = """
            let naxes = ax1.length;
            let i;
            for(i=0; i<naxes; i++){
                ax1[i].axis_label_text_font_size = `${cb_obj.value}pt`;
                ax2[i].axis_label_text_font_size = `${cb_obj.value}pt`;
            }
           """
    return code


def title_fs_callback():
    """JS callback for title font size slider

    Returns
    -------
    code : :obj:`str`
    """
    code = """
            let len = ax1.length;
            for (let i=0; i<len; i++){
                if (ax1[i].name == "ftitle"){
                    ax1[i].text_font_size = `${cb_obj.value}px`;
                    ax2[i].text_font_size = `${cb_obj.value}px`;
                }
            }

           """
    return code


def flag_callback():
    """JS callback for the flagging button

    Returns
    -------
    code : :obj:`str`
    """
    code = """
            //sources: list of all different sources for the different antennas and available sources
            //n_sources: number of sources
            //flagging: status of flag_data

            let n_sources =  sources.length;

            let init_src = Array();
            let src_1 = Array();

            for(item in sources){init_src[item] = sources[item][0];}
            for(item in sources){src_1[item] = sources[item][1];}

            if (cb_obj.active.includes(0)){
                for (i=0; i<n_sources; i++){
                    init_src[i].data.y1 = src_1[i].data.iy1;
                    init_src[i].data.y2 = src_1[i].data.iy2;
                    init_src[i].change.emit();
                }
            }
            else{
                for (i=0; i<n_sources; i++){
                    init_src[i].data.y1 = src_1[i].data.y1;
                    init_src[i].data.y2 = src_1[i].data.y2;
                    init_src[i].change.emit();
                }


            }
           """
    return code


def corr_select_callback():
    code = """
         /*bsize: total number of items in a batch
           bsel: the batch selector group buttons
           fsel: field selector group buttons
           ssel: spw selector group buttons
           nants: Total number of antennas
           nspws: Number of spectral windows
           ncorrs: number of available correlations
           nfields: number of available fields
           nbatches: total number of available batches
           p1 and p2: List containing plots for all antennas, fields and correlations
           count: keeping a cumulative sum of the traverse number
        */

        let count = 0;
        let new_bsize;
        for (let sp=0; sp<nspws; sp++){
            for (let c=0; c<ncorrs; c++){
                for (let f=0; f<nfields; f++){
                    //re-initialise new batch size
                    new_bsize = bsize;
                    for(let n=0; n<nbatches; n++){
                        // Reduce new_bsize to the size of the final batch
                        if (n == nbatches-1 && nants!=bsize){
                            new_bsize = nants % bsize;
                        }
                        for(let b=0; b<new_bsize; b++){
                            if (cb_obj.active.includes(c) && fsel.active.includes(f) &&
                                ssel.active.includes(sp)){
                                p1[count].visible = true;
                                p2[count].visible = true;
                                }
                            else{
                                p1[count].visible = false;
                                p2[count].visible = false;
                            }
                            count = count + 1;
                        }
                    }
                }
            }
        }

       """
    return code


def spw_select_callback():
    code = """
         /*bsize: total number of items in a batch
           bsel: the batch selector group buttons
           extra_axes1 and 2: for the extra frequency ax(es). Is 0 if none.
           fsel: field selector group buttons
           csel: corr selector group buttons
           nspws: Number of spectral windows
           ncorrs: number of available correlations
           nfields: number of available fields
           nbatches: total number of available batches
           p1 and p2: List containing plots for all antennas, fields and correlations
           count: keeping a cumulative sum of the traverse number
        */

        let count = 0;
        let new_bsize;
        for (let sp=0; sp<nspws; sp++){
            for (let c=0; c<ncorrs; c++){
                for (let f=0; f<nfields; f++){
                    //re-initialise new batch size
                    new_bsize = bsize;

                    for(let n=0; n<nbatches; n++){
                        // Reduce bsize to the size of the last batch
                        if (n == nbatches-1 && nants!=bsize){
                            new_bsize = nants % bsize;
                        }
                        for(let b=0; b<new_bsize; b++){
                            if (cb_obj.active.includes(sp) && fsel.active.includes(f) &&
                                csel.active.includes(c)){
                                p1[count].visible = true;
                                p2[count].visible = true;
                                }
                            else{
                                p1[count].visible = false;
                                p2[count].visible = false;
                            }
                            count = count + 1;
                        }
                    }
                }
            }

            //Make the extra y-axes visible only if corresponding spw selected
            if (cb_obj.active.includes(sp)){
                extra_axes1[sp].visible=true;
                extra_axes2[sp].visible=true;
                }
            else{
                extra_axes1[sp].visible=false;
                extra_axes2[sp].visible=false;

            }

        }

       """
    return code


def save_selected_callback():
    code = """
        /*uf_src: Unflagged data source
          f_src:  Flagged data source scanid antname
        */
        let out = `x, y1, y2, ant, corr, field, scan, spw\n`;

        //for all the data sources available
        for (let i=0; i<uf_src.length; i++){
            let sel_idx = uf_src[i].selected.indices;
            let data = uf_src[i].data;

            for (let j=0; j<sel_idx.length; j++){
                out +=  `${data['x'][sel_idx[j]]}, ` +
                        `${data['y1'][sel_idx[j]]}, ` +
                        `${data['y2'][sel_idx[j]]}, ` +
                        `${data['antname'][sel_idx[j]]}, ` +
                        `${data['corr'][sel_idx[j]]}, ` +
                        `${data['field'][sel_idx[j]]}, ` +
                        `${data['scanid'][sel_idx[j]]}, ` +
                        `${data['spw'][sel_idx[j]]}\n`;
            }

        }
        let answer = confirm("Download selected data?");
        if (answer){
            let file = new Blob([out], {type: 'text/csv'});
            let element = window.document.createElement('a');
            element.href = window.URL.createObjectURL(file);
            element.download = "data_selection.csv";
            document.body.appendChild(element);
            element.click();
            document.body.removeChild(element);
        }

    """

    return code


def create_legend_batches(num_leg_objs, li_ax1, batch_size=16):
    """Automates creation of antenna **batches of 16** each unless otherwise.

    This function takes in a long list containing all the generated legend items from the main function's iteration and divides this list into batches, each of size :attr:`batch_size`. The outputs provides the inputs to
    :meth:`ragavi.ragavi.create_legend_objs`.


    Parameters
    ----------
    batch_size : :obj:`int`, optional
        Number of antennas in a legend object. Default is 16
    li_ax1 : :obj:`list`
        # bokeh.models.annotations.LegendItem>`_ for antennas for 1st figure
        # items are in the form (antenna_legend, [renderer])
        List containing all `legend items <https://bokeh.pydata.org/en/latest/docs/reference/models/annotations.html
    num_leg_objs : :obj:`int`
        Number of legend objects to be created

    Returns
    -------
    bax1 : :obj:`list`
           Tuple containing List of lists which each have :attr:`batch_size` number of legend items for each batch.
           bax1 are batches for figure1 antenna legends, and ax2 batches for figure2 antenna legends

           e.g bax1 = [[batch0], [batch1], ...,  [batch_numOfBatches]]
    """

    bax1 = []

    # condense the returned list if two fields were plotted
    li_ax1 = condense_legend_items(li_ax1)

    j = 0
    for i in range(num_leg_objs):
        # in case the number is not a multiple of 16 or is <= num_leg_objs
        # or on the last iteration

        if i == num_leg_objs:
            bax1.extend([li_ax1[j:]])
        else:
            bax1.extend([li_ax1[j:j + batch_size]])

        j += batch_size

    return bax1


def create_legend_objs(num_leg_objs, bax1):
    """Creates legend objects using items from batches list
    Legend objects allow legends be positioning outside the main plot

    Parameters
    ----------
    num_leg_objs: :obj:`int`
        Number of legend objects to be created
    bax1 : :obj:`list`
        Batches for antenna legends of 1st figure

    Returns
    -------
    lo_ax1 : :obj:`dict`
        Dictionaries with legend objects for figure1 antenna legend objects


    """

    lo_ax1 = {}

    l_opts = dict(click_policy='hide',
                  orientation='horizontal',
                  label_text_font_size='9pt',
                  visible=False,
                  padding=2,
                  margin=1,
                  location='top_left',
                  glyph_width=10)

    for i in range(num_leg_objs):
        leg1 = Legend(items=bax1[i], **l_opts)
        lo_ax1['leg_%s' % str(i)] = leg1

    return lo_ax1


def gen_checkbox_labels(batch_size, num_leg_objs, antnames):
    """ Auto-generating Check box labels

    Parameters
    ----------
    batch_size : :obj:`int`
        Number of items in a single batch
    num_leg_objs : :obj:`int`
        Number of legend objects / Number of batches
    Returns
    ------
    labels : :obj:`list`
        Batch labels for the batch selection check box group
    """
    nants = len(antnames)

    labels = []
    s = 0
    e = batch_size - 1
    for i in range(num_leg_objs):
        if e < nants:
            labels.append("{} - {}".format(antnames[s], antnames[e]))
        else:
            labels.append("{} - {}".format(antnames[s], antnames[nants - 1]))
        # after each append, move start number to current+batchsize
        s = s + batch_size
        e = e + batch_size

    return labels


def save_html(hname, plot_layout):
    """Save plots in HTML format

    Parameters
    ----------
    hname : :obj:`str`
        HTML Output file name
    plot_layout : :obj:`bokeh.layouts`
                  Layout of the Bokeh plot, could be row, column, gridplot.
    """
    hname = hname + ".html"
    output_file(hname)
    output = save(plot_layout, hname, title=hname)
    # uncomment next line to automatically plot on web browser
    # show(layout)


def add_axis(fig, axis_range, ax_label, ax_name):
    """Add an extra axis to the current figure

    Parameters
    ----------
    fig : :obj:`bokeh.plotting.figure`
        The figure onto which to add extra axis

    axis_range : :obj:`float`, :obj:`float`
        Starting and ending point for the range
    ax_label : :obj:`str`
        Label of the new axis
    ax_name : :obj:`str`
        Name of the new model for the extra axis incase of multiple spectral windows.

    Returns
    ------
    fig : :obj:`bokeh.plotting.figure`
        Figure containing the extra axis
    """
    fig.extra_x_ranges[ax_name] = Range1d(
        start=axis_range[0], end=axis_range[-1])
    linaxis = LinearAxis(x_range_name=ax_name, axis_label=ax_label,
                         major_label_orientation='horizontal',
                         ticker=BasicTicker(desired_num_ticks=12),
                         axis_label_text_font_style='normal', name=ax_name)
    fig.add_layout(linaxis, 'above')
    return fig


def get_tooltip_data(xds_table_obj, gtype, antnames, freqs):
    """Get the data to be displayed on the tool-tip of the plots

    Parameters
    ----------
    antnames : :obj:`list`
        List containing antenna names
    gtype: :obj:`str`
        Type of gain table being plotted
    xds_table_obj : `xarray.Dataset`
        xarray-ms table object

    Returns
    -------
    spw_id : :obj:`numpy.ndarray`
        Spectral window ids
    scan_no : :obj:`numpy.ndarray`
        scan ids
    ttip_antnames : :obj:`numpy.ndarray`
        Antenna names to show up on the tool-tips
    """
    spw_id = xds_table_obj.SPECTRAL_WINDOW_ID.data.astype(np.int)
    scan_no = xds_table_obj.SCAN_NUMBER.data.astype(np.int)
    ant_id = xds_table_obj.ANTENNA1.data.astype(np.int)

    spw_id, scan_no, ant_id = compute(spw_id, scan_no, ant_id)

    # get available antenna names from antenna id
    ttip_antnames = np.array([antnames[x] for x in ant_id])

    # get the number of channels
    nchan = freqs.size

    if gtype == 'B' or gtype == 'D':
        # extend array to accommodate for multiple time slots
        spw_id = spw_id.repeat(nchan, axis=0)
        # tile because of different scan numbers available
        scan_no = np.tile(scan_no, nchan)
        ant_id = ant_id.repeat(nchan, axis=0)
        ttip_antnames = antnames[ant_id]

    return spw_id, scan_no, ttip_antnames


def stats_display(tab_name, gtype, ptype, corr, field, f_names=None, flag=True, spwid=None):
    """Display some statistics on the plots.
    These statistics are derived from a specific correlation and a specified field of the data.

    Note
    ----
        Currently, only the medians of these plots are displayed.

    Parameters
    ----------
    corr : :obj:`int`
        Correlation number of the data to be displayed
    field : :obj:`int`
        Integer field id of the field being plotted. If a string name was provided, it will be converted within the main function by :meth:`ragavi.vis_utils.name_2id`.
    gtype : :obj:`str`
        Type of gain table to be plotted.
    ptype : :obj:`str`
        Type of plot ap / ri

    Returns
    -------
    pre: :obj:`bokeh.models.widgets`
        Pre-formatted text containing the medians for both model. The object returned must then be placed within the widget box for display.
    """
    subtable = get_table(tab_name, fid=field, spwid=spwid)[0]

    dobj = DataCoreProcessor(subtable, tab_name, gtype,
                             corr=corr, flag=True, ddid=spwid)

    if gtype == 'K':
        y1 = dobj.y_only('delay').y.compute()
        med_y1 = np.nanmedian(y1)
        med_y2 = ' '

        return [spwid, f_names[field], corr, med_y1, med_y2]

    if ptype == 'ap':
        y1 = dobj.y_only('amplitude').y.compute()
        y2 = dobj.y_only('phase').y.compute()
    else:
        y1 = dobj.y_only('real').y.compute()
        y2 = dobj.y_only('imaginary').y.compute()

    med_y1 = np.nanmedian(y1)
    med_y2 = np.nanmedian(y2)

    return [spwid, f_names[field], corr, med_y1, med_y2]


def autofill_gains(t, g):
    """Normalise length of :attr:`f` and :attr:`g` lists to the length of :attr:`t`. This function is meant to support the ability to specify multiple gain tables while only specifying single values for gain table types.

    Note
    ----
        An assumption will be made that for all the specified tables, the same gain table type will be used.

    Parameters
    ----------
    g : :obj:`list`
        type of gain table [B,G,K,F].
    t : :obj:`list`
        list of the gain tables.


    Returns
    -------
    f : :obj:`list`
        lists of length ..code::`len(t)` containing gain types.
    """
    ltab = len(t)
    lgains = len(g)

    if ltab != lgains and lgains == 1:
        g = g * ltab
    return g


def condense_legend_items(inlist):
    """Combine renderers of legend items with the same legend labels. Must be done in case there are groups of renderers which have the same label due to iterations, to avoid a case where there are two or more groups of renderers containing the same label name.

    Parameters
    ----------
    inlist : :obj:`list`
             # bokeh.models.annotations.LegendItem>`_
             List containing legend items of the form (label, renders) as described in: `Bokeh legend items <https://bokeh.pydata.org/en/latest/docs/reference/models/annotations.html
    Returns
    -------
    outlist : :obj:`list`
              A reduction of :attr:`inlist`
    """
    od = OrderedDict()
    # put all renderers with the same legend labels together
    for key, value in inlist:
        if key in od:
            od[key].extend(value)
        else:
            od[key] = value

    # reformulate odict to list
    outlist = [(key, value) for key, value in od.items()]
    return outlist


def gen_flag_data_markers(y, fid=None, markers=None, fmarker='circle_x'):
    """Generate different markers for where data has been flagged.

    Parameters
    ----------
    fid : :obj:`int`
        field id number to identify the marker to be used
    fmarker : :obj:`str`
        Marker to be used for flagged data
    markers : :obj:`list`
        A list of all available bokeh markers
    y : :obj:`numpy.ndarray`
        The flagged data containing NaNs

    Returns
    -------
    masked_markers_arr : :obj:`numpy.ndarray`
        Returns an n-d array of shape :code:`y.shape` containing markers for valid data and :attr:`fmarker` where the data was NaN.
    """

    # fill an array with the unflagged marker value
    markers_arr = np.full(y.shape, fill_value=markers[fid], dtype='<U17')

    # mask only where there are nan values
    masked_markers_arr = np.ma.masked_where(np.isnan(y), markers_arr)
    # fill with the different marker
    masked_markers_arr.fill_value = fmarker

    # return filled matrix
    masked_markers_arr = masked_markers_arr.filled()

    return masked_markers_arr


def get_time_range(tab_name, unix_time=True):
    """Get the first TIME column before selections"""

    from pyrap.tables import table

    # unix_time = MJD - munix
    munix = 3506716800.0

    ms = table(tab_name, ack=False)
    i_time = ms.getcell('TIME', 0)
    f_time = ms.getcell('TIME', (ms.nrows() - 1))

    if unix_time:
        i_time = i_time - munix
        f_time = f_time - munix

    ms.close()

    return i_time, f_time


def make_table_name(tab_name):
    div = Div(text="Table: {}".format(tab_name))
    return div


def create_stats_table(stats, ptype):
    from bokeh.models.widgets import DataTable, TableColumn

    if ptype == 'ap':
        ys = "amplitude phase".split()
    else:
        ys = "real imaginary".split()

    stats = np.array(stats)
    d_stats = dict(spw=stats[:, 0],
                   field=stats[:, 1],
                   corr=stats[:, 2],
                   )
    d_stats[ys[0]] = stats[:, 3]
    d_stats[ys[1]] = stats[:, 4]
    source = ColumnDataSource(data=d_stats)
    cols = "spw field corr".split() + ys
    columns = [TableColumn(field=x, title=x.capitalize()) for x in cols]
    dtab = DataTable(source=source, columns=columns,
                     width=400, max_width=450,
                     height=100, max_height=150, sizing_mode='stretch_both')
    t_title = Div(text="Median Statistics")

    return column([t_title, dtab], sizing_mode='stretch_both')


def main(**kwargs):
    """Main function that launches the gains plotter"""
    if 'options' in kwargs:
        NB_RENDER = False
        # capture the parser options
        options = kwargs.get('options', None)

        _corr = options.corr
        ddid = options.ddid
        doplot = options.doplot
        fields = options.fields
        gain_types = options.gain_types
        html_name = options.html_name
        image_name = options.image_name
        mycmap = options.mycmap
        mytabs = options.mytabs
        plotants = options.plotants
        t0 = options.t0
        t1 = options.t1
        where = options.where
        kx = options.kx

    else:
        NB_RENDER = True

        _corr = kwargs.get('corr', None)
        ddid = kwargs.get('ddid', None)
        doplot = kwargs.get('doplot', 'ap')
        fields = kwargs.get('fields', None)
        gain_types = kwargs.get('gain_types', [])
        image_name = str(kwargs.get('image_name', ''))
        mycmap = str(kwargs.get('mycmap', 'coolwarm'))
        mytabs = kwargs.get('mytabs', [])
        plotants = kwargs.get('plotants', None)
        t0 = float(kwargs.get('t0', -1))
        t1 = float(kwargs.get('t1', -1))
        where = kwargs.get('where', None)
        kx = kwargs.get('kx', "time")

    # make into a string if list
    if isinstance(fields, list):
        fields = ",".join(fields)

    # To flag or not
    flag_data = True

    if len(mytabs) == 0:
        logger.error('Exiting: No gain table specified.')
        sys.exit(-1)

    mytabs = [x.rstrip("/") for x in mytabs]

    gain_types = [x.upper() for x in gain_types]

    if doplot not in ['ap', 'ri']:
        logger.error('Exiting: Plot selection must be ap or ri.')
        sys.exit(-1)

    # for notebook
    for gain_type in gain_types:
        if gain_type not in GAIN_TYPES:
            logger.error("Exiting: gtype {} invalid".format(gain_type))
            sys.exit(-1)

    gain_types = autofill_gains(mytabs, gain_types)
    # array to store final output image
    final_layout = []

    for mytab, gain_type in zip(mytabs, gain_types):

        # re-initialise plotant list for each table
        if NB_RENDER:
            plotants = kwargs.get('plotants', None)
        else:
            plotants = options.plotants

        if fields is not None:
            # Not using .isalnum coz the actual field names can be provided
            if '~' in fields or ':' in fields or fields.isnumeric():
                fields = vu.resolve_ranges(fields)
            elif ',' in fields:
                """
                 check if all are digits in fields, if not convert field
                 name to field id and join all the resulting field ids with a
                 comma
                """
                fields = ",".join(
                    [str(vu.name_2id(mytab, x)) if not x.isnumeric() else x for x in fields.split(',')])
                fields = vu.resolve_ranges(fields)
            else:
                fields = str(vu.name_2id(mytab, fields))
                fields = vu.resolve_ranges(fields)

        # select desired antennas as MS is opening
        if plotants is not None:
            plotants = vu.resolve_ranges(plotants)

        # Get the very initial and final time available in the table
        init_time, end_time = get_time_range(mytab, unix_time=False)

        logger.info('Acquiring table: {}'.format(mytab.split('/')[-1]))
        tt = get_table(mytab, spwid=ddid, where=where, fid=fields,
                       antenna=plotants)[0]

        # confirm a populous table is selected
        try:
            assert tt.FLAG.size != 0
        except AssertionError:
            logger.info(
                """Table contains no data. Check selected Antenna, Field, Scan or Spectral window. Skipping.""")
            continue

        # constrain the plots to a certain time period if specified
        if t0 != None or t1 != None:
            # selection of specified times
            time_s = tt.TIME - init_time
            if t0 != None:
                tt = tt.where((time_s >= t0), drop=True)
                # time_s = time_s.where(time_s >= t0, drop=True)
            if t1 != None:
                tt = tt.where((time_s <= t1), drop=True)
                # time_s = time_s.where(time_s <= t1, drop=True)

        antnames = vu.get_antennas(mytab).values

        # get string names of available fields
        fnames = vu.get_fields(mytab).data.compute()
        fsyms = [u'\u2B24', u'\u25C6', u'\u25FC', u'\u25B2',
                 u'\u25BC', u'\u2B22']

        # get all the ant ids present in a subtable
        plotants = np.unique(tt.ANTENNA1.values).astype(int)

        # get all unique field ids in the data
        ufids = np.unique(tt.FIELD_ID.values).astype(int)

        if _corr is not None:
            # because of iteration _corr and corrs must'nt be the same
            if ',' in _corr:
                _corr = [int(_) for _ in _corr.split(',')]
            else:
                _corr = [int(_corr)]
            corrs = np.array(_corr)
        else:
            corrs = tt.FLAG.corr.values

        # freqs = (vu.get_frequencies(
        #     mytab, spwid=vu.slice_data(ddid)) / GHZ).values

        # setting up colors for the antenna plots
        cNorm = colors.Normalize(vmin=0, vmax=plotants.size - 1)
        mymap = cm = cmx.get_cmap(mycmap)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=mymap)

        # creating bokeh figures for plots
        # linking plots ax1 and ax2 via the x_axes because of similarities in
        # range
        TOOLS = dict(tools="""box_select, box_zoom, reset, pan, save,
                            wheel_zoom, lasso_select""",
                     # sizing_mode="stretch_both",
                     min_border_right=20)

        if gain_type == 'G' or gain_type == 'F' or (kx == "time" and gain_type == 'K'):
            xaxis_type = "datetime"
        else:
            xaxis_type = "linear"

        logger.info('Setting up plotting canvas')

        ax1 = figure(x_axis_type=xaxis_type, **TOOLS)
        ax2 = figure(x_axis_type=xaxis_type, x_range=ax1.x_range, **TOOLS)

        # initialise plot containers
        ax1_plots = []
        ax2_plots = []

        # forming Legend object items for data and errors
        legend_items_ax1 = []

        ebars_ax1 = []
        ebars_ax2 = []

        # Statistical info holder
        stats_ax = []

        # for storing flagged and unflagged data sources
        sources = []

        uddids = np.unique(tt.SPECTRAL_WINDOW_ID.values)
        uddids = uddids.astype(int)

        for win in uddids:
            logger.info("Spectral Window {}".format(win))
            freqs = (vu.get_frequencies(mytab, spwid=win) / GHZ).values
            for corr in corrs:
                # check if selected corr is avail
                if corr not in tt.FLAG.corr.values:
                    logger.info("Corr {} not found. Skipping.".format(corr))
                    continue
                # enumerating available field ids incase of large fids
                for enum_fid, field in enumerate(ufids):
                    stats = stats_display(mytab, gain_type, doplot, corr,
                                          field, flag=flag_data,
                                          f_names=fnames, spwid=str(win))
                    stats_ax.append(stats)

                    logger.info(
                        """Plotting spw:{} field: {} corr: {}""".format(win, field, corr))
                    # for each antenna
                    for ant in plotants:

                        # creating legend labels
                        antlabel = antnames[ant]
                        legend = antnames[ant]
                        legend_err = "E" + antnames[ant]

                        # creating colors for maps
                        y1col = y2col = scalarMap.to_rgba(
                            float(ant), bytes=True)[:-1]

                        # filter for antenna and field
                        subtab = tt.where((tt.ANTENNA1 == int(ant)) &
                                          (tt.FIELD_ID == int(field)) &
                                          (tt.SPECTRAL_WINDOW_ID == int(win)),
                                          drop=True)

                        # depending on the status of flag_data, this may
                        # be either flagged or unflagged data
                        data_obj = DataCoreProcessor(subtab, mytab, gain_type,
                                                     fid=field,
                                                     doplot=doplot, corr=corr,
                                                     flag=flag_data, kx=kx,
                                                     ddid=win)
                        ready_data = data_obj.act()

                        prepd_x = ready_data.x
                        xlabel = ready_data.x_label

                        y1 = ready_data.y1
                        y1_err = ready_data.y1_err
                        y1label = ready_data.y1_label
                        y2 = ready_data.y2
                        y2_err = ready_data.y2_err
                        y2label = ready_data.y2_label

                        # inverse data object
                        infl_data_obj = DataCoreProcessor(
                            subtab, mytab, gain_type, fid=field,
                            doplot=doplot, corr=corr, flag=not flag_data,
                            kx=kx, ddid=win).act()

                        # for tooltips
                        spw_id, scan_no, ttip_antnames = get_tooltip_data(
                            subtab, gain_type, antnames, freqs)

                        tab_tooltips = [
                            ("({:.4}, {:.4})".format(xlabel, y1label),
                             "(@x, @y1)"),
                            ("spw", "@spw"),
                            ("scan_id", "@scanid"),
                            ("antenna", "@antname"),
                            ("corr", "@corr"),
                            ("field", "@field")]
                        tab_tooltips2 = tab_tooltips.copy()

                        # change tooltip info for the second plot
                        tab_tooltips2[0] = ("({:.4}, {:.4})".format(xlabel,
                                                                    y2label),
                                            "(@x, @y2)")

                        hover = HoverTool(tooltips=tab_tooltips, mode='mouse',
                                          point_policy='snap_to_data')

                        hover2 = HoverTool(tooltips=tab_tooltips2,
                                           mode='mouse',
                                           point_policy='snap_to_data')

                        ax1.xaxis.axis_label = ax1_xlabel = xlabel
                        ax2.xaxis.axis_label = ax2_xlabel = xlabel
                        ax1.yaxis.axis_label = ax1_ylabel = y1label
                        ax2.yaxis.axis_label = ax2_ylabel = y2label

                        ax1.axis.axis_label_text_font_style = 'normal'
                        ax2.axis.axis_label_text_font_style = 'normal'

                        if doplot == 'ap':
                            # Add degree sign to phase tick formatter
                            ax2.yaxis[0].formatter = PrintfTickFormatter(
                                format=u"%f\u00b0")

                        if gain_type == 'B' or gain_type == 'D':
                            # on the very last iterations
                            if corr == corrs.max() and ant == plotants.max() and field == ufids.max():
                                if freqs.ndim == 2:
                                    freqs = freqs[0, :]

                                ax1 = add_axis(ax1, [freqs[0], freqs[-1]],
                                               ax_label='Frequency [GHz]',
                                               ax_name="spw{}".format(win))
                                ax2 = add_axis(ax2, [freqs[0], freqs[-1]],
                                               ax_label='Frequency [GHz]',
                                               ax_name="spw{}".format(win))

                        if gain_type == 'K':
                            ax1_ylabel = y1label.replace('[ns]', '')
                            ax2_ylabel = y2label.replace('[ns]', '')

                        if xlabel == 'Time':
                            munix = 3506716800.0
                            ni_time = datetime.utcfromtimestamp(
                                init_time - munix).strftime("%d %b %Y, %H:%M:%S")
                            ne_time = datetime.utcfromtimestamp(
                                end_time - munix).strftime("%d %b %Y, %H:%M:%S")

                            ax1.xaxis.axis_label = xlabel + \
                                " from {} UTC".format(ni_time)
                            ax2.xaxis.axis_label = ax1.axis[0].axis_label

                            # format tootltip to time value
                            hover.formatters = dict(x="datetime")
                            hover2.formatters = dict(x="datetime")

                            tab_tooltips[0] = ("({:.4}, {:.4})".format(
                                xlabel, y1label), "(@x{%F %T}, @y1)")

                            tab_tooltips2[0] = ("({:.4}, {:.4})".format(
                                xlabel, y2label), "(@x{%F %T}, @y2)")

                        source = ColumnDataSource(
                            data={'x': prepd_x,
                                  'y1': y1,
                                  'y2': y2,
                                  'corr': [corr] * prepd_x.size,
                                  'field': [fnames[field]] * prepd_x.size,
                                  'spw': spw_id,
                                  'scanid': scan_no,
                                  'antname': ttip_antnames})

                        inv_source = ColumnDataSource(
                            data={'y1': y1,
                                  'y2': y2,
                                  'iy1': infl_data_obj.y1,
                                  'iy2': infl_data_obj.y2})

                        sources.append([source, inv_source])

                        p1, p1_err, p2, p2_err = make_plots(
                            source=source, color=y1col, ax1=ax1, ax2=ax2,
                            fid=enum_fid, y1err=y1_err, y2err=y2_err)

                        if (ant < BATCH_SIZE and win == uddids.min() and
                                corr == corrs.min() and enum_fid == 0):
                            # select the first spw,corr and field
                            p1.visible = p2.visible = True
                        else:
                            p1.visible = p2.visible = False

                        # collecting plot states for each iterations
                        ax1_plots.append(p1)
                        ax2_plots.append(p2)

                        # forming legend object items
                        legend_items_ax1.append((legend, [p1]))

                        # for the errors
                        ebars_ax1.append(p1_err)
                        ebars_ax2.append(p2_err)

                        subtab.close()
        tt.close()

        # configuring titles for the plots
        ax1_title = Title(text="{} vs {}".format(ax1_ylabel, ax1_xlabel),
                          align='center', text_font_size='15px',
                          name="ftitle")
        ax2_title = Title(text="{} vs {}".format(ax2_ylabel, ax2_xlabel),
                          align='center', text_font_size='15px',
                          name="ftitle")

        ax1.add_tools(hover)
        ax2.add_tools(hover2)

        # LEGEND CONFIGURATIONS
        # determining the number of legend objects required to be created
        # for each plot
        num_legend_objs = int(np.ceil(len(plotants) / BATCH_SIZE))

        batches_ax1 = create_legend_batches(num_legend_objs,
                                            legend_items_ax1,
                                            batch_size=BATCH_SIZE)

        legend_objs_ax1 = create_legend_objs(num_legend_objs,
                                             batches_ax1)

        # adding plot titles
        ax2.add_layout(ax2_title, 'above')
        ax1.add_layout(ax1_title, 'above')

        # adding legend objects to the layouts
        # add them in reverse so that the first batch is the topmost
        for i in reversed(range(num_legend_objs)):
            ax1.add_layout(legend_objs_ax1['leg_%s' % str(i)], 'above')

        stats_table = create_stats_table(stats_ax, doplot)

        ######################################################################
        ################ Defining widgets ###################################
        ######################################################################

        # widget dimensions
        w_dims = dict(width=150, height=30)

        # creating and configuring Antenna selection buttons
        ant_select = CheckboxGroup(labels=['Select all antennas'],
                                   active=[], **w_dims)

        # configuring toggle button for showing all the errors
        toggle_err = CheckboxGroup(labels=['Show error bars'], active=[],
                                   **w_dims)

        ant_labs = gen_checkbox_labels(BATCH_SIZE, num_legend_objs, antnames)

        batch_select = CheckboxGroup(labels=ant_labs, active=[0],
                                     **w_dims)

        # Dropdown to hide and show legends
        legend_toggle = CheckboxGroup(labels=['Show legends'], active=[],
                                      **w_dims)

        # creating glyph size slider for the plots
        # margin = [top, right, bottom, left]
        size_slider = Slider(end=15, start=0.4, step=0.1,
                             value=4, title='Glyph size', margin=(3, 5, 7, 5),
                             bar_color='#6F95C3', ** w_dims)

        # Alpha slider for the glyphs
        alpha_slider = Slider(end=1, start=0.1, step=0.1, value=1,
                              margin=(3, 5, 7, 5), title='Glyph alpha',
                              bar_color='#6F95C3', **w_dims)

        try:
            field_labels = ["Field {} {}".format(
                fnames[int(x)],
                fsyms[enum_fid]) for enum_fid, x in enumerate(ufids)]
        except UnicodeEncodeError:
            field_labels = ["Field {} {}".format(
                fnames[int(x)],
                fsyms[enum_fid].encode('utf-8')) for enum_fid,
                x in enumerate(ufids)]

        field_selector = CheckboxGroup(labels=field_labels,
                                       active=[0],
                                       **w_dims)
        axis_fontslider = Slider(end=20, start=3, step=0.5, value=10,
                                 margin=(7, 5, 3, 5), title='Axis label size',
                                 bar_color='#6F95C3', **w_dims)
        title_fontslider = Slider(end=35, start=10, step=1, value=15,
                                  margin=(3, 5, 7, 5), title='Title size',
                                  bar_color='#6F95C3', **w_dims)

        toggle_flag = CheckboxGroup(labels=['Show flagged data'],
                                    active=[], **w_dims)

        corr_labs = ["Correlation {}".format(str(_)) for _ in corrs]
        corr_select = CheckboxGroup(labels=corr_labs, active=[0],
                                    width=150)

        tname_div = make_table_name(mytab)

        spw_labs = ["Spw: {}".format(str(_)) for _ in uddids]
        spw_select = CheckboxGroup(labels=spw_labs,
                                   active=[0],
                                   width=150)
        save_selected = Button(label="Download data selection",
                               button_type="success", margin=(7, 5, 3, 5),
                               **w_dims)
        ######################################################################
        ############## Defining widget Callbacks ############################
        ######################################################################

        ant_select.js_on_change("active", CustomJS(
            args=dict(p1=ax1_plots, p2=ax2_plots, bsel=batch_select,
                      fsel=field_selector, csel=corr_select, ssel=spw_select,
                      nbatches=num_legend_objs, nfields=ufids.size,
                      ncorrs=corrs.size, nspws=uddids.size),
            code=ant_select_callback()))

        toggle_err.js_on_change("active", CustomJS(args=dict(ax1s=ax1_plots,
                                                             err1=ebars_ax1,
                                                             err2=ebars_ax2),
                                                   code=toggle_err_callback()))

        # BATCH SELECTION
        batch_select.js_on_change("active", CustomJS(
            args=dict(p1=ax1_plots, p2=ax2_plots, bsize=BATCH_SIZE,
                      nants=plotants.size, nbatches=num_legend_objs,
                      nfields=ufids.size, ncorrs=corrs.size,
                      nspws=uddids.size, fsel=field_selector,
                      csel=corr_select, antsel=ant_select, ssel=spw_select),
            code=batch_select_callback()))

        legend_toggle.js_on_change("active", CustomJS(
            args=dict(loax1=listvalues(legend_objs_ax1)),
            code=legend_toggle_callback()))

        size_slider.js_on_change('value',
                                 CustomJS(args=dict(slide=size_slider,
                                                    p1=ax1_plots,
                                                    p2=ax2_plots),
                                          code=size_slider_callback()))
        alpha_slider.js_on_change('value',
                                  CustomJS(args=dict(alpha=alpha_slider,
                                                     p1=ax1_plots,
                                                     p2=ax2_plots),
                                           code=alpha_slider_callback()))
        field_selector.js_on_change("active", CustomJS(
            args=dict(bsize=BATCH_SIZE, bsel=batch_select, csel=corr_select,
                      nants=plotants.size, nfields=ufids.size,
                      ncorrs=corrs.size, nbatches=num_legend_objs,
                      nspws=uddids.size, p1=ax1_plots,
                      p2=ax2_plots, ssel=spw_select),
            code=field_selector_callback()))
        axis_fontslider.js_on_change('value',
                                     CustomJS(args=dict(ax1=ax1.axis,
                                                        ax2=ax2.axis),
                                              code=axis_fs_callback()))
        title_fontslider.js_on_change('value',
                                      CustomJS(args=dict(ax1=ax1.above,
                                                         ax2=ax2.above),
                                               code=title_fs_callback()))

        toggle_flag.js_on_change("active", CustomJS(args=dict(sources=sources,
                                                              nants=plotants,
                                                              flagging=flag_data),
                                                    code=flag_callback()))
        corr_select.js_on_change("active", CustomJS(
            args=dict(bsel=batch_select, bsize=BATCH_SIZE,
                      fsel=field_selector, nants=plotants.size,
                      ncorrs=corrs.size, nfields=ufids.size,
                      nbatches=num_legend_objs, nspws=uddids.size,
                      p1=ax1_plots, p2=ax2_plots, ssel=spw_select),
            code=corr_select_callback()))

        # for frequency axis with  spws
        # make the additional axes accesible to spw selector for toggling
        if len(ax1.extra_x_ranges) > 0:
            extra_axes1 = ax1.select(layout='above', type=LinearAxis)
            extra_axes2 = ax2.select(layout='above', type=LinearAxis)
            # sort them
            extra_axes1 = sorted({_.id: _ for _ in extra_axes1}.items())
            extra_axes2 = sorted({_.id: _ for _ in extra_axes2}.items())
            extra_axes1 = [j for i, j in extra_axes1]
            extra_axes2 = [j for i, j in extra_axes2]
            for a in range(len(extra_axes1)):
                # force axis labels. They change for some reason
                extra_axes1[a].axis_label = "Frequency [GHz]"
                extra_axes2[a].axis_label = "Frequency [GHz]"
        else:
            extra_axes1 = 0
            extra_axes2 = 0

        spw_select.js_on_change("active", CustomJS(
            args=dict(bsel=batch_select, bsize=BATCH_SIZE, csel=corr_select,
                      fsel=field_selector, nants=plotants.size,
                      ncorrs=corrs.size, nfields=ufids.size,
                      nbatches=num_legend_objs, nspws=uddids.size,
                      p1=ax1_plots, p2=ax2_plots, extra_axes1=extra_axes1,
                      extra_axes2=extra_axes2),
            code=spw_select_callback()))

        save_selected.js_on_click(CustomJS(args=dict(
            uf_src=[x[0] for x in sources],
            f_src=[x[1] for x in sources]),
            code=save_selected_callback()))

        #################################################################
        ########## Define widget layouts #################################
        ##################################################################
        asel_div = Div(text="Select antenna group")
        fsel_div = Div(text="Fields")
        ssel_div = Div(text="Select spw")
        csel_div = Div(text="Select correlation")

        w_box1 = widgetbox([ant_select, legend_toggle, asel_div,
                            batch_select])
        w_box2 = widgetbox([toggle_err, toggle_flag, fsel_div,
                            field_selector])
        w_box3 = widgetbox([size_slider, alpha_slider, ssel_div,
                            spw_select])
        w_box4 = widgetbox([title_fontslider, axis_fontslider, csel_div,
                            corr_select])

        all_widgets = row([w_box1, w_box2, w_box3, w_box4,
                           widgetbox(save_selected, stats_table)],
                          sizing_mode='stretch_both', spacing=10)

        if gain_type == 'K':
            plots = gridplot([[ax1]], toolbar_location='above',
                             sizing_mode='stretch_both')
        else:
            plots = gridplot([[ax1, ax2]], plot_height=PLOT_HEIGHT,
                             toolbar_location='above',
                             sizing_mode='stretch_both')

        lay = layout([[tname_div], [all_widgets], [plots]],
                     sizing_mode='stretch_width')
        final_layout.append(lay)

        logger.info("Table {} done.".format(mytab))

    if image_name:
        save_svg_image(image_name, ax1, ax2,
                       legend_items_ax1, legend_items_ax2)

    if not NB_RENDER:
        if html_name:
            save_html(html_name, final_layout)

        else:
            # Remove path (if any) from table name
            if '/' in mytab:
                mytab = mytab.split('/')[-1]

            # if more than one table, give time based name
            if len(mytabs) > 1:
                mytab = datetime.now().strftime('%Y%m%d_%H%M%S')
            html_name = "{}_corr_{}_{}_field_{}".format(
                mytab, corr, doplot,
                "".join([str(x) for x in ufids.tolist()]))
            save_html(html_name, final_layout)

        logger.info("Rendered: {}.html".format(html_name))

    else:
        output_notebook()
        for lays in final_layout:
            show(lays)


def plot_table(mytabs, gain_types, **kwargs):
    """Plot gain tables within Jupyter notebooks.

    Parameters
    ----------
    _corr : :obj:`int, optional`
        Correlation index to plot. Can be a single integer or comma separated integers e.g '0,2'. Defaults to all.
    doplot : :obj:`str, optional`
        Plot complex values as amp and phase (ap) or real and imag (ri). Default is 'ap'.
    ddid : :obj:`int`
        SPECTRAL_WINDOW_ID or ddid number. Defaults to all
    fields : :obj:`str, optional`
        Field ID(s) / NAME(s) to plot. Can be specified as "0", "0,2,4", "0~3" (inclusive range), "0:3" (exclusive range), "3:" (from 3 to last) or using a field name or comma separated field names. Defaults to all.
    gain_types : :obj:`str`, :obj:`list`
        Cal-table (list of caltypes) type to be plotted. Can be either 'B'-bandpass, 'D'- D jones leakages, G'-gains, 'K'-delay or 'F'-flux. Default is none
    image_name : :obj:`str, optional`
        Output image name. Default is of the form: table_corr_doplot_fields
    mycmap : `str, optional`
        Matplotlib colour map to use for antennas. Default is coolwarm
    mytabs : :obj:`str` or :obj:`list` required
        The table (list of tables) to be plotted.
    plotants : :obj:`str, optional`
        Plot only this antenna, or comma-separated string of antennas. Default is all
    taql: :obj:`str, optional`
        TAQL where clause
    t0  : :obj:`int, optional`
        Minimum time [in seconds] to plot. Default is full range
    t1 : :obj:`int, optional`
        Maximum time [in seconds] to plot. Default is full range
    """
    if mytabs == None:
        print("Please specify a gain table to plot.")
        logger.error('Exiting: No gain table specfied.')
        sys.exit(-1)
    else:
        # getting the name of the gain table specified
        if isinstance(mytabs, str):
            kwargs['mytabs'] = [str(mytabs)]
            kwargs['gain_types'] = [str(gain_types)]
        else:
            kwargs['mytabs'] = [x.rstrip("/") for x in mytabs]
            kwargs['gain_types'] = [str(x) for x in gain_types]

    main(**kwargs)

    return
