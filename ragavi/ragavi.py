from __future__ import division

import argparse
import logging
import glob
import numpy as np
import re
import sys
import warnings

import dask.array as da
import matplotlib.cm as cmx
import matplotlib.colors as colors
import xarray as xa
import xarrayms as xm


from argparse import ArgumentParser
from builtins import map
from collections import OrderedDict, namedtuple
from dask import delayed, compute
from datetime import datetime
from future.utils import listitems, listvalues

from bokeh.io import (export_png, export_svgs, output_file, output_notebook,
                      save, show)
from bokeh.layouts import row, column, gridplot, widgetbox, grid
from bokeh.models import (BasicTicker, CheckboxGroup, ColumnDataSource,
                          CustomJS, HoverTool, Range1d, Legend, LinearAxis,
                          PrintfTickFormatter, Select, Slider, Text, Title,
                          Toggle)
from bokeh.models.markers import (Circle, CircleCross, Diamond, Hex,
                                  InvertedTriangle, Square, SquareCross,
                                  Triangle)
from bokeh.models.widgets import Div, PreText
from bokeh.plotting import figure

from . import vis_utils as vu

# defining some constants
# default plot dimensions
PLOT_WIDTH = 900
PLOT_HEIGHT = 700

# gain types supported
GAIN_TYPES = ['B', 'F', 'G', 'K']

# valueof 1 gigahertz
GHZ = 1e9

# Switch for rendering in notebook
NB_RENDER = None

# Number of antennas to be grouped together
BATCH_SIZE = 16


logger = vu.config_logger()
sys.excepthook = vu._handle_uncaught_exceptions

#######################################################################
#################### Define some data Processing class ################


class DataCoreProcessor:

    def __init__(self, xds_table_obj, ms_name, gtype, fid=None, antenna=None,
                 doplot='ap', corr=0, flag=True):

        self.xds_table_obj = xds_table_obj
        self.ms_name = ms_name
        self.gtype = gtype
        self.fid = fid
        self.antenna = antenna
        self.doplot = doplot
        self.corr = corr
        self.flag = flag

    def get_phase(self, ydata, wrap=True):
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
        phase = xa.apply_ufunc(da.angle, ydata,
                               dask='allowed', kwargs=dict(deg=True))
        if wrap:
            # delay dispatching of wrapped phase
            phase = xa.apply_ufunc(np.unwrap, phase, dask='allowed')
        return phase

    def get_amplitude(self, ydata):
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

    def get_real(self, ydata):
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

    def get_imaginary(self, ydata):
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

    def compute_ydata(self, ydata, yaxis):
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
            y = self.get_amplitude(ydata)
        elif yaxis == 'imaginary':
            y = self.get_imaginary(ydata)
        elif yaxis == 'phase':
            y = self.get_phase(ydata, wrap=True)
        elif yaxis == 'real':
            y = self.get_real(ydata)
        elif yaxis == 'delay' or yaxis == 'error':
            y = ydata
        return y

    def get_errors(self, xds_table_obj):
        """Function to get error data from PARAMERR column.
        Inputs
        ------
        xdstable_obj: xarrayms table object.

        Outputs
        errors: xarray data array
                Error data.
        """
        errors = xds_table_obj.PARAMERR
        return errors

    def get_xaxis_data(self, xds_table_obj, ms_name, gtype):
        """Function to get x-axis data. It is dependent on the gaintype.
            This function also returns the relevant x-axis labels for both pairs of plots.
        Inputs
        ------
        xds_table_obj: xarray Dataset
                       Table as xarray dataset from xarrayms
        ms_name: str
                 Name of gain table
        gtype: str
               Type of the gain table
        Outputs
        -------
        Tuple (xdadta, x_label)

        xdata: xarray DataArray
               X-axis data depending  x-axis selected.
        x_label: str
                     Label to appear on the x-axis of the plots.
        """

        if gtype == 'B':
            xdata = vu.get_frequencies(ms_name)
            x_label = 'Channel'
        elif gtype == 'F' or gtype == 'G':
            xdata = xds_table_obj.TIME
            x_label = 'Time bin'
        elif gtype == 'K':
            xdata = xds_table_obj.ANTENNA1
            x_label = 'Antenna1'
        else:
            logger.error("Invalid xaxis name")
            return

        return xdata, x_label

    def prep_xaxis_data(self, xdata, gtype='G'):
        """Prepare the x-axis data for plotting.
        Inputs
        ------
        xdata: xarray DataArray
               X-axis data depending  x-axis selected.
        gtype: str
               Gain table type.
        freq: xarray DataArray or float
              Frequency(ies) from which corresponding wavelength will be obtained.
              REQUIRED ONLY when xaxis specified is 'uvwave'.
        Outputs
        -------
        prepdx: xarray DataArray
                Prepared data for the x-axis.
        """
        if gtype == 'B':
            prepdx = xdata.chan
        elif gtype == 'G' or gtype == 'F':
            prepdx = xdata - xdata[0]
            # prepdx = vu.time_convert(xdata)
        elif gtype == 'K':
            prepdx = xdata
        return prepdx

    def get_yaxis_data(self, xds_table_obj, ms_name, yaxis):
        """Extract the required column for the y-axis data.
        Inputs
        -----
        xds_table_obj: xarray Dataset
                       MS as xarray dataset from xarrayms
        ms_name: str
                 Name of table.
        yaxis: str
               yaxis to plot.

        Outputs
        -------
        ydata: xarray DataArray
               y-axis data depending  y-axis selected.
        y_label: str
                 Label to appear on the y-axis of the plots.
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

    def prep_yaxis_data(self, xds_table_obj, ms_name, ydata, yaxis=None, corr=0, flag=False):
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
                 Name of gain table.
        ydata: xarray DataArray
               y-axis data to be processed
        yaxis: str
               selected y-axis
        corr: int
              Correlation number to select
        flag: bool
              Option on whether to flag the data or not
        Outputs
        -------
        y: xarray DataArray
           Processed yaxis data.
        """
        if corr != None:
            ydata = ydata.sel(corr=corr)
            flags = vu.get_flags(xds_table_obj).sel(corr=corr)

        # if flagging enabled return a list of DataArrays otherwise return a
        # single dataarray
        if flag:
            processed = self.compute_ydata(ydata, yaxis=yaxis)
            y = processed.where(flags < 1)
        else:
            y = self.compute_ydata(ydata, yaxis=yaxis)

        # if B table transpose table to be in shape (chans, solns) rather than
        #(solns, chans)
        if self.gtype == 'B':
            y = y.T

        return y

    def blackbox(self, xds_table_obj, ms_name, gtype, fid=None, antenna=None,
                 doplot='ap', corr=0, flag=True):

        Data = namedtuple('Data',
                          'x x_label y1 y1_label y1_err y2 y2_label y2_err')

        # this o be ran once
        xdata, xlabel = self.get_xaxis_data(xds_table_obj, ms_name, gtype)
        prepd_x = self.prep_xaxis_data(xdata, gtype=gtype)

        yerr = self.get_errors(xds_table_obj)

        ##################################################################
        ##### confirm K table is only plotted in ap mode #################
        ##################################################################

        if gtype == 'K':
            # because only one plot should be generated
            y1data, y1_label = self.get_yaxis_data(xds_table_obj, ms_name,
                                                   'delay')
            y1 = self.prep_yaxis_data(xds_table_obj, ms_name, y1data,
                                      yaxis='delay', corr=corr, flag=flag)[:, 0]
            y1_err = self.prep_yaxis_data(xds_table_obj, ms_name, yerr,
                                          yaxis='error', corr=corr, flag=flag)
            y2 = 0
            y2_err = 0
            y2_label = 0

            prepd_x, y1, y1_err = compute(prepd_x.data, y1.data, y1_err.data)

            # shorting y2 to y1 to avoid problems during plotted
            # y2 does not exist for this table
            d = Data(x=prepd_x, x_label=xlabel, y1=y1, y1_label=y1_label,
                     y1_err=y1_err, y2=y1, y2_label=y1_label, y2_err=y1_err)

            return d

        ##################################################################
        ####### The rest of the tables can be plotted in ap or ri mode ###
        ##################################################################

        if doplot == 'ap':
            y1data, y1_label = self.get_yaxis_data(xds_table_obj, ms_name,
                                                   'amplitude')
            y2data, y2_label = self.get_yaxis_data(
                xds_table_obj, ms_name, 'phase')
            y1 = self.prep_yaxis_data(xds_table_obj, ms_name, y1data,
                                      yaxis='amplitude', corr=corr, flag=flag)
            y2 = self.prep_yaxis_data(xds_table_obj, ms_name, y2data,
                                      yaxis='phase', corr=corr, flag=flag)
            y1_err = self.prep_yaxis_data(xds_table_obj, ms_name, yerr,
                                          yaxis='error', corr=corr, flag=flag)
            y2_err = self.prep_yaxis_data(xds_table_obj, ms_name, yerr,
                                          yaxis='error', corr=corr, flag=flag)
        elif doplot == 'ri':
            y1data, y1_label = self.get_yaxis_data(
                xds_table_obj, ms_name, 'real')
            y2data, y2_label = self.get_yaxis_data(xds_table_obj, ms_name,
                                                   'imaginary')
            y1 = self.prep_yaxis_data(xds_table_obj, ms_name, y1data,
                                      yaxis='real', corr=corr, flag=flag)[:, 0]
            y2 = self.prep_yaxis_data(xds_table_obj, ms_name, y2data,
                                      yaxis='imaginary', corr=corr, flag=flag)[:, 0]
            y1_err = self.prep_yaxis_data(xds_table_obj, ms_name, yerr,
                                          yaxis='error', corr=corr, flag=flag)
            y2_err = self.prep_yaxis_data(xds_table_obj, ms_name, yerr,
                                          yaxis='error', corr=corr, flag=flag)

        prepd_x, y1, y1_err, y2, y2_err = compute(prepd_x.data, y1.data,
                                                  y1_err.data, y2.data,
                                                  y2_err.data)

        d = Data(x=prepd_x, x_label=xlabel, y1=y1, y1_label=y1_label,
                 y1_err=y1_err, y2=y2, y2_label=y2_label, y2_err=y2_err)

        return d

    def act(self):
        return self.blackbox(self.xds_table_obj, self.ms_name, self.gtype,
                             self.fid, self.antenna, self.doplot, self.corr,
                             self.flag)

    def x_only(self):
        Data = namedtuple('Data', 'x x_label')
        xdata, xlabel = self.get_xaxis_data(
            self.xds_table_obj, self.ms_name,                                self.gtype)
        prepd_x = self.prep_xaxis_data(self.xdata, gtype=self.gtype)
        prepd_x = prepd_x.data.compute()

        d = Data(x=prepd_x, x_label=xlabel)
        return d

    def y_only(self, yaxis=None):

        Data = namedtuple('Data', 'y y_label')
        ydata, y_label = self.get_yaxis_data(self.xds_table_obj, self.ms_name,
                                             yaxis)
        y = self.prep_yaxis_data(self.xds_table_obj, self.ms_name, ydata,
                                 yaxis=yaxis, corr=self.corr,
                                 flag=self.flag)[:, 0]
        y = y.data.compute()
        d = Data(y=y, y_label=y_label)
        return d


def get_table(tab_name, antenna=None, fid=None, where=None):

    # defining part of the gain table schema
    tab_schema = {'CPARAM': ('chan', 'corr'),
                  'FLAG': ('chan', 'corr'),
                  'FPARAM': ('chan', 'corr'),
                  'PARAMERR': ('chan', 'corr'),
                  'SNR': ('chan', 'corr'),
                  }

    if where == None:
        where = []
        if antenna != None:
            where.append("ANTENNA1=={}".format(antenna))
        if fid != None:
            where.append("FIELD_ID=={}".format(fid))

        where = "&&".join(where)

    try:
        tab_objs = xm.xds_from_table(tab_name, taql_where=where,
                                     table_schema=tab_schema)
        return tab_objs
    except:
        logging.exception("Invalid ANTENNA id, FIELD_ID or TAQL clause")
        sys.exit(-1)


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


def determine_table(table_name):
    """Find pattern at end of string to determine table to be plotted.
       The search is not case sensitive

    Input
    -----
    table_name: str
                Name of table /  gain type to be plotted

    """
    pattern = re.compile(r'\.(G|K|B|F)\d*$', re.I)
    found = pattern.search(table_name)
    try:
        result = found.group()[:2]
        return result.upper()
    except AttributeError:
        return -1


def errorbar(fig, x, y, xerr=None, yerr=None, color='red', point_kwargs={},
             error_kwargs={}):
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

    return h


def make_plots(source, ax1, ax2, fid=0, color='red', y1_err=None,
               y2_err=None):
    """Generate a plot

    Inputs
    ------

    source: ColumnDataSource
        Main data to plot
    ax1: figure
        First figure
    ax2: figure
        Second Figure
    fid: int
         field id number to set the line width
    color: str
        Data points' color
    y1_err: numpy.ndarray
        y1 error data
    y2_err: numpy.ndarray
        y2 error data

    Outputs
    -------
    (p1, p1_err, p2, p2_err ): tuple
        Tuple of glyphs

    """
    markers = [Circle, Diamond, Square, Triangle, InvertedTriangle, Hex]
    glyph_opts = {'size': 4,
                  'fill_alpha': 1,
                  'fill_color': color,
                  'line_width': 0,
                  'line_color': 'black'}

    nonsel_glyph = markers[fid]()
    nonsel_glyph.update(fill_color='#7D7D7D',
                        fill_alpha=0.3,)

    # create an instance of the glyph
    glyph_ax1 = markers[fid]()
    glyph_ax1.update(x='x', y='y1', **glyph_opts)

    p1 = ax1.add_glyph(source, glyph=glyph_ax1,
                       nonselection_glyph=nonsel_glyph)
    p1_err = errorbar(fig=ax1, x=source.data['x'], y=source.data['y1'],
                      color=color, yerr=y1_err)

    glyph_ax2 = markers[fid]()
    glyph_ax2.update(x='x', y='y2', **glyph_opts)
    p2 = ax2.add_glyph(source, glyph=glyph_ax2,
                       nonselection_glyph=nonsel_glyph)
    p2_err = errorbar(fig=ax2, x=source.data['x'], y=source.data['y2'],
                      color=color, yerr=y2_err)

    return p1, p1_err, p2, p2_err


def ant_select_callback():
    """JS callback for the selection and deselection of antennas
        Returns : string
    """

    code = """
            var i;
             //if toggle button active
            if (this.active==false)
                {
                    this.label='Select all Antennas';


                    for(i=0; i<glyph1.length; i++){
                        glyph1[i][1][0].visible = false;
                        glyph2[i][1][0].visible = false;
                    }

                    batchsel.active = []
                }
            else{
                    this.label='Deselect all Antennas';
                    for(i=0; i<glyph1.length; i++){
                        glyph1[i][1][0].visible = true;
                        glyph2[i][1][0].visible = true;

                    }

                    batchsel.active = [0,1,2,3]
                }
            """

    return code


def toggle_err_callback():
    """JS callback for Error toggle  Toggle button
        Returns : string
    """
    code = """
            var i;
             //if toggle button active
            if (this.active==false)
                {
                    this.label='Show All Error bars';


                    for(i=0; i<err1.length; i++){
                        err1[i][1][0].visible = false;
                        //checking for error on phase and imaginary planes as these tend to go off
                        if (err2[i][1][0]){
                            err2[i][1][0].visible = false;
                        }



                    }
                }
            else{
                    this.label='Hide All Error bars';
                    for(i=0; i<err1.length; i++){
                        err1[i][1][0].visible = true;
                        if (err2[i][1][0]){
                            err2[i][1][0].visible = true;
                        }
                    }
                }
            """
    return code


def batch_select_callback():
    """JS callback for batch selection Checkboxes
        Returns : string
    """
    code = """
            // bax[i][k][l][m]

            // k: item number in batch
            // i: batch number
            // l: 1 legend item number, must be 1 coz of legend specs
            // m: item number 0 which is the glyph
            // nfields: number of fields to be plotted by the script. We shall get this from the number of glyph renderers attached to the same legend label
            // f: field number represented

            let num_of_batches = bax1.length;

            //sampling a single item for the length of fields
            let nfields = bax1[0][0][1].length;




            //for each batch in total number of batches
            for(i=0; i<num_of_batches; i++){
                //check whether batch number is included in the active list
                if (this.active.includes(i)){
                    k=0;
                    while (k < batch_size){
                        //show all items in the active batch

                        for (f=0; f<nfields; f++){
                            bax1[i][k][1][f].visible = true;
                            bax2[i][k][1][f].visible = true;
                        }
                        k++;

                    }
                }

                else{
                    k=0;
                    while (k < batch_size){
                        for (f=0; f<nfields; f++){
                            bax1[i][k][1][f].visible = false;
                            bax2[i][k][1][f].visible = false;
                        }
                        k++;
                    }
                }
            }

            if (this.active.length == num_of_batches){
                antsel.active = true;
                antsel.label =  "Deselect all Antennas";
            }
            else if(this.active.length == 0){
                antsel.active = false;
                antsel.label = "Select all Antennas";
            }
           """
    return code


def legend_toggle_callback():
    """JS callback for legend toggle Dropdown menu
        Returns : string
    """
    code = """
                var len = loax1.length;
                var i ;
                if (this.value == "alo"){
                    for(i=0; i<len; i++){
                        loax1[i].visible = true;
                        loax2[i].visible = true;

                    }
                }

                else{
                    for(i=0; i<len; i++){
                        loax1[i].visible = false;
                        loax2[i].visible = false;

                    }
                }



                if (this.value == "elo"){
                    for(i=0; i<len; i++){
                        loax1_err[i].visible = true;
                        loax2_err[i].visible = true;

                    }
                }

                else{
                    for(i=0; i<len; i++){
                        loax1_err[i].visible = false;
                        loax2_err[i].visible = false;

                    }
                }

                if (this.value == "non"){
                    for(i=0; i<len; i++){
                        loax1[i].visible = false;
                        loax2[i].visible = false;
                        loax1_err[i].visible = false;
                        loax2_err[i].visible = false;

                    }
                }
           """
    return code


def size_slider_callback():
    """JS callback to select size of glyphs
    """

    code = """

            var pos, i, numplots;

            numplots = p1.length;
            pos = slide.value;

            for (i=0; i<numplots; i++){
                p1[i].glyph.size = pos;
                p2[i].glyph.size = pos;
            }
           """
    return code


def alpha_slider_callback():
    """JS callback to select alpha of glyphs"""

    code = """

            var pos, i, numplots;

            //debugger;
            numplots = p1.length;
            pos = alpha.value;


            for (i=0; i<numplots; i++){
                p1[i].glyph.fill_alpha = pos;
                p2[i].glyph.fill_alpha = pos;
            }
           """
    return code


def field_selector_callback():
    code = """
            //nants: number of antennas in each field
            // nfields: number of fields

            let nants = ants.length;
            let nfields = p1.length / nants;
            //to keep track of the last antenna number visibilities because
            //p1 and p2 is are single lists containing all the elements in
            //all fields
            let ant_count = 0;

            for(f=0; f<nfields; f++){
                for(a=0; a<nants; a++){
                    if (this.active.includes(f)){
                        p1[a+ant_count].visible = true;
                        p2[a+ant_count].visible = true;
                    }
                    else{
                        p1[a+ant_count].visible = false;
                        p2[a+ant_count].visible = false;
                    }
                }
                ant_count+=nants;
            }



           """
    return code


def axis_fs_callback():
    code = """
            let naxes = ax1.length;


            for(i=0; i<naxes; i++){
                ax1[i].axis_label_text_font_size = `${this.value}pt`;
                ax2[i].axis_label_text_font_size = `${this.value}pt`;
            }
           """
    return code


def title_fs_callback():
    code = """
            last_idx = ax1.length-1;
            ax1[last_idx].text_font_size = `${this.value}px`;
            ax2[last_idx].text_font_size = `${this.value}px`;
           """
    return code


def create_legend_batches(num_leg_objs, li_ax1, li_ax2, lierr_ax1, lierr_ax2, batch_size=16):
    """Automates creation of antenna batches of 16 each unless otherwise

        batch_0 : li_ax1[:16]
        equivalent to
        batch_0 = li_ax1[:16]

    Inputs
    ------

    batch_size: int
                Number of antennas in a legend object
    num_leg_objs: int
                Number of legend objects to be created
    li_ax1: list
                List containing all legend items for antennas for 1st figure
                Items are in the form (antenna_legend, [glyph])
    li_ax2: list
                List containing all legend items for antennas for 2nd figure
                Items are in the form (antenna_legend, [glyph])
    lierr_ax1: list
                List containing legend items for errorbars for 1st figure
                Items are in the form (error_legend, [glyph])
    lierr_ax2: list
                List containing legend items for errorbars for 2nd figure
                Items are in the form (error_legend, [glyph])

    Outputs
    -------

    (bax1, bax1_err, bax2, bax2_err): Tuple
                Tuple containing List of lists which have batch_size number of legend items for each batch.
                Results in batches for figure1 antenna legends, figure1 error legends, figure2 antenna legends, figure2 error legends.

                e.g bax1 = [[batch0], [batch1], ...,  [batch_numOfBatches]]

    """

    bax1, bax1_err, bax2, bax2_err = [], [], [], []

    # condense the returned list if two fields were plotted
    li_ax1, li_ax2, lierr_ax1, lierr_ax2 = list(map(condense_legend_items,
                                                    [li_ax1, li_ax2,
                                                     lierr_ax1, lierr_ax2]))

    j = 0
    for i in range(num_leg_objs):
        # in case the number is not a multiple of 16 or is <= num_leg_objs
        # or on the last iteration

        if i == num_leg_objs:
            bax1.extend([li_ax1[j:]])
            bax2.extend([li_ax2[j:]])
            bax1_err.extend([lierr_ax1[j:]])
            bax2_err.extend([lierr_ax2[j:]])
        else:
            bax1.extend([li_ax1[j:j + batch_size]])
            bax2.extend([li_ax2[j:j + batch_size]])
            bax1_err.extend([lierr_ax1[j:j + batch_size]])
            bax2_err.extend([lierr_ax2[j:j + batch_size]])

        j += batch_size

    return bax1, bax1_err, bax2, bax2_err


def create_legend_objs(num_leg_objs, bax1, baerr_ax1, bax2, baerr_ax2):
    """Creates legend objects using items from batches list
       Legend objects allow legends be positioning outside the main plot

   Inputs
   ------
   num_leg_objs: int
                 Number of legend objects to be created
   bax1: list
         Batches for antenna legends of 1st figure
   bax2: list
         Batches for antenna legends of 2nd figure
   baerr_ax1: list
         Batches for error bar legends of 1st figure
   baerr_ax2: list
         Batches for error bar legends of 2nd figure


   Outputs
   -------
   (lo_ax1, loerr_ax1, lo_ax2, loerr_ax2) tuple
            Tuple containing dictionaries with legend objects for
            ax1 antenna legend objects, ax1 error bar legend objects,
            ax2 antenna legend objects, ax2 error bar legend objects

            e.g.
            leg_0 : Legend(items=batch_0, location='top_right', click_policy='hide')
            equivalent to
            leg_0 = Legend(
                items=batch_0, location='top_right', click_policy='hide')
    """

    lo_ax1, lo_ax2, loerr_ax1, loerr_ax2 = {}, {}, {}, {}

    l_opts = dict(click_policy='hide',
                  orientation='horizontal',
                  label_text_font_size='9pt',
                  visible=False,
                  glyph_width=10)

    for i in range(num_leg_objs):
        lo_ax1['leg_%s' % str(i)] = Legend(items=bax1[i],
                                           **l_opts)
        lo_ax2['leg_%s' % str(i)] = Legend(items=bax2[i],
                                           **l_opts)
        loerr_ax1['leg_%s' % str(i)] = Legend(items=baerr_ax1[i],
                                              **l_opts)
        loerr_ax2['leg_%s' % str(i)] = Legend(items=baerr_ax2[i],
                                              **l_opts)

    return lo_ax1, loerr_ax1, lo_ax2, loerr_ax2


def gen_checkbox_labels(batch_size, num_leg_objs, antnames):
    """ Auto-generating Checkbox labels

    Inputs
    ------
    batch_size: int
                Number of items in a single batch
    num_leg_objs: int
                Number of legend objects / Number of batches
    Outputs
    ------
    labels: list
            Batch labels for the check box
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
    """Save [and show] resultant HTML file
    Inputs
    ------
    hname: str
           HTML Output file name
    plot_layout: Bokeh layout object
                 Layout of the Bokeh plot, could be row, column, gridplot

    Outputs
    -------
    Nothing
    """
    output_file(hname + ".html")
    output = save(plot_layout, hname + ".html", title=hname)
    # uncomment next line to automatically plot on web browser
    # show(layout)


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
                         major_label_orientation='horizontal',
                         ticker=BasicTicker(desired_num_ticks=12),
                         axis_label_text_font_style='normal')
    fig.add_layout(linaxis, 'above')
    return fig


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
    field_names = vu.get_fields(tab_name).data.compute()
    field_name = field_name.upper()

    if field_name in field_names:
        field_id = field_names.index(field_name)
        return int(field_id)
    else:
        return -1


def get_tooltip_data(xds_table_obj, gtype, antnames, freqs):
    """Function to get the data to be displayed on the mouse tooltip on the plots.
    Inputs
    ------
    xds_table_obj: xarray dataset
    gtype: str
           Type of gain table being plotted
    antnames: list
              List of the antenna names

    Outputs
    -------
    spw_id: ndarray
            Spectral window ids
    scan_no: ndarray
             scan ids
    ttip_antnames: array
                   Antenna names to show up on the tooltips


    """
    spw_id = xds_table_obj.SPECTRAL_WINDOW_ID.data.astype(np.int)
    scan_no = xds_table_obj.SCAN_NUMBER.data.astype(np.int)
    ant_id = xds_table_obj.ANTENNA1.data.astype(np.int)

    spw_id, scan_no, ant_id = compute(spw_id, scan_no, ant_id)

    # get available antenna names from antenna id
    ttip_antnames = np.array([antnames[x] for x in ant_id])

    # get the number of channels
    nchan = freqs.size

    if gtype == 'B':
        spw_id = spw_id[0].repeat(nchan, axis=0)
        scan_no = scan_no[0].repeat(nchan, axis=0)
        ant_id = ant_id[0].repeat(nchan, axis=0)
        ttip_antnames = antnames[ant_id]
    return spw_id, scan_no, ttip_antnames


def stats_display(tab_name, gtype, ptype, corr, field):
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
    subtable = get_table(tab_name, where='FIELD_ID=={}'.format(field))[0]

    dobj = DataCoreProcessor(subtable, tab_name, gtype, corr=corr, flag=True)

    if gtype == 'K':
        y1 = dobj.y_only('delay').y
        med_y1 = np.nanmedian(y1)
        text = "Field {}: Median Delay: {:.4f}".format(field, med_y1)
        text2 = ' '
        return text, text2

    if ptype == 'ap':
        y1 = dobj.y_only('amplitude').y
        y2 = dobj.y_only('phase').y
        med_y1 = np.nanmedian(y1)
        med_y2 = np.nanmedian(y2)
        text = "Field {} Median: {:.4f}".format(field, med_y1)
        text2 = "Field {} Median: {:.4f}{}".format(field, med_y2,
                                                   u"\u00b0")
    else:
        y1 = dobj.y_only('real').y
        y2 = dobj.y_only('imaginary').y

        med_y1 = np.nanmedian(y1)
        med_y2 = np.nanmedian(y2)
        text = "Field {} Median: {:.4f}".format(field, med_y1)
        text2 = "Field {} Median: {:.4f}".format(field, med_y2)

    return text, text2


def autofill_gains(t, g):
    """Normalise length of f and g lists to the length of
       t list. This function is meant to support  the ability to specify multiple gain tables while only specifying single values for gain table types. An assumption will be made that for all the specified tables, the same gain table type will be used.

    Inputs
    ------
    t: list
          list of the gain tables.

    g: list
           type of gain table [B,G,K,F].


    Outputs
    -------
    f: list
                    lists of length lengthof(t) containing gain types.
    """
    ltab = len(t)
    lgains = len(g)

    if ltab != lgains and lgains == 1:
        g = g * ltab
    return g


def get_argparser():
    """Get argument parser"""
    parser = ArgumentParser(usage='%(prog)s [options] <value>',
                            description='A RadioAstronomy Visibility and Gains Inspector')
    parser.add_argument('-a', '--ant', dest='plotants', type=str, metavar=' ',
                        help='Plot only this antenna, or comma-separated list\
                              of antennas',
                        default=[-1])
    parser.add_argument('-c', '--corr', dest='corr', type=int, metavar=' ',
                        help='Correlation index to plot (usually just 0 or 1,\
                              default = 0)',
                        default=0)
    parser.add_argument('--cmap', dest='mycmap', type=str, metavar=' ',
                        help='Matplotlib colour map to use for antennas\
                             (default=coolwarm)',
                        default='coolwarm')
    parser.add_argument('-d', '--doplot', dest='doplot', type=str,
                        metavar=' ',
                        help='Plot complex values as amp and phase (ap)'
                        'or real and imag (ri) (default = ap)', default='ap')
    parser.add_argument('-f', '--field', dest='fields', nargs='*', type=str,
                        metavar=' ', help='Field ID(s) / NAME(s) to plot',
                        default=None)
    parser.add_argument('-g', '--gaintype', nargs='+', type=str, metavar=' ',
                        dest='gain_types', choices=['B', 'G', 'K', 'F'],
                        help='Type of table(s) to be plotted: B, G, K, F',
                        default=[])
    parser.add_argument('--htmlname', dest='html_name', type=str, metavar=' ',
                        help='Output HTMLfile name', default='')
    parser.add_argument('-p', '--plotname', dest='image_name', type=str,
                        metavar=' ', help='Output png/svg image name',
                        default='')
    parser.add_argument('-t', '--table', dest='mytabs',
                        nargs='+', type=str, metavar=(' '),
                        help='Table(s) to plot (default = None)', default=[])
    parser.add_argument('--t0', dest='t0', type=float, metavar=' ',
                        help='Minimum time to plot (default = full range)',
                        default=-1)
    parser.add_argument('--t1', dest='t1', type=float, metavar=' ',
                        help='Maximum time to plot (default = full range)',
                        default=-1)
    parser.add_argument('--yu0', dest='yu0', type=float, metavar=' ',
                        help='Minimum y-value to plot for upper panel (default=full range)',
                        default=-1)
    parser.add_argument('--yu1', dest='yu1', type=float, metavar=' ',
                        help='Maximum y-value to plot for upper panel (default=full range)',
                        default=-1)
    parser.add_argument('--yl0', dest='yl0', type=float, metavar=' ',
                        help='Minimum y-value to plot for lower panel (default=full range)',
                        default=-1)
    parser.add_argument('--yl1', dest='yl1', type=float, metavar=' ',
                        help='Maximum y-value to plot for lower panel (default=full range)',
                        default=-1)

    return parser


def condense_legend_items(inlist):
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


def main(**kwargs):
    """Main function"""
    if len(kwargs) == 0:
        NB_RENDER = False

        parser = get_argparser()
        options = parser.parse_args()

        corr = int(options.corr)
        doplot = options.doplot
        field_ids = options.fields
        gain_types = options.gain_types
        html_name = options.html_name
        image_name = options.image_name
        mycmap = options.mycmap
        mytabs = options.mytabs
        plotants = options.plotants
        t0 = options.t0
        t1 = options.t1
        yu0 = options.yu0
        yu1 = options.yu1
        yl0 = options.yl0
        yl1 = options.yl1

    else:
        NB_RENDER = True

        field_ids = kwargs.get('fields', None)
        doplot = kwargs.get('doplot', 'ap')
        plotants = kwargs.get('plotants', [-1])
        corr = int(kwargs.get('corr', 0))
        t0 = float(kwargs.get('t0', -1))
        t1 = float(kwargs.get('t1', -1))
        yu0 = float(kwargs.get('yu0', -1))
        yu1 = float(kwargs.get('yu1', -1))
        yl0 = float(kwargs.get('yl0', -1))
        yl1 = float(kwargs.get('yl1', -1))
        mycmap = str(kwargs.get('mycmap', 'coolwarm'))
        image_name = str(kwargs.get('image_name', ''))
        mytabs = kwargs.get('mytabs', [])
        gain_types = kwargs.get('gain_types', [])

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

        # reinitialise plotant list for each table
        if NB_RENDER:
            plotants = kwargs.get('plotants', [-1])
        else:
            plotants = options.plotants

        tt = get_table(mytab)[0]

        antnames = vu.get_antennas(mytab).data.compute()

        ants = np.unique(tt.ANTENNA1.data.compute())

        fields = np.unique(tt.FIELD_ID.data.compute())

        ncorrs = tt.FLAG.corr.data

        # convert field ids to strings
        if field_ids is None:
            field_ids = [str(f) for f in fields.tolist()]

        freqs = (vu.get_frequencies(mytab, spwid=0) / GHZ).data.compute()

        # setting up colors for the antenna plots
        cNorm = colors.Normalize(vmin=0, vmax=len(ants) - 1)
        mymap = cm = cmx.get_cmap(mycmap)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=mymap)

        if plotants[0] != -1:
            # creating a list for the antennas to be plotted
            plotants = plotants.split(',')

            for ant in plotants:
                if int(ant) not in ants:
                    plotants.remove(ant)
                    logger.info('Antenna ID {} not found.'.format(ant))

            # check if plotants still has items
            if len(plotants) == 0:
                logger.error('Exiting: No valid antennas requested')
                sys.exit(-1)
            else:
                plotants = np.array(plotants, dtype=int)
        else:
            plotants = ants

        # creating bokeh figures for plots
        # linking plots ax1 and ax2 via the x_axes because of similarities in
        # range
        TOOLS = dict(tools='box_select, box_zoom, reset, pan, save,\
                            wheel_zoom, lasso_select')
        ax1 = figure(**TOOLS)
        ax2 = figure(x_range=ax1.x_range, **TOOLS)

        # initialise plot containers
        ax1_plots = []
        ax2_plots = []

        # forming Legend object items for data and errors
        legend_items_ax1 = []
        legend_items_ax2 = []
        legend_items_err_ax1 = []
        legend_items_err_ax2 = []

        stats_ax1 = []
        stats_ax2 = []
        for field in field_ids:

            if field.isdigit():
                field = int(field)
            else:
                field = name_2id(field, field_src_ids)

            if int(field) not in fields.tolist():
                logger.info(
                    'Skipping table: {} : Field id {} not found.'.format(mytab, field))
                continue

            stats_text = stats_display(mytab, gain_type, doplot, corr, field)
            stats_ax1.append(stats_text[0])
            stats_ax2.append(stats_text[1])

            newtab = get_table(mytab, fid=field)[0]

            # for each antenna
            for ant in plotants:

                # creating legend labels
                antlabel = antnames[ant]
                legend = antnames[ant]
                legend_err = "E" + antnames[ant]

                # creating colors for maps
                y1col = y2col = scalarMap.to_rgba(float(ant), bytes=True)[:-1]

                subtab = newtab.where(newtab.ANTENNA1 == int(ant), drop=True)

                # To Do: Have a flag option in cmdline
                data_obj = DataCoreProcessor(subtab, mytab, gain_type,
                                             fid=field, antenna=ant,
                                             doplot=doplot, corr=corr,
                                             flag=True)
                ready_data = data_obj.act()

                prepd_x = ready_data.x
                xlabel = ready_data.x_label

                y1 = ready_data.y1
                y1_err = ready_data.y1_err
                y1label = ready_data.y1_label
                y2 = ready_data.y2
                y2_err = ready_data.y2_err
                y2label = ready_data.y2_label

                # for tooltips
                spw_id, scan_no, ttip_antnames = get_tooltip_data(subtab,
                                                                  gain_type,
                                                                  antnames,
                                                                  freqs)

                tab_tooltips = [("(x, y)", "($x, $y)"),
                                ("spw", "@spw"),
                                ("scan_id", "@scanid"),
                                ("antenna", "@antname")]

                hover = HoverTool(tooltips=tab_tooltips,
                                  mode='mouse', point_policy='snap_to_data')
                hover2 = HoverTool(tooltips=tab_tooltips,
                                   mode='mouse', point_policy='snap_to_data')

                ax1.xaxis.axis_label = ax1_xlabel = xlabel
                ax2.xaxis.axis_label = ax2_xlabel = xlabel
                ax1.yaxis.axis_label = ax1_ylabel = y1label
                ax2.yaxis.axis_label = ax2_ylabel = y2label

                ax1.axis.axis_label_text_font_style = 'normal'
                ax2.axis.axis_label_text_font_style = 'normal'

                if doplot == 'ap':
                    ax2.yaxis[0].formatter = PrintfTickFormatter(
                        format=u"%f\u00b0")

                if gain_type == 'B':
                    if y1.ndim > 1:
                        y1 = y1[:, 0]
                        y2 = y2[:, 0]

                    if ant == plotants[-1]:
                        ax1 = add_axis(ax1, [freqs[0], freqs[-1]],
                                       ax_label='Frequency [GHz]')
                        ax2 = add_axis(ax2, [freqs[0], freqs[-1]],
                                       ax_label='Frequency [GHz]')

                if gain_type == 'K':
                    ax1_ylabel = y1label.replace('[ns]', '')
                    ax2_ylabel = y2label.replace('[ns]', '')

                source = ColumnDataSource(data={'x': prepd_x,
                                                'y1': y1,
                                                'y2': y2,
                                                'spw': spw_id,
                                                'scanid': scan_no,
                                                'antname': ttip_antnames})

                p1, p1_err, p2, p2_err = make_plots(
                    source=source, color=y1col, ax1=ax1, ax2=ax2, fid=field,
                    y1_err=y1_err, y2_err=y2_err)

                # hide all the other plots until legend is clicked
                if ant > 0:
                    p1.visible = p2.visible = False

                # collecting plot states for each iterations
                ax1_plots.append(p1)
                ax2_plots.append(p2)

                # forming legend object items
                legend_items_ax1.append((legend, [p1]))
                legend_items_ax2.append((legend, [p2]))
                # for the errors
                legend_items_err_ax1.append((legend_err, [p1_err]))
                legend_items_err_ax2.append((legend_err, [p2_err]))

                subtab.close()

        tt.close()

        # configuring titles for the plots
        ax1_title = Title(text="{} vs {} ({})".format(ax1_ylabel,
                                                      ax1_xlabel,
                                                      " ".join(stats_ax1)),
                          align='center', text_font_size='15px')
        ax2_title = Title(text="{} vs {} ({})".format(ax2_ylabel,
                                                      ax2_xlabel,
                                                      " ".join(stats_ax2)),
                          align='center', text_font_size='15px')

        ax1.add_tools(hover)
        ax2.add_tools(hover2)
        # LEGEND CONFIGURATIONS
        # determining the number of legend objects required to be created
        # for each plot
        num_legend_objs = int(np.ceil(len(plotants) / BATCH_SIZE))

        batches_ax1, batches_ax1_err, batches_ax2, batches_ax2_err = \
            create_legend_batches(num_legend_objs, legend_items_ax1,
                                  legend_items_ax2, legend_items_err_ax1,
                                  legend_items_err_ax2,
                                  batch_size=BATCH_SIZE)

        legend_objs_ax1, legend_objs_ax1_err, legend_objs_ax2,\
            legend_objs_ax2_err = create_legend_objs(num_legend_objs,
                                                     batches_ax1,
                                                     batches_ax1_err,
                                                     batches_ax2,
                                                     batches_ax2_err)

        # adding legend objects to the layouts
        for i in range(num_legend_objs):
            ax1.add_layout(legend_objs_ax1['leg_%s' % str(i)], 'below')
            ax2.add_layout(legend_objs_ax2['leg_%s' % str(i)], 'below')

            ax1.add_layout(legend_objs_ax1_err['leg_%s' % str(i)], 'below')
            ax2.add_layout(legend_objs_ax2_err['leg_%s' % str(i)], 'below')

        # adding plot titles
        ax2.add_layout(ax2_title, 'above')
        ax1.add_layout(ax1_title, 'above')

        ######################################################################
        ################ Defining widgets ###################################
        ######################################################################

        # widget dimensions
        w_dims = dict(width=150, height=30)

        # creating and configuring Antenna selection buttons
        ant_select = Toggle(label='Select All Antennas',
                            button_type='success', **w_dims)

        # configuring toggle button for showing all the errors
        toggle_err = Toggle(label='Show All Error bars',
                            button_type='warning', **w_dims)

        ant_labs = gen_checkbox_labels(BATCH_SIZE, num_legend_objs, antnames)

        batch_select = CheckboxGroup(labels=ant_labs, active=[],
                                     width=150)

        # Dropdown to hide and show legends
        legend_toggle = Select(title="Showing Legends: ", value="non",
                               options=[("alo", "Antennas"),
                                        ("elo", "Errors"), ("non", "None")],
                               width=150, height=45)

        # creating size slider for the plots
        size_slider = Slider(end=15, start=1, step=0.5,
                             value=4, title='Glyph size',
                             **w_dims)

        # Alpha slider for the glyphs
        alpha_slider = Slider(end=1, start=0.1, step=0.1, value=1,
                              title='Glyph alpha', **w_dims)

        fnames = vu.get_fields(mytab).data.compute()
        fsyms = [u'\u2B24', u'\u25C6', u'\u25FC', u'\u25B2',
                 u'\u25BC', u'\u2B22']
        field_labels = ["Field {} {}".format(fnames[int(x)],
                                             fsyms[int(x)]) for x in fields]

        field_selector = CheckboxGroup(labels=field_labels,
                                       active=fields.tolist(),
                                       **w_dims)

        axis_fontslider = Slider(end=20, start=3, step=0.5, value=10,
                                 title='Axis label size', **w_dims)
        title_fontslider = Slider(end=35, start=10, step=1, value=15,
                                  title='Title size', **w_dims)

        ######################################################################
        ############## Defining widget Callbacks ############################
        ######################################################################

        ant_select.callback = CustomJS(args=dict(glyph1=legend_items_ax1,
                                                 glyph2=legend_items_ax2,
                                                 batchsel=batch_select),
                                       code=ant_select_callback())

        toggle_err.callback = CustomJS(args=dict(err1=legend_items_err_ax1,
                                                 err2=legend_items_err_ax2),
                                       code=toggle_err_callback())

        # BATCH SELECTION
        batch_select.callback = CustomJS(
            args=dict(bax1=batches_ax1,
                      bax1_err=batches_ax1_err,
                      bax2=batches_ax2,
                      bax2_err=batches_ax2_err,
                      batch_size=BATCH_SIZE,
                      antsel=ant_select),
            code=batch_select_callback())

        legend_toggle.callback = CustomJS(
            args=dict(
                loax1=listvalues(legend_objs_ax1),
                loax1_err=listvalues(legend_objs_ax1_err),
                loax2=listvalues(legend_objs_ax2),
                loax2_err=listvalues(legend_objs_ax2_err)),
            code=legend_toggle_callback())

        size_slider.callback = CustomJS(args={'slide': size_slider,
                                              'p1': ax1_plots,
                                              'p2': ax2_plots},
                                        code=size_slider_callback())
        alpha_slider.callback = CustomJS(args={'alpha': alpha_slider,
                                               'p1': ax1_plots,
                                               'p2': ax2_plots},
                                         code=alpha_slider_callback())
        field_selector.callback = CustomJS(args={'fselect': field_selector,
                                                 'p1': ax1_plots,
                                                 'p2': ax2_plots,
                                                 'ants': plotants},
                                           code=field_selector_callback())
        axis_fontslider.callback = CustomJS(args=dict(ax1=ax1.axis,
                                                      ax2=ax2.axis),
                                            code=axis_fs_callback())
        title_fontslider.callback = CustomJS(args=dict(ax1=ax1.above,
                                                       ax2=ax2.above),
                                             code=title_fs_callback())

        a = row([ant_select, toggle_err, legend_toggle, size_slider,
                 alpha_slider])
        b = row([batch_select, field_selector, axis_fontslider,
                 title_fontslider])
        #*stats

        plot_widgets = widgetbox([a, b], sizing_mode='scale_both')

        # setting the gridspecs
        # gridplot while maintaining the set aspect ratio
        grid_specs = dict(plot_width=PLOT_WIDTH,
                          plot_height=PLOT_HEIGHT,
                          sizing_mode='stretch_width')
        if gain_type != 'K':
            layout = gridplot([[plot_widgets], [ax1, ax2]],
                              **grid_specs)
        else:
            layout = gridplot([[plot_widgets], [ax1]],
                              **grid_specs)

        final_layout.append(layout)
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
            html_name = "{}_corr_{}_{}_field_{}".format(mytab, corr,
                                                        doplot,
                                                        ''.join(field_ids))
            save_html(html_name, final_layout)

        logger.info("Rendered: {}.html".format(html_name))

    else:
        output_notebook()
        for lays in final_layout:
            show(lays)


def plot_table(mytabs, gain_types, fields, **kwargs):
    """
    Function for plotting tables

    Inputs
    --------
    Required
    --------
        mytabs       : The table (list of tables) to be plotted
        gain_types   : Cal-table (list of caltypes) type to be plotted.
                      Can be either 'B'-bandpass, 'G'-gains, 'K'-delay or 'F'-flux (default=None)
        fields       : Field ID / Name (list of field ids or name) to plot
                      (default = 0)',default=0)

    Optional
    --------

        doplot      : Plot complex values as amp and phase (ap) or real and
                      imag (ri) (default = ap)',default='ap')
        plotants    : Plot only this antenna, or comma-separated string of
                      antennas',default=[-1])
        corr        : Correlation index to plot (usually just 0 or 1,
                      default = 0)',default=0)
        t0          : Minimum time to plot (default = full range)',default=-1)
        t1          : Maximum time to plot (default = full range)',default=-1)
        yu0         : Minimum y-value to plot for upper panel
                      (default = full   range)',default=-1)
        yu1         : Maximum y-value to plot for upper panel
                      (default = full range)',default=-1)
        yl0         : Minimum y-value to plot for lower panel
                      (default = full range)',default=-1)
        yl1         : Maximum y-value to plot for lower panel
                      (default = full range)',default=-1)
        mycmap      : Matplotlib colour map to use for antennas
                      (default = coolwarm)',default='coolwarm')
        image_name  : Output image name (default = something sensible)'


    Outputs
    ------
    Returns nothing

    """
    if mytabs == None:
        print("Please specify a gain table to plot.")
        logger.error('Exiting: No gain table specfied.')
        sys.exit(-1)
    else:
        # getting the name of the gain table specified
        if type(mytabs) is str:
            kwargs['mytabs'] = [str(mytabs)]
            kwargs['gain_types'] = [str(gain_types)]
            kwargs['fields'] = [str(fields)]
        else:
            kwargs['mytabs'] = [x.rstrip("/") for x in mytabs]
            kwargs['gain_types'] = [str(x) for x in gain_types]
            kwargs['fields'] = [str(x) for x in fields]

    main(**kwargs)

    return
