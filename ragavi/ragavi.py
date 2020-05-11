import glob
import numpy as np
import re
import sys
import os

from collections import OrderedDict, namedtuple
from datetime import datetime

import dask.array as da
import daskms as xm
import xarray as xr

from dask import delayed, compute
from bokeh.io import (export_png, export_svgs, output_file, output_notebook,
                      save, show)
from bokeh.layouts import column, grid, gridplot, layout, row
from bokeh.models import (Button, CheckboxGroup,
                          ColumnDataSource, CustomJS, HoverTool,
                          Legend, LinearAxis, Toolbar, PrintfTickFormatter,
                          Slider, Scatter, Toggle, Whisker)

from bokeh.models.widgets import DataTable, TableColumn, Div, PreText
from itertools import product
from pyrap.tables import table

import ragavi.utils as vu
from ragavi.plotting import create_bk_fig, add_axis

# defining some constants
_FLAG_DATA_ = True

_PLOT_WIDTH_ = 900
_PLOT_HEIGHT_ = 710

_GHZ_ = 1e9
_NB_RENDER_ = False
_BATCH_SIZE_ = 16

logger = vu.logger
excepthook = vu.__handle_uncaught_exceptions


#################### Define Dataprocessor ##############################

class DataCoreProcessor:
    """Process gain table data into forms desirable for visualisation. This class is responsible for:
        * x and y data column selection
        * x and y label generation
        * Preparation of x and y data into ColumnDataSource friendly forms
        * Flagging


    Parameters
    ----------
    xds_table_obj : :obj:`xarray.Dataset`
        Table object from which data will be extracted
    ms_name : :obj:`str`
        Name / path to the table being plotted
    gtype : :obj:`str`
        Gain table type
    yaxis: :obj:`str`
        Desired y-axis being plotted
    corr : :obj:`int`
        Correlation index being plotted
    chan : :obj:`slice`
        A slicer to select channels
    ddid : :obj:`int`
        SPW id fo data being plotted
    flag : :obj:`bool`
        To flag or not. Default is True.
    fid : :obj:`int`
        Field id being plotted
    kx : :obj:`str`
        Applicable to delay (K) tables. Determines the x-axis of the plot. Default is time.
    """

    def __init__(self, xds_table_obj, ms_name, gtype, yaxis, chan=None,
                 corr=0, ddid=None, fid=None, flag=None, kx=None):

        self.xds_table_obj = xds_table_obj
        self.ms_name = ms_name
        self.gtype = gtype
        self.fid = fid
        self.chan = chan
        self.corr = corr
        self.ddid = ddid
        self.flag = flag
        self.kx = kx
        self.xaxis = self.set_xaxis()
        self.yaxis = yaxis

    def set_xaxis(self):
        if self.gtype in ['B', 'D']:
            xaxis = "channel"
        elif self.gtype in ['G', 'F']:
            xaxis = "time"
        elif self.gtype == "K":
            if self.kx == "time":
                xaxis = "time"
            else:
                xaxis = "antenna"
        return xaxis

    def process_data(self, ydata, yaxis=None):
        """Abstraction for processing y-data passes it to the processing function.

        Parameters
        ----------
        ydata: :obj:`xarray.DataArray`
               y-data to process

        Returns
        -------
        y: :obj:`xarray.DataArray`
           Processed :attr:`ydata`
        """
        if yaxis is None:
            yaxis = self.yaxis

        if yaxis == "amplitude":
            y = vu.calc_amplitude(ydata)
        elif yaxis == "imaginary":
            y = vu.calc_imaginary(ydata)
        elif yaxis == "phase":
            y = vu.calc_phase(ydata, wrap=False)
        elif yaxis == "real":
            y = vu.calc_real(ydata)
        elif yaxis == "delay" or yaxis == "error":
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
            # TODO check if this flattening works proper
            y = y.T.flatten()
            return y

    def get_errors(self):
        """Get error data from PARAMERR column.
        Returns
        -------
        errors: :obj:`xarray.DataArray`
            Data from the PARAMERR column
        """
        errors = self.xds_table_obj.PARAMERR
        return errors

    def get_xaxis_data(self):
        """Get x-axis data. This function also returns the relevant x-axis label

        Returns
        -------
        xdata : :obj:`xarray.DataArray`
            X-axis data depending  x-axis selected.
        x_label : :obj:`str`
            Label to appear on the x-axis of the plots.
        """

        if self.xaxis in ["antenna", "antenna1"]:
            xdata = self.xds_table_obj.ANTENNA1
            x_label = "Antenna"
        elif self.xaxis == "channel":
            xdata = vu.get_frequencies(self.ms_name, spwid=self.ddid,
                                       chan=self.chan)
            x_label = "Channel"
        elif self.xaxis == "time":
            xdata = self.xds_table_obj.TIME
            x_label = "Time"
        else:
            logger.error("Invalid xaxis name")
            return
        return xdata, x_label

    def prep_xaxis_data(self, xdata, freq=None):
        """Prepare the x-axis data for plotting. Selections also performed here.

        Parameters
        ----------
        xdata: :obj:`xarray.DataArray`
            x-axis data depending  x-axis selected

        freq: :obj:`xarray.DataArray` or :obj:`float`
            Frequency(ies) from which corresponding wavelength will be obtained.
            Note
            ----
                Only required when xaxis specified is "uvwave"

        Returns
        -------
        prepdx: :obj:`xarray.DataArray`
            Prepared :attr:`xdata`
        """
        if self.xaxis in ["channel", "frequency"]:
            prepdx = xdata.chan
        elif self.xaxis == "time":
            prepdx = vu.time_convert(xdata)
        elif self.xaxis in ["antenna1", "antenna"]:
            prepdx = np.full((self.xds_table_obj.row.size), (xdata))
        return prepdx

    def get_yaxis_data(self):
        """Extract the required column for the y-axis data.

        Returns
        -------
        ydata, y_label : :obj:`xarray.DataArray`, :obj:`str`
                         :attr:`ydata` containing y-axis data depending  y-axis selected and `y_label` which is the label to appear on the y-axis of the plots.
        """

        # default data column
        datacol = "CPARAM"

        if self.yaxis == "amplitude":
            y_label = "Amplitude"
        elif self.yaxis == "imaginary":
            y_label = "Imaginary"
        elif self.yaxis == "phase":
            y_label = "Phase [deg]"
        elif self.yaxis == "real":
            y_label = "Real"
        elif self.yaxis == "delay":
            datacol = "FPARAM"
            y_label = "Delay[ns]"
        # attempt to get the specified column from the table
        try:
            ydata = self.xds_table_obj[datacol]
        except KeyError:
            logger.exception("Column '{}'' not Found".format(datacol))
            return sys.exit(-1)

        return ydata, y_label

    def prep_yaxis_data(self, ydata, yaxis=None):
        """Process data for the y-axis.

        This function is responsible for :

            * Correlation selection
            * Flagging
            * Conversion form complex to the required form
            * Data transposition if `yaxis` is frequency or channel

        Data selection and flagging are done by this function itself, however ap and ri conversion are handled by :meth:`ragavi.ragavi.DataCoreProcessor.compute_ydata`

        Parameters
        ----------
        ydata : :obj:`xarray.DataArray`
            y-axis data to be processed
        Returns
        -------
        y : :obj:`xarray.DataArray`
            Processed :attr:`ydata`
        """
        flags = vu.get_flags(self.xds_table_obj)

        if self.corr != None:
            ydata = ydata.sel(corr=self.corr)
            flags = flags.sel(corr=self.corr)

        # if flagging enabled return a list of DataArrays otherwise return a
        # single dataarray
        if self.flag:
            # if flagging is activated select data only where flag mask is 0
            processed = self.process_data(ydata)
            y = processed.where(flags == False)
        else:
            y = self.process_data(ydata)

        if self.xaxis == "channel":
            y = y.T
        return y

    def blackbox(self):
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

        Data = namedtuple("Data", "x x_label y y_label y_err")
        x = self.x_only()
        xdata = x.x
        xlabel = x.x_label

        y = self.y_only()
        ydata = y.y
        ylabel = y.y_label
        hi, lo = y.y_err

        if self.xaxis == "channel":
            xdata = self.flatten_bandpass_data(ydata, x=xdata)
            ydata = self.flatten_bandpass_data(ydata)
            hi = self.flatten_bandpass_data(hi)
            lo = self.flatten_bandpass_data(lo)

        xdata, ydata, hi, lo = compute(xdata, ydata, hi, lo)

        d = Data(x=xdata, x_label=xlabel, y=ydata,
                 y_label=ylabel, y_err=(hi, lo))
        return d

    def act(self):
        """Activate the :meth:`ragavi.ragavi.DataCoreProcessor.blackbox`
        """
        return self.blackbox()

    def x_only(self):
        """Return only x-axis data and label

        Returns
        -------
        d : :obj:`collections.namedtuple`
            Named tuple containing x-axis data and x-axis label. Items in the tuple can be accessed by using the dot notation.
        """
        Data = namedtuple("Data", "x x_label")
        xdata, xlabel = self.get_xaxis_data()
        xdata = self.prep_xaxis_data(xdata)
        if not isinstance(xdata, np.ndarray):
            xdata = xdata.data
        d = Data(x=xdata, x_label=xlabel)
        return d

    def y_only(self, add_error=True):
        """Return only y-axis data and label

        Returns
        -------
        d : :obj:`collections.namedtuple`
            Named tuple containing x-axis data and x-axis label. Items in the tuple can be accessed by using the dot notation.
        """

        Data = namedtuple("Data", "y y_label y_err")
        ydata, y_label = self.get_yaxis_data()
        p_ydata = self.prep_yaxis_data(ydata)
        if add_error:
            err_data = self.get_errors()
            hi = self.prep_yaxis_data(ydata + err_data)
            lo = self.prep_yaxis_data(ydata - err_data)
        else:
            hi = None
            lo = None

        d = Data(y=p_ydata.data, y_label=y_label, y_err=(hi.data, lo.data))
        return d


################### Some table related functions #######################

def get_table(tab_name, antenna=None, fid=None, spwid=None, where=[],
              group_cols=None):
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
    where: :obj:`list`, optional
        TAQL where clause to be used within the table.

    Returns
    -------
    tab_objs: :obj:`list`
        A list containing :obj:`xarray.Dataset` objects where each item on the list is determined by how the data is grouped

    """

    # defining part of the gain table schema
    tab_schema = {"CPARAM": {"dims": ("chan", "corr")},
                  "FLAG": {"dims": ("chan", "corr")},
                  "FPARAM": {"dims": ("chan", "corr")},
                  "PARAMERR": {"dims": ("chan", "corr")},
                  "SNR": {"dims": ("chan", "corr")},
                  }

    # where is now a list
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
    if len(where) > 0:
        where = " && ".join(where)
    else:
        where = ""

    if group_cols is None:
        group_cols = ["SPECTRAL_WINDOW_ID", "FIELD_ID", "ANTENNA1"]

    try:
        tab_objs = xm.xds_from_table(tab_name, taql_where=where,
                                     table_schema=tab_schema,
                                     group_cols=group_cols)
        return tab_objs
    except:
        logger.exception("""Invalid ANTENNA id, FIELD_ID,
                            SPECTRAL_WINDOW_ID or TAQL clause""")
        sys.exit(-1)


def get_time_range(tab_name, unix_time=True):
    """Get the initial and final TIME column before selections

    Returns
    -------
    init_time, f_time : :obj:`tuple`
        Containing initial time and final time available in the ms
    """

    # unix_time = MJD - munix
    munix = 3506716800.0

    ms = table(tab_name, ack=False)
    i_time = ms.getcell("TIME", 0)
    f_time = ms.getcell("TIME", (ms.nrows() - 1))

    if unix_time:
        i_time = i_time - munix
        f_time = f_time - munix

    ms.close()

    return i_time, f_time


def get_tooltip_data(xds_table_obj, gtype, freqs):
    """Get the data to be displayed on the tool-tip of the plots

    Parameters
    ----------
    gtype: :obj:`str`
        Type of gain table being plotted
    xds_table_obj : `xarray.Dataset`
        xarray-ms table object
    freqs : :obj:`np.array`
        An array containing frequencies for the current SPW

    Returns
    -------
    spw_id : :obj:`numpy.ndarray`
        Spectral window ids
    scan_no : :obj:`numpy.ndarray`
        scan ids

    """
    scan_no = xds_table_obj.SCAN_NUMBER.values
    spw_id = [xds_table_obj.SPECTRAL_WINDOW_ID] * scan_no.size

    # get the number of channels
    nchan = freqs.size

    if gtype in ['B', 'D']:
        # extend array to accommodate for multiple time slots
        spw_id = np.array(spw_id).repeat(nchan, axis=0)
        # tile because of different scan numbers available
        scan_no = np.tile(scan_no, nchan)

    return spw_id, scan_no

#################### JS callbacks ######################################
# Visible data selection callbacks


def ant_select_callback():
    """JS callback for the selection and de-selection of antennas

    Returns
    -------
    code : :obj:`str`
    """

    code = """
            /*ax: list containing glyph renderers
              bsel: batch selection group button
              nbatches: Number of batches available
              fsel: field_selector widget,
              csel: corr_select widget,
              ssel: spw_select widget,
              nfields: Number of available fields,
              ncorrs: Number of avaiblable corrs,
              nspws: Number of spectral windows,
            */
            let nplots = ax.length;
            if (cb_obj.active.includes(0))
                {
                    for(let i=0; i<nplots; i++){
                        ax[i].visible = true;
                    }
                    //activate all the checkboxes whose antennas are active
                    bsel.active = [...Array(nbatches).keys()]
                    csel.active = [...Array(ncorrs).keys()]
                    fsel.active = [...Array(nfields).keys()]
                    ssel.active = [...Array(nspws).keys()]

                }
            else{
                    for(let i=0; i<nplots; i++){
                       ax[i].visible = false;
                    }

                    bsel.active = []
                    csel.active = []
                    fsel.active = []
                    ssel.active = []
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
           ax: List containing glyphs for all antennas, fields and correlations
           count: keeping a cumulative sum of the traverse number
        */

        let count = 0;
        let new_bsize;
        for (let sp=0; sp<nspws; sp++){
            for (let f=0; f<nfields; f++){
                for (let c=0; c<ncorrs; c++){
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
                                ax[count].visible = true;
                                }
                            else{
                                ax[count].visible = false;
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


def corr_select_callback():
    """Correlation selection callback"""
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
           ax: List containing glyphs for a single plot for all antennas, fields and correlations
           count: keeping a cumulative sum of the traverse number
        */

        let count = 0;
        let new_bsize;
        for (let sp=0; sp<nspws; sp++){
            for (let f=0; f<nfields; f++){
                for (let c=0; c<ncorrs; c++){
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
                                ax[count].visible = true;
                                }
                            else{
                                ax[count].visible = false;
                            }
                            count = count + 1;
                        }
                    }
                }
            }
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
         ax: List containing glyphs for all antennas, fields and  correlations
         count: keeping a cumulative sum of the traverse number
        */
        let count = 0;
        let new_bsize;
        for (let sp=0; sp<nspws; sp++){
            for (let f=0; f<nfields; f++){
                for (let c=0; c<ncorrs; c++){
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
                                ax[count].visible = true;
                                }
                            else{
                                ax[count].visible = false;
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
    """SPW selection callaback"""
    code = """
         /*bsize: total number of items in a batch
           bsel: the batch selector group buttons
           ex_ax: for the extra frequency ax(es). Is 0 if none.
           fsel: field selector group buttons
           csel: corr selector group buttons
           nspws: Number of spectral windows
           ncorrs: number of available correlations
           nfields: number of available fields
           nbatches: total number of available batches
           ax: List containing glyphs for all antennas, fields and correlations
           count: keeping a cumulative sum of the traverse number
        */

        let count = 0;
        let new_bsize;
        for (let sp=0; sp<nspws; sp++){
            for (let f=0; f<nfields; f++){
                for (let c=0; c<ncorrs; c++){
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
                                ax[count].visible = true;
                                }
                            else{
                                ax[count].visible = false
                            }
                            count = count + 1;
                        }
                    }
                }
            }

            //Make the extra y-axes visible only if corresponding spw selected
            if (cb_obj.active.includes(sp) && ex_ax[sp].name==`spw${sp}`){
                ex_ax[sp].visible=true;
                console.log("Trueuee");
            }
            else{
                ex_ax[sp].visible=false;

            }

        }

       """
    return code


# Show additional data callbacks

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

            let n_sources =  uf_sources.length;

            if (cb_obj.active.includes(0)){
                for (let i=0; i<n_sources; i++){
                    uf_sources[i].data.y1 = f_sources[i].data.iy1;
                    uf_sources[i].data.y2 = f_sources[i].data.iy2;
                    uf_sources[i].change.emit();
                }
            }
            else{
                for (let i=0; i<n_sources; i++){
                    uf_sources[i].data.y1 = f_sources[i].data.y1;
                    uf_sources[i].data.y2 = f_sources[i].data.y2;
                    uf_sources[i].change.emit();
                }


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
            let n_glyphs = ax.length;
             //if toggle button active
            if (cb_obj.active.includes(0))
                {
                    for(let i=0; i<n_glyphs; i++){
                        //only switch on if corresponding plot is on
                        if(ax[i].visible){
                            ax_err[i].visible = true;
                        }

                    }
                }
            else{
                    for(let i=0; i<n_glyphs; i++){
                        ax_err[i].visible = false;
                    }
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
                //Show only the legend for the first item
                if (cb_obj.active.includes(0)){
                        legs[0].visible = true;
                }

                else{
                        legs[0].visible = false;
                }

           """
    return code


# Plot layout callbacks

def size_slider_callback():
    """JS callback to select size of glyphs

    Returns
    -------
    code : :obj:`str`
    """

    code = """
            let n_rs = ax.length;
            for (let i=0; i<n_rs; i++){
                ax[i].size = cb_obj.value;
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
            let n_rs = ax.length;
            for (let i=0; i<n_rs; i++){
                ax[i].fill_alpha = cb_obj.value;
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
            let n_axes = ax.length;
            for (let i=0; i<n_axes; i++){
                ax[i].axis_label_text_font_size = `${cb_obj.value}pt`;
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
            ax.text_font_size = `${cb_obj.value}pt`;
           """
    return code


# Download data selection

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


####################### Plot Related functions ########################

def condense_legend_items(inlist):
    """Combine renderers of legend items with the same legend labels. Must be 
    done in case there are groups of renderers which have the same label due 
    to iterations, to avoid a case where there are two or more groups of 
    renderers containing the same label name.

    Parameters
    ----------
    inlist : :obj:`list`
             # bokeh.models.annotations.LegendItem>`_
             List containing legend items of the form (label, renders) as 
             described in: `Bokeh legend items <https://bokeh.pydata.org/en/latest/docs/reference/models/annotations.html
    Returns
    -------
    outlist : :obj:`list`
              A reduction of :attr:`inlist` containing renderers sharing the 
              same labels grouped together.
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

    # condense the returned list if two fields / corrs/ spws were plotted
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
        Batches for antenna legends for plot

    Returns
    -------
    lo_ax1 : :obj:`dict`
        Dictionaries with legend objects for a plot


    """

    lo_ax1 = {}

    l_opts = dict(click_policy="hide", glyph_height=20,
                  glyph_width=20, label_text_font_size="9pt",
                  label_text_font="monospace", location="top_left",
                  margin=1, orientation="horizontal",
                  level="annotation", spacing=1,
                  padding=2, visible=False)
    for i in range(num_leg_objs):
        leg1 = Legend(items=bax1[i], name=f"leg{i}", **l_opts)
        lo_ax1["leg_%s" % str(i)] = leg1

    return lo_ax1


def errorbar(x, y, yerr=None, color="red"):
    """Add errorbars to Figure object based on :attr:`x`, :attr:`y` and attr:`yerr`

    Parameters
    ----------
    color : :obj:`str`
        Color for the error bars
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

        ebars = Whisker(source=src, base="base", upper="upper",
                        lower="lower", line_color=color, name="p_whisker",
                        visible=False)
        ebars.upper_head.line_color = color
        ebars.lower_head.line_color = color
    return ebars


def gen_flag_data_markers(y, fid=None, markers=None, fmarker="circle_x"):
    """Generate different markers for where data has been flagged.

    Parameters
    ----------
    fid : :obj:`int`
        Field id number to identify the marker to be used
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
    markers_arr = np.full(y.shape, fill_value=markers[fid], dtype="<U17")

    # mask only where there are nan values
    masked_markers_arr = np.ma.masked_where(np.isnan(y), markers_arr)
    # fill with the different marker
    masked_markers_arr.fill_value = fmarker

    # return filled matrix
    masked_markers_arr = masked_markers_arr.filled()

    return masked_markers_arr


def link_plots(all_figures=None, all_fsources=None, all_ebars=None):
    """ Link all plots generated by this script. Done to unify interactivity 
    within the plots such as panning, zoomin etc.

    Parameters
    ----------
    all_figures : :obj:`list`
        Containing all the figures generated in the plot
    all_fsources: :obj:`list`
        All data sources containing flagged data for all figures
    all_ebars : :obj:`list`
        All error bars for all figures

    Returns
    -------
    all_fsrc, all_ufsrc : :obj:`tuple`
        Unified flagged and un-flagged data sources
    """
    n_figs = len(all_figures)
    fig1 = all_figures[0]
    fig1_ebars = all_ebars[0]

    # number of axes in the figure
    n_axis = len(fig1.axis)

    # number of renderers
    n_rs = len(fig1.renderers)
    all_ufsrc = []
    all_fsrc = []

    for f in range(1, n_figs):
        # link the x- ranges
        all_figures[f].x_range = fig1.x_range
        # link the titles font sizes
        fig1.select(name="p_title")[0].js_link(
            "text_font_size", all_figures[f].select(name="p_title")[0],
            "text_font_size")

        # link x and y-axes font sizes
        for _i in range(n_axis):
            fig1.axis[_i].js_link(
                "axis_label_text_font_size", all_figures[f].axis[_i],
                "axis_label_text_font_size")
            fig1.axis[_i].js_link(
                "visible", all_figures[f].axis[_i],
                "visible")

        for _i in range(n_rs):
            # add y data from fig 1
            fig1.renderers[_i].data_source.add(
                all_figures[f].renderers[_i].data_source.data[f"y{f+1}"],
                name=f"y{f+1}")

            shared_cds = ColumnDataSource(
                data=dict(fig1.renderers[_i].data_source.data))

            # ensure that the cds and its view are the same otherwise error
            all_figures[f].renderers[_i].data_source = shared_cds
            all_figures[f].renderers[_i].view.source = shared_cds
            fig1.renderers[_i].data_source = shared_cds
            fig1.renderers[_i].view.source = shared_cds

            all_fsources[0][_i].add(all_fsources[f][_i].data[f"y{f+1}"],
                                    name=f"y{f+1}")
            all_fsources[0][_i].add(all_fsources[f][_i].data[f"iy{f+1}"],
                                    name=f"iy{f+1}")

            # link visible values for the renderers
            fig1.renderers[_i].js_link("visible",
                                       all_figures[f].renderers[_i],
                                       "visible")
            fig1.renderers[_i].glyph.js_link("size",
                                             all_figures[f].renderers[_i].glyph, "size")
            fig1.renderers[_i].glyph.js_link("fill_alpha",
                                             all_figures[f].renderers[_i].glyph, "fill_alpha")
            if fig1_ebars[_i]:
                fig1_ebars[_i].js_link("visible", all_ebars[f][_i], "visible")
            all_ufsrc.append(shared_cds)
            all_fsrc.append(all_fsources[0][_i])

    return all_fsrc, all_ufsrc


def make_plots(source, fid=0, color="red", yerr=None, yidx=None):
    """Generate a pair of plots

    Parameters
    ----------
    fig : :obj:`bokeh.plotting.figure`
        First figure
    color : :obj:`str`
        Glyph color[s]
    fid : :obj:`int`
        field id number to set the line width
    source : :obj:`bokeh.models.ColumnDataSource`
            Data source for the renderer
    yerr : :obj:`numpy.ndarray`, optional
        Y-axis error margins
    yidx : :obj:`int`
        Current enumerated y-axis number. Used for keeping track of the 
        figures.


    Returns
    -------
    p_glyph, ebars: (:obj:`bokeh.models.Glyph`, :obj:`bokeh.models.Whisker`)
    Tuple of containing glyphs and error bars to be used (from :func:`ragavi.ragavi.errorbar`).

    """
    markers = ["circle", "diamond", "square", "triangle",
               "hex"]
    fmarker = "inverted_triangle"
    glyph_opts = {"size": 4,
                  "fill_alpha": 1,
                  "fill_color": color,
                  "line_color": "black",
                  # "nonselection_fill_color": "#7D7D7D",
                  # "nonselection_fill_alpha": 0.3,
                  "name": f"p_glyph_{yidx}"}

    # if there is any flagged data enforce fmarker where flag is active
    fmarkers = None
    if np.any(np.isnan(source.data[f"y{yidx}"])):
        fmarkers = gen_flag_data_markers(source.data[f"y{yidx}"], fid=fid,
                                         markers=markers, fmarker=fmarker)
        # update the data source with markers
        source.add(fmarkers, name="fmarkers")

        p_glyph = Scatter(x='x', y=f"y{yidx}", marker="fmarkers",
                          line_width=0, angle=0.7, **glyph_opts)
    else:
        p_glyph = Scatter(x='x', y=f"y{yidx}", marker=markers[fid],
                          line_width=0, **glyph_opts)

    # add a check for whether all the in y data were NaNs
    # this causes the errorbars to fail if they all are
    # they must be checked
    if np.all(np.isnan(source.data[f"y{yidx}"])):
        ebars = errorbar(x=source.data['x'], y=source.data[f"y{yidx}"],
                         color=color, yerr=None)
    else:
        ebars = errorbar(x=source.data['x'], y=source.data[f"y{yidx}"],
                         color=color, yerr=yerr)
        p_glyph.js_link("fill_alpha", ebars, "line_alpha")

    return p_glyph, ebars


################### Stats Related Funcs ################################

def create_stats_table(stats, yaxes):
    """ Create data table with median statistics
    Parameters
    ----------
    stats : :obj:`list`
        List of lists containing data stats for each iterations from :func:`ragavi.ragavi.stats_display`
    yaxes : :obj:`list`
        Contains y-axes for the current plot

    Returns
    -------
        Bokeh column layout containing data table with stats
    """
    # number of y-axes
    n_ys = len(yaxes)
    # number of fields, spws and corrs
    n_items = len(stats) // n_ys
    stats = np.array(stats)
    d_stats = dict(
        spw=stats[:n_items, 0],
        field=stats[:n_items, 1],
        corr=stats[:n_items, 2],
    )

    # get the data in a more useable format
    datas = stats[:, 3].reshape(-1, n_items).T
    for y in range(n_ys):
        d_stats[yaxes[y]] = datas[:, y]

    source = ColumnDataSource(data=d_stats)
    cols = "spw field corr".split() + yaxes

    columns = [TableColumn(field=x, title=x.capitalize()) for x in cols]
    dtab = DataTable(source=source, columns=columns,
                     fit_columns=True, height=150,
                     max_height=180, max_width=600,
                     sizing_mode="stretch_both", width=500)
    t_title = Div(text="Median Statistics")

    return column([t_title, dtab], sizing_mode="stretch_both")


def make_table_name(tab_name):
    """Create div for stats data table"""
    div = Div(text="Table: {}".format(tab_name))
    return div


def stats_display(tab_name, yaxis, gtype, corr, field, f_names=None,
                  flag=True, spwid=None):
    """Display some statistics on the plots.
    These statistics are derived from a specific correlation and a specified field of the data.

    Note
    ----
        Currently, only the medians of these plots are displayed.

    Parameters
    ----------
    corr : :obj:`int`
        Correlation number of the data to be displayed
    f_names : :obj:`list`
        List with all the available field names
    field : :obj:`int`
        Integer field id of the field being plotted. If a string name was provided, it will be converted within the main function by 
        :meth:`ragavi.vis_utils.name_2id`.
    flag : :obj:`bool`
        Whether to flag data or not
    gtype : :obj:`str`
        Type of gain table to be plotted.
    spwid : :obj:`int`
        Spectral window to be selected
    yaxis : :obj:`str`
        Can be amplitude, phase, real, imaginary or delay

    Returns
    -------
    List containing stats
    """
    subtable = get_table(tab_name, fid=field, spwid=str(spwid),
                         group_cols=[], where=[])[0]
    if subtable.row.size == 0:
        return None

    dobj = DataCoreProcessor(subtable, tab_name, gtype, yaxis,
                             corr=corr, flag=True, ddid=spwid)

    y = dobj.y_only(yaxis).y.compute()

    med_y = np.nanmedian(y)

    return [spwid, f_names[field], corr, med_y, yaxis]


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


################## Saving Plot funcs ###################################

def save_html(name, plot_layout):
    """Save plots in HTML format

    Parameters
    ----------
    hname : :obj:`str`
        HTML Output file name
    plot_layout : :obj:`bokeh.layouts`
                  Layout of the Bokeh plot, could be row, column, gridplot.
    """
    if "html" not in name:
        name = name + ".html"
    output_file(name)
    output = save(plot_layout, name, title=name)


def save_static_image(name, figs=None):
    """Save plots in SVG format

    Note
    ----
        The number of resulting images depends on `doplot`. By default, this means to objects unless otherwise.
        For SVG, the python package `selenium`, and node package `phantomjs` are required. More information `Exporting bokeh plots <https://bokeh.pydata.org/en/latest/docs/user_guide/export.html>`_

    Parameters
    ----------
    name : :obj:`str`
        Desired image name
    figs : :obj:`list`
         A list containing :obj:`bokeh.plotting.Plot` objects (The figures to be plotted.)

    """
    from selenium import webdriver
    driver = webdriver.Firefox()

    for _i in range(len(figs)):
        yaxis = figs[_i].yaxis.axis_label
        legs = figs[_i].legend

        for _l in legs:
            _l.visible = True

        for _r in figs[_i].renderers:
            _r.visible = True

        if "png" in name.lower():
            export_png(obj=figs[_i], filename=f"{name}_{yaxis}.{o_type}",
                       width=_PLOT_WIDTH_, height=_PLOT_HEIGHT_,
                       webdriver=driver)
        elif "svg" in name.lower():
            figs[_i].output_backend = "svg"
            export_svgs(figs[_i], filename=f"{name}_{yaxis}.{o_type}")


################### Main ###############################################
def main(**kwargs):
    """Main function that launches the gains plotter"""
    if "options" in kwargs:
        NB_RENDER = False
        # capture the parser options
        options = kwargs.get("options", None)

        corr = options.corr
        ddid = options.ddid
        doplot = options.doplot
        fields = options.fields
        gain_types = options.gain_types
        html_name = options.html_name
        image_name = options.image_name
        mytabs = options.mytabs
        plotants = options.plotants
        t0 = options.t0
        t1 = options.t1
        where = options.where
        kx = options.kx

    if isinstance(options.fields, list):
        fields = ",".join(fields)

    tables = [os.path.abspath(tab) for tab in options.mytabs]

    # ensure the length of gain type and table type are the same
    if len(options.gain_types) < len(tables):
        gains = gain_types * len(tables)
    else:
        gains = options.gain_types

    doplot = options.doplot

    # parent container of the items in this pot
    final_layout = []
    for tab, gain in zip(tables, gains):

        if doplot == "ap":
            y_axes = ["amplitude", "phase"]
        else:
            y_axes = ["real", "imaginary"]
        if gain == "K":
            y_axes = ["delay"]

        if options.fields:
            # Not using .isalnum coz the actual field names can be provided
            if (any(_ in fields for _ in ":~") or
                    fields.isnumeric()):
                fields = vu.resolve_ranges(fields)
            elif ',' in fields:
                """
                 convert field name to field id and join all the resulting field ids with a comma
                """
                fields = ",".join([str(vu.name_2id(tab, x))
                                   if not x.isnumeric() else x
                                   for x in fields.split(',')])
                fields = vu.resolve_ranges(fields)
            else:
                fields = str(vu.name_2id(tab, fields))
                fields = vu.resolve_ranges(fields)

        if options.corr:
            if ',' in options.corr:
                corrs = [int(_) for _ in options.corr.split(',')]
            else:
                corrs = [int(options.corr)]
            corrs = np.array(corrs)
        else:
            s_tab = table(tab, ack=False)
            corrs = np.arange(s_tab.getcell("FLAG", 0).shape[-1])
            s_tab.close()

        if options.ddid is not None:
            # TODO: Check for the input comming in from argparse
            # check if it needs to be resolved
            ddid = vu.resolve_ranges(options.ddid)
            # get all frequencies anyways
            freqs = vu.get_frequencies(tab).values / _GHZ_
        else:
            freqs = vu.get_frequencies(tab).values / _GHZ_

        if options.plotants:
            plotants = vu.resolve_ranges(options.plotants)

        init_time, end_time = get_time_range(tab, unix_time=False)

        if options.where is None:
            where = []
        else:
            where = [where]

        if options.mycmap is None:
            cmap = "coolwarm"
        else:
            cmap = options.mycmap

        # perform a time selections
        if t0:
            where.append(f"TIME - {str(init_time)} >= {str(t0)}")
        if t1:
            where.append(f"TIME - {str(init_time)} <= {str(t1)}")

        if gain in ["G", "F"] or (options.kx == "time" and gain == "K"):
            x_axis_type = "datetime"
        else:
            x_axis_type = "linear"

        all_figures = []

        # store generated plots
        all_glyphs = []

        # store legend items
        all_legends = []

        all_ebars = []
        # ebars_ax2 = []

        # Store stats
        all_stats = []

        # store flagged and unflaged data stources
        all_ufsources = []
        all_fsources = []

        logger.info("Acquiring table: {}".format(os.path.basename(tab)))
        subs = get_table(tab, spwid=ddid, where=where,
                         fid=fields, antenna=plotants)

        # confirm a populous table is selected
        try:
            assert len(subs) > 0
        except AssertionError:
            logger.info(
                "Table contains no data. Check data selection. Skipping.")
            continue

        spw_ids = np.unique(np.array([s.SPECTRAL_WINDOW_ID for s in subs]))

        # get antenna names
        ant_names = vu.get_antennas(tab).values
        ant_ids = np.unique(np.array([s.ANTENNA1 for s in subs]))

        field_names = vu.get_fields(tab).values
        field_ids = np.unique(np.array([s.FIELD_ID for s in subs]))

        field_ucodes = [u"\u2B24", u"\u25C6", u"\u25FC", u"\u25B2",
                        u"\u25BC", u"\u2B22"]

        # setting up colors for the antenna plot
        cmap = vu.get_linear_cmap(cmap, ant_ids.size)

        for _y, yaxis in enumerate(y_axes, start=1):
            fig_glyphs = []
            fig_renderers = []
            fig_legends = []
            fig_ebars = []
            fig_stats = []
            ufsources = []
            fsources = []

            iters = product(spw_ids, field_ids, corrs)

            for spw, fid, corr in iters:
                fname = field_names[fid]
                stats = stats_display(tab_name=tab, yaxis=yaxis, gtype=gain,
                                      corr=corr, field=fid, flag=_FLAG_DATA_,
                                      f_names=field_names, spwid=spw)
                if stats:
                    fig_stats.append(stats)

                logger.info(f"Spw: {spw}, Field: {fname}, Corr: {corr} {yaxis}")

                for _a, ant in enumerate(ant_ids):
                    legend = ant_names[ant]
                    colour = cmap[_a]
                    for sub in subs:
                        if (sub.SPECTRAL_WINDOW_ID == spw and
                                sub.FIELD_ID == fid and
                                sub.ANTENNA1 == ant):
                            # for tooltips
                            spw_id, scan = get_tooltip_data(sub, gain, freqs)
                            source = ColumnDataSource(
                                data={"scanid": scan,
                                      "corr": [corr] * scan.size,
                                      "field": [fname] * scan.size,
                                      "spw": spw_id,
                                      "antname": [legend] * scan.size
                                      })
                            inv_source = ColumnDataSource(data={})

                            logger.debug(f"Processing data for {legend}")

                            data_obj = DataCoreProcessor(
                                sub, tab, gain,
                                fid=fid, yaxis=yaxis, corr=corr,
                                flag=_FLAG_DATA_, kx=options.kx, ddid=spw)

                            data = data_obj.act()
                            xaxis = data_obj.xaxis

                            logger.debug(f"Getting the flagged data")
                            infl_data = DataCoreProcessor(
                                sub, tab, gain, fid=fid, yaxis=yaxis,
                                corr=corr, flag=not _FLAG_DATA_,
                                kx=options.kx, ddid=spw)

                            infl_data = infl_data.act()

                            x = data.x
                            x_label = data.x_label

                            y = data.y
                            y_err = data.y_err
                            y_label = data.y_label

                            iy = infl_data.y

                            logger.debug(f"Done")

                            source.add(x, name='x')
                            source.add(y, name=f"y{_y}")

                            inv_source.add(y, name=f"y{_y}")
                            inv_source.add(iy, name=f"iy{_y}")

                            ufsources.append(source)
                            fsources.append(inv_source)

                            glyphs, bars = make_plots(
                                source=source,
                                color=colour,
                                fid=np.where(field_ids == fid)[0][0],
                                yerr=y_err,
                                yidx=_y)

                            fig_glyphs.append(glyphs)
                            fig_legends.append((legend, []))
                            fig_ebars.append(bars)
                            sub.close()

            title = f"{yaxis.capitalize()} vs {xaxis.capitalize()}"
            fig = create_bk_fig(xlab=x_label, ylab=y_label, title=title,
                                x_min=x.min(), x_max=x.max(),
                                x_axis_type=x_axis_type, x_name=xaxis,
                                y_name=yaxis, pw=_PLOT_WIDTH_,
                                ph=_PLOT_HEIGHT_, add_grid=True,
                                add_title=True, add_xaxis=True,
                                add_yaxis=True, fix_plotsize=False,
                                ti_specs=dict(desired_num_ticks=8),
                                ax_specs=dict(minor_tick_line_alpha=1))

            h_tool = fig.select(name="p_htool")[0]

            tooltips = [("({:.4}, {:.4})".format(x_label, y_label),
                         f"(@x, @y{_y})"),
                        ("spw", "@spw"), ("scan_id", "@scanid"),
                        ("antenna", "@antname"),
                        ("corr", "@corr"), ("field", "@field")]

            if xaxis == "channel":
                for spw in spw_ids:
                    # do this here otherwise extra x added after title
                    title = fig.above.pop()
                    logger.debug("Adding extra chan/freq axis")
                    fig = add_axis(fig, (freqs[spw][0], freqs[spw][-1]),
                                   ax_label="Frequency [GHz]",
                                   ax_name=f"spw{spw}")
                    fig.add_layout(title, "above")
            elif xaxis == "time":
                munix = 3506716800.0
                ni_time = datetime.utcfromtimestamp(
                    init_time - munix).strftime("%d %b %Y, %H:%M:%S")
                ne_time = datetime.utcfromtimestamp(
                    end_time - munix).strftime("%d %b %Y, %H:%M:%S")

                fig.xaxis.axis_label = "{} from {} UTC".format(
                    x_label, ni_time)

                # format tootltip to time value
                h_tool.formatters = {"@x": "datetime"}

                tooltips[0] = ("({:.4}, {:.4})".format(x_label, y_label),
                               "(@x{%F %T}, $y)")

            if yaxis == "phase":
                 # Add degree sign to phase tick formatter
                fig.yaxis.formatter = PrintfTickFormatter(format=u"%f\u00b0")

            h_tool.tooltips = tooltips

            n_cds = len(ufsources)

            for _s in range(n_cds):
                p = fig.add_glyph(ufsources[_s], fig_glyphs[_s])

                if _s >= _BATCH_SIZE_:
                    p.visible = False

                fig_renderers.append(p)
                # form legend item
                fig_legends[_s][1].append(p)
                # add error bars
                if fig_ebars[_s]:
                    fig.add_layout(fig_ebars[_s])

            n_leg_objs = int(np.ceil(len(ant_ids) / _BATCH_SIZE_))

            leg_batches = create_legend_batches(n_leg_objs, fig_legends,
                                                batch_size=_BATCH_SIZE_)

            leg_objs = create_legend_objs(n_leg_objs, leg_batches)

            # make the last created legend object the bottom one
            for i in reversed(range(n_leg_objs)):
                fig.add_layout(leg_objs[f"leg_{str(i)}"], "above")
                all_legends.append(leg_objs[f"leg_{str(i)}"])

            all_glyphs.append(fig_glyphs)
            all_ebars.append(fig_ebars)
            all_fsources.append(fsources)
            all_ufsources.append(ufsources)
            all_stats.extend(fig_stats)
            all_figures.append(fig)

        stats_table = create_stats_table(all_stats, y_axes)

        # linking a bunch of stuff
        if len(all_figures) > 1:
            all_fsources, all_ufsources = link_plots(
                all_figures=all_figures, all_fsources=all_fsources,
                all_ebars=all_ebars)

        ######################################################################
        ################ Defining widgets ###################################
        ######################################################################

        # widget dimensions
        w_dims = dict(width=150, height=30)

        # Selection group
        # creating and configuring Antenna selection buttons
        ant_select = CheckboxGroup(labels=["Select all antennas"],
                                   active=[], **w_dims)

        ant_labs = gen_checkbox_labels(_BATCH_SIZE_, n_leg_objs, ant_names)
        batch_select = CheckboxGroup(labels=ant_labs, active=[0],
                                     **w_dims)

        corr_labs = ["Correlation {}".format(str(_)) for _ in corrs]
        corr_select = CheckboxGroup(labels=corr_labs, active=[0],
                                    width=150)

        field_labels = ["Field {} {}".format(
            field_names[int(x)],
            field_ucodes[enum_fid]) for enum_fid, x in enumerate(field_ids)]

        field_selector = CheckboxGroup(labels=field_labels, active=[0],
                                       **w_dims)

        spw_labs = ["Spw: {}".format(str(_)) for _ in spw_ids]
        spw_select = CheckboxGroup(labels=spw_labs, active=[0], width=150)

        # Additions group
        # Checkbox to hide and show legends
        legend_toggle = CheckboxGroup(labels=["Show legends"], active=[],
                                      **w_dims)

        save_selected = Button(label="Download data selection",
                               button_type="success", margin=(7, 5, 3, 5),
                               **w_dims)

        # configuring toggle button for showing all the errors
        toggle_err = CheckboxGroup(labels=["Show error bars"], active=[],
                                   **w_dims)

        toggle_flag = CheckboxGroup(labels=["Show flagged data"],
                                    active=[], **w_dims)

        # creating glyph size slider for the plots
        # margin = [top, right, bottom, left]
        size_slider = Slider(end=15, start=0.4, step=0.1,
                             value=4, title="Glyph size", margin=(3, 5, 7, 5),
                             bar_color="#6F95C3", ** w_dims)

        # Alpha slider for the glyphs
        alpha_slider = Slider(end=1, start=0.1, step=0.1, value=1,
                              margin=(3, 5, 7, 5), title="Glyph alpha",
                              bar_color="#6F95C3", **w_dims)

        # Plot layout options
        axis_fontslider = Slider(end=20, start=3, step=0.5, value=10,
                                 margin=(7, 5, 3, 5), title="Axis label size",
                                 bar_color="#6F95C3", **w_dims)
        title_fontslider = Slider(end=35, start=10, step=1, value=15,
                                  margin=(3, 5, 7, 5), title="Title size",
                                  bar_color="#6F95C3", **w_dims)

        tname_div = make_table_name(tab)

        ######################################################################
        ############## Defining widget Callbacks ############################
        ######################################################################

        ant_select.js_on_change("active", CustomJS(
            args=dict(ax=all_figures[0].renderers, bsel=batch_select,
                      fsel=field_selector, csel=corr_select, ssel=spw_select,
                      nbatches=n_leg_objs, nfields=field_ids.size,
                      ncorrs=corrs.size, nspws=spw_ids.size),
            code=ant_select_callback()))

        # BATCH SELECTION
        batch_select.js_on_change("active", CustomJS(
            args=dict(ax=all_figures[0].renderers, bsize=_BATCH_SIZE_,
                      nants=ant_ids.size, nbatches=n_leg_objs,
                      nfields=field_ids.size, ncorrs=corrs.size,
                      nspws=spw_ids.size, fsel=field_selector,
                      csel=corr_select, antsel=ant_select, ssel=spw_select),
            code=batch_select_callback()))

        corr_select.js_on_change("active", CustomJS(
            args=dict(bsel=batch_select, bsize=_BATCH_SIZE_,
                      fsel=field_selector, nants=ant_ids.size,
                      ncorrs=corrs.size, nfields=field_ids.size,
                      nbatches=n_leg_objs, nspws=spw_ids.size,
                      ax=all_figures[0].renderers, ssel=spw_select),
            code=corr_select_callback()))

        field_selector.js_on_change("active", CustomJS(
            args=dict(bsize=_BATCH_SIZE_, bsel=batch_select, csel=corr_select,
                      nants=ant_ids.size, nfields=field_ids.size,
                      ncorrs=corrs.size, nbatches=n_leg_objs,
                      nspws=spw_ids.size, ax=all_figures[0].renderers,
                      ssel=spw_select),
            code=field_selector_callback()))

        ex_ax = all_figures[0].select(type=LinearAxis, layout="above")
        ex_ax = sorted({_.id: _ for _ in ex_ax}.items())
        ex_ax = [_[1] for _ in ex_ax]

        spw_select.js_on_change("active", CustomJS(
            args=dict(bsel=batch_select, bsize=_BATCH_SIZE_, csel=corr_select,
                      fsel=field_selector, nants=ant_ids.size,
                      ncorrs=corrs.size, nfields=field_ids.size,
                      nbatches=n_leg_objs, nspws=spw_ids.size,
                      ax=all_figures[0].renderers,
                      ex_ax=ex_ax),
            code=spw_select_callback()))

        legend_toggle.js_on_change("active", CustomJS(
            args=dict(legs=all_legends),
            code=legend_toggle_callback()))

        toggle_err.js_on_change("active", CustomJS(
            args=dict(ax=all_figures[0].renderers,
                      ax_err=all_ebars[0]),
            code=toggle_err_callback()))

        toggle_flag.js_on_change("active", CustomJS(
            args=dict(f_sources=all_fsources,
                      uf_sources=all_ufsources),
            code=flag_callback()))

        save_selected.js_on_click(CustomJS(args=dict(
            uf_src=all_ufsources,
            f_src=all_fsources),
            code=save_selected_callback()))

        alpha_slider.js_on_change("value",
                                  CustomJS(args=dict(
                                      ax=all_glyphs[0]),
                                      code=alpha_slider_callback()))

        size_slider.js_on_change("value",
                                 CustomJS(args=dict(slide=size_slider,
                                                    ax=all_glyphs[0]),
                                          code=size_slider_callback()))

        axis_fontslider.js_on_change("value",
                                     CustomJS(args=dict(
                                         ax=all_figures[0].axis),
                                         code=axis_fs_callback()))
        title_fontslider.js_on_change(
            "value",
            CustomJS(args=dict(ax=all_figures[0].select(name="p_title")[0]),
                     code=title_fs_callback()))

        #################################################################
        ########## Define widget layouts #################################
        ##################################################################
        asel_div = Div(text="Select antenna group")
        fsel_div = Div(text="Fields")
        ssel_div = Div(text="Select spw")
        csel_div = Div(text="Select correlation")

        w_box = grid(
            children=[
                [ant_select, toggle_err, size_slider, title_fontslider],
                [legend_toggle, toggle_flag, alpha_slider, axis_fontslider],
                [asel_div, fsel_div, ssel_div, csel_div],
                [batch_select, field_selector, spw_select, corr_select]
            ])

        all_widgets = row([w_box, save_selected, stats_table],
                          sizing_mode=None, spacing=10, max_height=180)

        plots = gridplot([all_figures], toolbar_location="above",
                         sizing_mode="stretch_width")

        lay = layout([[tname_div], [all_widgets], [plots]],
                     sizing_mode="stretch_width")
        final_layout.append(lay)

        logger.info("Table {} done.".format(tab))

        if options.image_name:
            save_static_image(name=options.image_name, figs=all_figures)

    if _NB_RENDER_:
        return final_layout
    else:
        if not html_name:
            t_name = os.path.basename(tables[0])
            html_name = f"{t_name}_{doplot}"
            if len(tables) > 1:
                t_now = datetime.now().strftime("%Y%m%d_%H%M%S")
                html_name = html_name.replace(t_name, t_now)
        save_html(html_name, final_layout)

        logger.info("Rendered: {}.html".format(html_name))
        return 0


##################### Notebook Entry point #############################
def plot_table(**kwargs):
    """Plot gain tables within Jupyter notebooks. Parameter names correspond to the long names of each argument (i.e those with --) from the `ragavi-vis` command line help

    Parameters
    ----------
    **Required**
    gaintype: :obj:`str`, :obj:`list`
        Cal-table (list of caltypes) type to be plotted. Can be either 
        'B'-bandpass, 'D'- D jones leakages, G'-gains, 'K'-delay or 'F'-flux.
         Default is none
    table : :obj:`str` or :obj:`list` required
        The table (list of tables) to be plotted.

    **Optional**
    ant : :obj:`str`
        Plot only specific antennas, or comma-separated list of antennas.
    corr : :obj:`int, optional`
        Correlation index to plot. Can be a single integer or comma separated 
        integers e.g "0,2". Defaults to all.
    cmap : `str, optional`
        Matplotlib colour map to use for antennas. Default is coolwarm
    ddid : :obj:`int`
        SPECTRAL_WINDOW_ID or ddid number. Defaults to all
    doplot : :obj:`str, optional`
        Plot complex values as amp and phase (ap) or real and imag (ri). 
        Default is "ap".
    field : :obj:`str, optional`
        Field ID(s) / NAME(s) to plot. Can be specified as "0", "0,2,4", 
        "0~3" (inclusive range), "0:3" (exclusive range), "3:" (from 3 to 
        last) or using a field name or comma separated field names. Defaults 
        to all.
    k-xaxis: :obj:`str`
        Choose the x-xaxis for the K table. Valid choices are: time or 
        antenna. Defaults to time.
    taql: :obj:`str, optional`
        TAQL where clause
    t0  : :obj:`int, optional`
        Minimum time [in seconds] to plot. Default is full range
    t1 : :obj:`int, optional`
        Maximum time [in seconds] to plot. Default is full range

    """
    from ragavi.arguments import gains_argparser

    nargs = []
    for k, v in kwargs.items():
        nargs.append(f"--{k}")
        if isinstance(v, list):
            nargs.extend(v)
        else:
            nargs.append(v)
    parser = gains_argparser()
    options = parser.parse_args(nargs)

    # set NB rendering to true
    global _NB_RENDER_
    _NB_RENDER_ = True
    output_notebook()

    lays = main(options=options)

    for _l in lays:
        show(_l)
    logger.info("Done Plotting")

    return
