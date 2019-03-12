import sys
import glob
import re
import logging

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import xarrayms as xm
import xarray as xa
import dask.array as da
import numpy as np

from africanus.averaging import time_and_channel as ntc
from dask import compute, delayed
from pyrap.tables import table
from argparse import ArgumentParser
from datetime import datetime


from bokeh.plotting import figure
from bokeh.models.widgets import Div
from bokeh.layouts import row, column, gridplot, widgetbox
from bokeh.io import (output_file, show, output_notebook, export_svgs,
                      export_png, save)
from bokeh.models import (Range1d, HoverTool, ColumnDataSource, LinearAxis,
                          BasicTicker, Legend, Toggle, CustomJS, Title,
                          CheckboxGroup, Select, Text)


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
    pattern = re.compile(r'\.(G|K|B)\d*$', re.I)
    found = pattern.search(table_name)
    try:
        result = found.group()
        return result.upper()
    except AttributeError:
        return -1


def color_denormalize(ycol):
    """Converting rgb values from 0 to 255"""
    ycol = np.array(ycol)
    ycol = np.array(ycol * 255, dtype=int)
    ycol = ycol.tolist()
    ycol = tuple(ycol)[:-1]
    return ycol


def errorbar(fig, x, y, xerr=None, yerr=None, color='red', point_kwargs={}, error_kwargs={}):
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

    fig.legend.click_policy = 'hide'

    return h


def make_plots(source, ax1, ax2, color='purple', y1_err=None, y2_err=None):
    """Generate a plot

    Inputs
    ------

    source: ColumnDataSource
        Main data to plot
    ax1: figure
        First figure
    ax2: figure
        Second Figure
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
    x, y1, y2 = source.data['x'], source.data['y1'], source.data['y2']
    x, y1, y2 = compute(x, y1, y2)
    loopies = y1.shape[-1][:10]
    source.data['x'], source.data['y1'], source.data['y2'] = x, y1, y2

    for i in list(range(loopies)):

        p1 = ax1.circle(x=source.data['x'], y1=source.data['y1'][i],
                        size=8, alpha=1, color=color,
                        nonselection_color='#7D7D7D',
                        nonselection_fill_alpha=0.3)
        p2 = ax1.circle(x=source.data['x'], y2=source.data['y2'][i],
                        size=8, alpha=1, color=color,
                        nonselection_color='#7D7D7D',
                        nonselection_fill_alpha=0.3)

    return p1, p2


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
            # bax = [ [batch1], [batch2], [batch3] ]

            # j is batch number
            # i is glyph number
            j=0
            i=0

            if 0 in this.active
                i=0
                while i < bax1[j].length
                    bax1[0][i][1][0].visible = true
                    bax2[0][i][1][0].visible = true
                    i++
            else
                i=0
                while i < bax1[0].length
                    bax1[0][i][1][0].visible = false
                    bax2[0][i][1][0].visible = false
                    i++

            if 1 in this.active
                i=0
                while i < bax1[j].length
                    bax1[1][i][1][0].visible = true
                    bax2[1][i][1][0].visible = true
                    i++
            else
                i=0
                while i < bax1[0].length
                    bax1[1][i][1][0].visible = false
                    bax2[1][i][1][0].visible = false
                    i++

            if 2 in this.active
                i=0
                while i < bax1[j].length
                    bax1[2][i][1][0].visible = true
                    bax2[2][i][1][0].visible = true
                    i++
            else
                i=0
                while i < bax1[0].length
                    bax1[2][i][1][0].visible = false
                    bax2[2][i][1][0].visible = false
                    i++

            if 3 in this.active
                i=0
                while i < bax1[j].length
                    bax1[3][i][1][0].visible = true
                    bax2[3][i][1][0].visible = true
                    i++
            else
                i=0
                while i < bax1[0].length
                    bax1[3][i][1][0].visible = false
                    bax2[3][i][1][0].visible = false
                    i++



            if this.active.length == 4
                antsel.active = true
                antsel.label =  "Deselect all Antennas"
            else if this.active.length == 0
                antsel.active = false
                antsel.label = "Select all Antennas"
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

                if (this.value == "all"){
                    for(i=0; i<len; i++){
                        loax1[i].visible = true;
                        loax2[i].visible = true;
                        loax1_err[i].visible = true;
                        loax2_err[i].visible = true;

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

    j = 0
    for i in range(num_leg_objs):
        # in case the number is not a multiple of 16
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

    for i in range(num_leg_objs):
        lo_ax1['leg_%s' % str(i)] = Legend(items=bax1[i],
                                           location='top_right',
                                           click_policy='hide')
        lo_ax2['leg_%s' % str(i)] = Legend(items=bax2[i],
                                           location='top_right',
                                           click_policy='hide')
        loerr_ax1['leg_%s' % str(i)] = Legend(
            items=baerr_ax1[i],
            location='top_right',
            click_policy='hide',
            visible=False)
        loerr_ax2['leg_%s' % str(i)] = Legend(
            items=baerr_ax2[i],
            location='top_right',
            click_policy='hide',
            visible=False)

    return lo_ax1, loerr_ax1, lo_ax2, loerr_ax2


def gen_checkbox_labels(batch_size, num_leg_objs):
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

    labels = []
    s = 0
    e = batch_size - 1
    for i in range(num_leg_objs):
        labels.append("A%s - A%s" % (s, e))
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
                         major_label_orientation='horizontal', ticker=BasicTicker(desired_num_ticks=12))
    return linaxis


def name_2id(val, dic):
    """Translate field name to field id

    Inputs
    -----
    val: string
         Field ID name to convert
    dic: dict
         Dictionary containing enumerated source ID names

    Outputs
    -------
    key: int
         Integer field id
    """
    upperfy = lambda x: x.upper()
    values = dic.values()
    values = [upperfy(x) for x in values]
    val = val.upper()

    if val in values:
        val_index = values.index(val)
        keys = [x for x in dic.keys()]

        # get the key to that index from the key values
        key = keys[val_index]
        return int(key)
    else:
        return -1


def get_polarizations(ms_name):

    # Stokes types in this case are 1 based and NOT 0 based.
    stokes_types = ['I', 'Q', 'U', 'V', 'RR', 'RL', 'LR', 'LL', 'XX', 'XY', 'YX', 'YY', 'RX', 'RY', 'LX', 'LY', 'XR', 'XL', 'YR',
                    'YL', 'PP', 'PQ', 'QP', 'QQ', 'RCircular', 'LCircular', 'Linear', 'Ptotal', 'Plinear', 'PFtotal', 'PFlinear', 'Pangle']
    subname = "::".join((ms_name, 'POLARIZATION'))
    pol_subtable = list(xm.xds_from_table(subname))
    pol_subtable = pol_subtable[0]
    # ofsetting the acquired corr typeby one to match correctly the stokes type
    corr_types = pol_subtable.CORR_TYPE.sel(row=0).data.compute() - 1
    cor2stokes = []
    for typ in corr_types:
        cor2stokes.append(stokes_types[typ])
    return cor2stokes


def get_frequencies(ms_name, spwid=0):
    """Function to get channel frequencies from the SPECTRAL_WINDOW subtable.
    Inputs
    ------
    table_obj: pyrap table object
    spwid: int
           Spectral window id number. Defaults to 0

    Outputs
    -------
    freqs: xarray.core.dataarray.DataArray
           Channel centre frequencies for specified spectral window.
    """
    subname = "::".join((ms_name, 'SPECTRAL_WINDOW'))
    spw_subtab = list(xm.xds_from_table(subname, group_cols='__row__'))
    spw = spw_subtab[spwid]
    freqs = spw.CHAN_FREQ
    # spw_subtab('close')
    return freqs


def get_antennas(ms_name):
    """Function to get antennae names from the ANTENNA subtable.
    Inputs
    ------
    table_obj: pyrap table object

    Outputs
    -------
    ant_names: xarray.core.dataarray.DataArray
               Names for all the antennas available.

    """
    subname = "::".join((ms_name, 'ANTENNA'))
    ant_subtab = list(xm.xds_from_table(subname))
    ant_subtab = ant_subtab[0]
    ant_names = ant_subtab.NAME
    # ant_subtab('close')
    return ant_names


def get_flags(xds_table_obj, corr=None):
    """Function to get Flag values from the FLAG column
    Allows the selection of flags for a single correlation. If none is specified the entire data is then selected.
    Inputs
    ------
    table_obj: pyrap table object
    corr: int
          Correlation number to select.

    Outputs
    -------
    flags: xarray.core.dataarray.DataArray
           Array containing selected flag values.

    """
    flags = xds_table_obj.FLAG
    if corr is None:
        return flags
    else:
        flags = flags.sel(corr=corr)
    return flags


def get_errors(xds_table_obj, corr=None):
    # CoONFIRM IF ITWORKS ACTUALLY
    #-------------------------------------------
    """Function to get error data from PARAMERR column.
    Inputs
    ------
    table_obj: pyrap table object.

    Outputs
    errors: ndarray
            Error data.
    """
    errors = xds_table_obj.PARAMERR
    return errors


def get_fields(ms_name):
    """Function to get field names from the FIELD subtable.
    Inputs
    ------
    table_obj: pyrap table object

    Outputs
    -------
    field_names: xarray.core.dataarray.DataArray
                 String names for the available data in the table
    """
    subname = "::".join((ms_name, 'FIELD'))
    field_subtab = list(xm.xds_from_table(subname, ack=False))
    field_subtab = field_subtab[0]
    field_names = field_subtab.NAME
    return field_names


def get_tooltip_data(xds_table_obj, ms_name, xaxis):
    """Function to get the data to be displayed on the mouse tooltip on the plots.
    Inputs
    ------
    table_obj: pyrap table object
    gtype: str
           Type of gain table being plotted

    Outputs
    -------
    spw_id: ndarray
            Spectral window ids
    scan_no: ndarray
             scan ids
    ttip_antnames: array
                   Antenna names to show up on the tooltips


    """
    spw_id = xds_table_obj.DATA_DESC_ID
    scan_no = xds_table_obj.SCAN_NUMBER.data.compute()
    ant1_id = xds_table_obj.ANTENNA1.data.compute()
    ant2_id = xds_table_obj.ANTENNA2.data.compute()

    # MAYBE  CHANGE THIS TOOLTIP TO baseline!!!!!!!!!
    antnames = get_antennas(ms_name)
    # get available antenna names from antenna id
    ttip_antnames = np.asarray(['({}, {})'.format(
        antnames[x], antnames[y]) for x, y in zip(ant1_id, ant2_id)])

    freqs = get_frequencies(ms_name)
    nchan = len(freqs)

    if xaxis == 'channel' or xaxis == 'frequency':
        """spw_id = spw_id.reshape(spw_id.size, 1)
                                spw_id = spw_id.repeat(nchan, axis=1)
                                scan_no = scan_no.reshape(scan_no.size, 1)
                                scan_no = scan_no.repeat(nchan, axis=1)
                                ant1_id = ant1_id.reshape(ant1_id.size, 1)
                                ant1_id = ant1_id.repeat(nchan, axis=1)
                                ant2_id = ant2_id.reshape(ant2_id.size, 1)
                                ant2_id = ant2_id.repeat(nchan, axis=1)"""
        spw_id = spw_id.reshape(1, spw_id.size)
        spw_id = spw_id.repeat(nchan, axis=0)
        scan_no = scan_no.reshape(1, scan_no.size)
        scan_no = scan_no.repeat(nchan, axis=0)
        ant1_id = ant1_id.reshape(1, ant1_id.size)
        ant1_id = ant1_id.repeat(nchan, axis=0)
        ant2_id = ant2_id.reshape(1, ant2_id.size)
        ant2_id = ant2_id.repeat(nchan, axis=0)
        ttip_antnames = ttip_antnames.reshape(1, ttip_antnames.size)
        ttip_antnames = ttip_antnames.repeat(nchan, axis=0)

    return spw_id, scan_no, ttip_antnames


def calc_uvdist(uvw):
    """ Function to Calculate uv distance in metres
    Inputs
    ------
    uvw: xarray.core.dataarray.DataArray

    Outputs
    -------
    uvdist: xarray.core.dataarray.DataArray
            uv distance in meters
    """
    u = uvw.isel(**{'(u,v,w)': 0})
    v = uvw.isel(**{'(u,v,w)': 1})
    uvdist = xa.ufuncs.sqrt(xa.ufuncs.square(u) + xa.ufuncs.square(v))
    return uvdist


def calc_uvwave(uvw, freq):
    """
        Calculate uv distance in wavelength for availed frequency.
        This function also calculates the corresponding wavelength.

    Inputs
    ------
    uvw: xarray.core.dataarray.DataArray
    freq: float
          Frequency from which corresponding wavelength will be obtained.

    Outputs
    -------
    uvwave: xarray.core.dataarray.DataArray
            uv distance in wavelength for specific frequency
    """

    # speed of light
    C = 3e8

    # wavelength = velocity / frequency
    wavelength = (C / freq)
    uvdist = calc_uvdist(uvw)

    uvwave = uvdist / wavelength
    return uvwave


def get_xaxis_data(xds_table_obj, ms_name, xaxis):
    """Function to get x-axis data. It is dependent on the gaintype.
        This function also returns the relevant x-axis labels for both pairs of plots.
    Inputs
    ------
    table_obj: pyrap table object

    gtype:  str
            Type of gain table being plotted.

    Outputs
    -------
    xdata: ndarray
           X-axis data depending on the gain table to be plotted.
    xaxis_label: str
                 Label to appear on the x-axis of the plots. This is shared amongst both plots.
    """
    if xaxis == 'antenna1':
        xdata = xds_table_obj.ANTENNA1
        xaxis_label = 'Antenna1'
    elif xaxis == 'antenna2':
        xdata = xds_table_obj.ANTENNA2
        xaxis_label = 'Antenna2'
    elif xaxis == 'frequency' or xaxis == 'channel':
        xdata = get_frequencies(ms_name)
        xaxis_label = 'Channel'
    elif xaxis == 'scan':
        xdata = xds_table_obj.SCAN_NUMBER
        xaxis_label = 'Scan'
    elif xaxis == 'time':
        xdata = xds_table_obj.TIME
        xaxis_label = 'Time [s]'
    elif xaxis == 'uvdistance':
        xdata = xds_table_obj.UVW
        xaxis_label = 'UV Distance [m]'
    elif xaxis == 'uvwave':
        xdata = xds_table_obj.UVW
        xaxis_label = 'UV Wave [lambda]'
    else:
        print("Invalid xaxis name")
        return

    return xdata, xaxis_label


def prep_xaxis_data(xdata, xaxis, freq=None):
    """Function to Prepare the x-axis data.
    Inputs
    ------
    xdata: 1-D array
           Data for the xaxis to be prepared
    gtype: str
           gain type of the table to plot
    ptype: str
           Type of plot, whether ap or ri

    freq: float
          REQUIRED ONLY when xaxis specified is 'uvwave'. In this case.
          this function must be the called within a loop containing all
          the frequencies available in  a spectral window.

    Outputs
    -------
    prepdx: 1-D array
            Data for the x-axid of the plots
    """
    if xaxis == 'channel':
        prepdx = delayed(xdata.chan.data)
    elif xaxis == 'frequency':
        prepdx = xdata.data
    elif xaxis == 'time':
        prepdx = (xdata - xdata[0]).data
    elif xaxis == 'uvdistance':
        prepdx = calc_uvdist(xdata).data
    elif xaxis == 'uvwave':
        prepdx = calc_uvwave(xdata, freq).data
    elif xaxis == 'antenna1' or xaxis == 'antenna2' or xaxis == 'scan':
        prepdx = xdata.data
    return prepdx


def get_yaxis_data(xds_table_obj, ms_name, ptype):
    """ Function to extract the required column for the y-axis data.
    This column is determined by ptype which can be amplitude vs phase 'ap'
    or real vs imaginary 'ri'.

    Inputs
    -----
    table_obj: python casacore table object
               Table in which to get the data

    gtype: str
           Gain table type B, F, G or K.

    ptype: str
           Plot type ap / ri

    Outputs
    -------
    Returns np.ndarray dataa as well as the y-axis labels (str) for both plots.
    """
    if ptype == 'ap':
        y1_label = 'Amplitude'
        y2_label = 'Phase[deg]'
    else:
        y1_label = 'Real'
        y2_label = 'Imaginary'

    ydata = xds_table_obj.DATA
    return ydata, y1_label, y2_label


def prep_yaxis_data(xds_table_obj, ms_name, ydata, ptype='ap', corr=0, flag=True):
    """Function to process data for the y-axis. Part of the processing includes:
    - Selecting correlation for the data and error
    - Flagging
    - Complex correlation parameter conversion to amplitude, phase, real and
      imaginary for processing
    Data selection and flagging are done by this function itself, however ap and ri conversion are done by specified functions.

    Inputs
    ------
    table_obj: pyrap table object
               table object for an already open table
    ydata: ndarray
           Relevant y-axis data to be processed
    gtype: str
           Gain table type  B, F, G or K.
    ptype: str
           Plot type 'ap' / 'ri'
    corr: int
          Correlation number to select
    flag: bool
          Option on whether to flag the data or not

    Outputs
    -------
    y1: masked ndarray
        Amplitude / real part of the complex input data.
    y1_err: masked ndarray
        Error data for y1.
    y2: masked ndarray
        Phase angle / Imaginary part of input data.
    y2_err: masked ndarray
        Error data for y2.

    """
    # select data correlation for both the data and the errors
    ydata = ydata.sel(corr=corr)
    # ydata_errors = get_errors(ms_name).sel(corr=corr)
    # ydata_errors = get_errors(table_obj)[:, :, corr]

    if flag:
        flags = get_flags(xds_table_obj).sel(corr=corr)
        ydata = da.ma.masked_array(data=ydata, mask=flags)
        # ydata_errors = da.ma.masked_array(data=ydata_errors, mask=flags)

    y1, y2 = process_data(ydata, ptype=ptype)

    return y1, y2


def get_phase(ydata, unwrap=True):
    phase = xa.ufuncs.angle(ydata, deg=True)
    if unwrap:
        # delay dispatching of unwrapped phase
        phase = delayed(np.unwrap)(phase)
    return phase


def get_amplitude(ydata):
    amplitude = da.absolute(ydata)
    if hasattr(amplitude, 'data'):
        amplitude = amplitude.data
    return amplitude


def get_real(ydata):
    real = ydata.real
    if hasattr(real, 'data'):
        real = real.data
    return real


def get_imaginary(ydata):
    imag = ydata.imag
    if hasattr(imag, 'data'):
        imag = imag.data
    return imag


def process_data(ydata, ptype):
    if ptype == 'ap':
        y1 = get_amplitude(ydata)
        y2 = get_phase(ydata)
    else:
        y1 = get_real(ydata)
        y2 = get_imaginary(ydata)

    return y1,  y2


def blackbox(xds_table_obj, ms_name, xaxis, ptype, corr, showFlagged=True, ititle=None):
    x_data, xlabel = get_xaxis_data(xds_table_obj, ms_name, xaxis)
    y_data, y1label, y2label = get_yaxis_data(xds_table_obj, ms_name, ptype)
    y1_prepd, y2_prepd = prep_yaxis_data(xds_table_obj, ms_name, y_data,
                                         ptype=ptype, corr=corr,
                                         flag=showFlagged)
    if xaxis == 'channel' or xaxis == 'frequency':
        y1_prepd = y1_prepd.transpose()
        y2_prepd = y2_prepd.transpose()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(40, 20))

    # plt.tight_layout()
    if xaxis == 'uvwave':
        freqs = get_frequencies(ms_name).compute()
        x_prepd = prep_xaxis_data(x_data, xaxis, freq=freqs)
    else:
        x_prepd = prep_xaxis_data(x_data, xaxis)
        # for bokeh
        # for mpl
    f1, f2 = plotter(ax1, ax2, x_prepd, y1_prepd, y2_prepd)

    f1.set_xlabel(xlabel)
    f2.set_xlabel(xlabel)

    f1.set_ylabel(y1label)
    f2.set_ylabel(y2label)

    if ititle is not None:
        f1.set_title("{}: {} vs {}".format(ititle, y1label, xlabel))
        f2.set_title("{}: {} vs {}".format(ititle, y2label, xlabel))
        plt.savefig(fname="{}_{}_{}".format(ititle, ptype, xaxis))
    else:
        f1.set_title("{} vs {}".format(y1label, xlabel))
        f2.set_title("{} vs {}".format(y2label, xlabel))
        plt.savefig(fname="{}_{}".format(ptype, xaxis))
    # plt.clf()
    plt.close('all')


def plotter(ax1, ax2, x, y1, y2):
    x, y1, y2 = compute(x, y1, y2)
    ax1.plot(x, y1, marker='o', markersize=0.5, linewidth=0)
    ax2.plot(x, y2, marker='o', markersize=0.5, linewidth=0)

    return ax1, ax2


def iter_scan(ms_name, xaxis, ptype, corr, showFlagged=True):
    scans = list(xm.xds_from_ms(ms_name, group_cols='SCAN_NUMBER'))
    for scan in scans:
        title = "Scan_{}".format(scan.SCAN_NUMBER)
        blackbox(scan, ms_name, xaxis, ptype, corr, showFlagged=True,
                 ititle=title)


def iter_specwin(ms_name, xaxis, ptype, corr, showFlagged=True):
    specwins = list(xm.xds_from_ms(ms_name, group_cols='DATA_DESC_ID'))
    for spw in specwins:
        title = "SPW_{}".format(spw.DATA_DESC_ID)
        blackbox(spw, ms_name, xaxis, ptype, corr, showFlagged=True,
                 ititle=title)


def iter_correlation(ms_name, xaxis, ptype):
    # data selection will be done over each spectral window
    specwins = list(xm.xds_from_ms(ms_name, group_cols='DATA_DESC_ID'))
    corr_names = get_polarizations(ms_name)
    for spw in specwins:
        for corr in spw.corr.data:
            title = "Corr_{}".format(corr_names[corr])
            blackbox(spw, ms_name, xaxis, ptype, corr=corr, showFlagged=True,
                     ititle=title)


def stats_display(table_obj, gtype, ptype, corr, field):
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
    subtable = table_obj.query(query="FIELD_ID=={}".format(field))
    ydata, y1label, y2label = get_yaxis_data(subtable, gtype, ptype)

    flags = get_flags(subtable)[:, :, corr]
    ydata = ydata[:, :, corr]
    m_ydata = np.ma.masked_array(data=ydata, mask=flags)

    if ptype == 'ap':
        y1 = np.ma.abs(m_ydata)
        y2 = np.unwrap(np.ma.angle(m_ydata, deg=True))
        med_y1 = np.ma.median(y1)
        med_y2 = np.ma.median(y2)
        text = "Median Amplitude: {}\nMedian Phase: {} deg".format(
            med_y1, med_y2)
        if gtype == 'K':
            text = "Median Amplitude: {}".format(med_y1)
    else:
        y1 = np.ma.real(m_ydata)
        y2 = np.ma.imag(m_ydata)

        med_y1 = np.ma.median(y1)
        med_y2 = np.ma.median(y2)
        text = "Median Real: {}\nMedian Imaginary: {}".format(med_y1, med_y2)

    pre = PreText(text=text)

    return pre


def stringify(inp):
    """Function to convert multiple strings in a list into a single string.
    Inputs
    ------
    inp: iterable sequence
         sequence of strings

    Outputs
    -------
    text: str
          A single string combining all the strings in the input sequence.
    """
    text = ''
    for item in inp:
        text = text + item
    return text


def get_argparser():
    """Get argument parser"""
    parser = ArgumentParser(usage='prog [options] <value>')
    parser.add_argument('-a', '--ant', dest='plotants', type=str,
                        help='Plot only this antenna, or comma-separated list\
                              of antennas',
                        default=[-1])
    parser.add_argument('-c', '--corr', dest='corr', type=int,
                        help='Correlation index to plot (usually just 0 or 1,\
                              default = 0)',
                        default=0)
    parser.add_argument('--cmap', dest='mycmap', type=str,
                        help='Matplotlib colour map to use for antennas\
                             (default=coolwarm)',
                        default='coolwarm')
    parser.add_argument('-d', '--doplot', dest='doplot', type=str,
                        help='Plot complex values as amp and phase (ap)'
                        'or real and imag (ri) (default = ap)', default='ap')
    parser.add_argument('-f', '--field', dest='fields', nargs='*', type=str,
                        help='Field ID(s) / NAME(s) to plot', default=None)
    parser.add_argument('--htmlname', dest='html_name', type=str,
                        help='Output HTMLfile name', default='')
    parser.add_argument('-p', '--plotname', dest='image_name', type=str,
                        help='Output image name', default='')
    parser.add_argument('-t', '--table', dest='mytabs',
                        nargs='*', type=str,
                        help='Table(s) to plot (default = None)', default=[])
    parser.add_argument('--t0', dest='t0', type=float,
                        help='Minimum time to plot (default = full range)',
                        default=-1)
    parser.add_argument('--t1', dest='t1', type=float,
                        help='Maximum time to plot (default = full range)',
                        default=-1)
    parser.add_argument('--yu0', dest='yu0', type=float,
                        help='Minimum y-value to plot for upper panel (default=full range)',
                        default=-1)
    parser.add_argument('--yu1', dest='yu1', type=float,
                        help='Maximum y-value to plot for upper panel (default=full range)',
                        default=-1)
    parser.add_argument('--yl0', dest='yl0', type=float,
                        help='Minimum y-value to plot for lower panel (default=full range)',
                        default=-1)
    parser.add_argument('--yl1', dest='yl1', type=float,
                        help='Maximum y-value to plot for lower panel (default=full range)',
                        default=-1)
    parser.add_argument('--xaxis', dest='xaxis', type=str,
                        help='x-axis to plot', default='time')

    parser.add_argument('--iterate', dest='iterate', type=str,
                        choices=['scan', 'corr', 'spw'],
                        help='Select which variable to iterate over \
                              (defaults to none)',
                        default=None)
    parser.add_argument('--timebin', dest='timebin', type=str,
                        help='Number of timestamsp in each bin')
    parser.add_argument('--chanbin', dest='chanbin', type=str,
                        help='Number of channels in each bin')

    return parser


def config_logger():
    logfile_name = 'ragavi.log'
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ragavi_logger = logging.getLogger('ragavi')
    ragavi_logger.setLevel(logging.ERROR)
    rfh = logging.FileHandler(logfile_name)
    rfh.setLevel(logging.DEBUG)
    rfh.setFormatter(formatter)
    ragavi_logger.addHandler(rfh)

    xm_logger = logging.getLogger('xarrayms')
    xm_logger.setLevel(logging.ERROR)

    fh = logging.FileHandler('vis.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    xm_logger.addHandler(fh)


def main(**kwargs):
    # disable all warnings
    # logging.disable(logging.WARNING)
    config_logger()

    if len(kwargs) == 0:
        NB_RENDER = False

        parser = get_argparser()
        options = parser.parse_args()

        corr = int(options.corr)
        doplot = options.doplot
        field_ids = options.fields
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
        xaxis = options.xaxis
        iterate = options.iterate
        timebin = options.timebin
        chanbin = options.chanbin

    else:
        NB_RENDER = True

        field_ids = kwargs.get('fields', [])
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

    if len(mytabs) > 0:
        mytabs = [x.rstrip("/") for x in mytabs]
    else:
        logging.info('ragavi exited: No gain table specified.')
        sys.exit(-1)

    if len(field_ids) == 0:
        loggging.info('ragavi exited: No field id specified.')
        sys.exit(-1)

    for mytab, field in zip(mytabs, field_ids):

        field_names = get_fields(mytab).data.compute()
        # get id: field_name pairs
        field_ids = dict(enumerate(field_names))
        if field.isdigit():
            field = int(field)
        else:
            field = name_2id(field, field_ids)

        # by default data is grouped in field ids and data description
        partitions = list(xm.xds_from_ms(mytab))

        for chunk in partitions:
            # only plot specified field
            if chunk.FIELD_ID == field:
                if iterate is None:
                    blackbox(chunk, mytab, xaxis, doplot, corr,
                             showFlagged=True, ititle=None)
                else:
                    if iterate == 'scan':
                        iter_scan(mytab, xaxis, doplot, corr,
                                  showFlagged=True)
                    elif iterate == 'corr':
                        iter_correlation(mytab, xaxis, doplot)
                    elif iterate == 'spw':
                        iter_specwin(mytab, xaxis, doplot, corr,
                                     showFlagged=True)


# for demo
if __name__ == '__main__':
    main()
