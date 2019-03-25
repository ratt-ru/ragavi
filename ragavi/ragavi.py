import sys
import glob
import numpy as np
import re
import logging
import warnings

from datetime import datetime
import matplotlib.cm as cmx
import matplotlib.pylab as pylab
from pyrap.tables import table
from argparse import ArgumentParser
import matplotlib.colors as colors

from bokeh.plotting import figure
from bokeh.models.widgets import Div, PreText
from bokeh.layouts import row, column, gridplot, widgetbox
from bokeh.io import (output_file, show, output_notebook, export_svgs,
                      export_png, save)
from bokeh.models import (Range1d, HoverTool, ColumnDataSource, LinearAxis,
                          BasicTicker, Legend, Toggle, CustomJS, Title,
                          CheckboxGroup, Select, Text, Slider)


PLOT_WIDTH = 700
PLOT_HEIGHT = 600
GAIN_TYPES = ['B', 'F', 'G', 'K']
GHZ = 1e9


def config_logger():
    """This function is used to configure the logger for ragavi and catch
        all warnings output by sys.stdout.
    """
    logfile_name = 'ragavi.log'
    # capture only a single instance of a matching repeated warning
    warnings.filterwarnings('default')

    # setting the format for the logging messages
    start = " (O_o) ".center(80, "=")
    form = '{}\n%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    form = form.format(start)
    formatter = logging.Formatter(form, datefmt='%d.%m.%Y@%H:%M:%S')

    # setup for ragavi logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # capture all stdout warnings
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger('py.warnings')
    warnings_logger.setLevel(logging.DEBUG)

    # setup for logfile handing ragavi
    fh = logging.FileHandler(logfile_name)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    warnings_logger.addHandler(logger)
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
    p1 = ax1.circle('x', 'y1', size=4, alpha=1, color=color, source=source,
                    nonselection_color='#7D7D7D', nonselection_fill_alpha=0.3)
    p1_err = errorbar(fig=ax1, x=source.data['x'], y=source.data['y1'],
                      color=color, yerr=y1_err)

    p2 = ax2.circle('x', 'y2', size=4, alpha=1, color=color, source=source,
                    nonselection_color='#7D7D7D', nonselection_fill_alpha=0.3)
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


def size_slider_callback():
    """JS callback to sel
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
                                           click_policy='hide',
                                           visible=False)
        lo_ax2['leg_%s' % str(i)] = Legend(items=bax2[i],
                                           location='top_right',
                                           click_policy='hide',
                                           visible=False)
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
    values = map(upperfy, values)
    val = val.upper()

    if val in values:
        val_index = values.index(val)
        keys = dic.keys()

        # get the key to that index from the key values
        key = keys[val_index]
        return int(key)
    else:
        return -1


def data_prep_G(masked_data, masked_data_err, doplot):
    """Preparing the data for plotting gain cal-table

    Inputs
    ------
    masked_data: numpy.ndarray
        Flagged data from CPARAM column to be plotted.
    masked_data_err : numpy.ndarray
        Flagged data from the PARAMERR column to be plotted
    doplot: str
        Either 'ap' or 'ri'


    Outputs
    -------
    (y1_data_array, y1_error_data_array, y2_data_array, y2_error_data_array) : tuple
        Tuple with arrays of the different data

    """

    if doplot == 'ap':
        y1 = np.ma.abs(masked_data)
        y1_err = np.ma.abs(masked_data_err)
        y2 = np.ma.angle(masked_data, deg=True)
        # Remove phase limit from -pi to pi
        #y2 = np.unwrap(y2)
        y2_err = None
    else:
        y1 = np.real(masked_data)
        y1_err = np.ma.abs(masked_data_err)
        y2 = np.imag(masked_data)
        y2_err = None

    return y1, y1_err, y2, y2_err


def data_prep_B(masked_data, masked_data_err, doplot):
    """Preparing the data for plotting bandpass cal-table

    Inputs
    ------
    masked_data     : numpy.ndarray
        Flagged data from CPARAM column to be plotted.
    masked_data_err : numpy.ndarray
        Flagged data from the PARAMERR column to be plotted
    doplot: str
        Either 'ap' or 'ri'

    Outputs
    -------
    (y1_data_array, y1_error_data_array, y2_data_array, y2_error_data_array): tuple
        Tuple with arrays of the different data

    """
    if doplot == 'ap':
        y1 = np.ma.abs(masked_data)
        y1_err = np.ma.abs(masked_data_err)
        y2 = np.ma.angle(masked_data, deg=True)
        #y2 = np.unwrap(y2)
        y2_err = np.ma.angle(masked_data_err, deg=True)
        y2_err = np.unwrap(y2_err)
    else:
        y1 = np.real(masked_data)
        y1_err = np.abs(masked_data_err)
        y2 = np.imag(masked_data)
        y2_err = None

    return y1, y1_err, y2, y2_err


def data_prep_K(masked_data, masked_data_err, doplot):
    """Preparing the data for plotting delay cal-table. Doplot must be 'ap'.

    Inputs
    ------
    masked_data: numpy.ndarray
        Flagged data from CPARAM column to be plotted.
    masked_data_err: numpy.ndarray
        Flagged data from the PARAMERR column to be plotted

    Outputs
    -------
    (y1_data_array, y1_error_data_array, y2_data_array, y2_error_data_array): tuple
        Tuple with arrays of the different data

    """
    y1 = masked_data
    y1_err = masked_data_err

    # quick fix to hide plot without generating errors
    y2 = y1
    y2_err = y1

    return y1, y1_err, y2, y2_err


def data_prep_F(masked_data, masked_data_err, doplot):
    """Preparing the data for plotting flux cal table

    Inputs
    ------
    masked_data: numpy.ndarray
        Flagged data from CPARAM column to be plotted.
    masked_data_err : numpy.ndarray
        Flagged data from the PARAMERR column to be plotted
    doplot: str
        Either 'ap' or 'ri'

    Outputs
    -------
    (y1_data_array, y1_error_data_array, y2_data_array, y2_error_data_array) : tuple
        Tuple with arrays of the different data

    """

    if doplot == 'ap':
        y1 = np.ma.abs(masked_data)
        y1_err = np.ma.abs(masked_data_err)
        y2 = np.ma.angle(masked_data, deg=True)
        # Remove phase limit from -pi to pi
        #y2 = np.unwrap(y2)
        y2_err = None
    else:
        y1 = np.real(masked_data)
        y1_err = np.ma.abs(masked_data_err)
        y2 = np.imag(masked_data)
        y2_err = None
    return y1, y1_err, y2, y2_err


def get_yaxis_data(table_obj, gtype, ptype):
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
    Returns np.ndarray data as well as the y-axis labels (str) for both plots.
    """
    if ptype == 'ap':
        y1_label = 'Amplitude'
        y2_label = 'Phase[deg]'
    else:
        y1_label = 'Real'
        y2_label = 'Imaginary'

    if gtype == 'K':
        data_column = 'FPARAM'
    else:
        data_column = 'CPARAM'

    ydata = table_obj.getcol(data_column)
    return ydata, y1_label, y2_label


def prep_yaxis_data(table_obj, ydata, gtype, ptype='ap', corr=0, flag=True):
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
    ydata = ydata[:, :, corr]
    ydata_errors = get_errors(table_obj)[:, :, corr]

    if flag:
        flags = get_flags(table_obj, corr)
        ydata = np.ma.masked_array(data=ydata, mask=flags)
        ydata_errors = np.ma.masked_array(data=ydata_errors, mask=flags)

    if gtype == 'B':
        y1, y1_err, y2, y2_err = data_prep_B(ydata, ydata_errors, ptype)
    elif gtype == 'F':
        ydata = ydata[:, 0]
        ydata_errors = ydata_errors[:, 0]
        y1, y1_err, y2, y2_err = data_prep_F(ydata, ydata_errors, ptype)
    elif gtype == 'G':
        ydata = ydata[:, 0]
        ydata_errors = ydata_errors[:, 0]
        y1, y1_err, y2, y2_err = data_prep_G(ydata, ydata_errors, ptype)
    elif gtype == 'K':
        ydata = ydata[:, 0]
        ydata_errors = ydata_errors[:, 0]
        y1, y1_err, y2, y2_err = data_prep_K(ydata, ydata_errors, ptype)

    return y1, y1_err, y2, y2_err


def get_frequencies(table_obj):
    """Function to get channel frequencies from the SPECTRAL_WINDOW subtable.
    Inputs
    ------
    table_obj: pyrap table object

    Outputs
    -------
    freqs: 1D-array
           Channel centre frequencies.
    """
    spw_subtab = table(table_obj.getkeyword('SPECTRAL_WINDOW'), ack=False)
    freqs = spw_subtab.getcell('CHAN_FREQ', 0)
    spw_subtab.close()
    return freqs


def get_antennas(table_obj):
    """Function to get antennae names from the ANTENNA subtable.
    Inputs
    ------
    table_obj: pyrap table object

    Outputs
    -------
    ant_names: 1D-array
               Names for all the antennas available.

    """
    ant_subtab = table(table_obj.getkeyword('ANTENNA'), ack=False)
    ant_names = ant_subtab.getcol('NAME')
    ant_subtab.close()
    return ant_names


def get_flags(table_obj, corr=None):
    """Function to get Flag values from the FLAG column
    Allows the selection of flags for a single correlation. If none is specified the entire data is then selected.
    Inputs
    ------
    table_obj: pyrap table object
    corr: int
          Correlation number to select.

    Outputs
    -------
    flags: ndarray
           Array containing selected flag values.

    """
    flags = table_obj.getcol('FLAG')
    if corr is None:
        return flags
    else:
        flags = flags[:, :, corr]
    return flags


def get_errors(table_obj):
    """Function to get error data from PARAMERR column.
    Inputs
    ------
    table_obj: pyrap table object.

    Outputs
    errors: ndarray
            Error data. 
    """
    errors = table_obj.getcol('PARAMERR')
    return errors


def get_fields(table_obj):
    """Function to get field names from the FIELD subtable.
    Inputs
    ------
    table_obj: pyrap table object

    Outputs
    -------
    field_names: 1-D array
                 String names for the available data in the table
    """
    field_subtab = table(table_obj.getkeyword('FIELD'), ack=False)
    field_names = field_subtab.getcol('NAME')
    field_subtab.close()
    return field_names


def get_tooltip_data(table_obj, gtype):
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
    spw_id = table_obj.getcol('SPECTRAL_WINDOW_ID')
    scan_no = table_obj.getcol('SCAN_NUMBER')
    ant_id = table_obj.getcol('ANTENNA1')
    antnames = get_antennas(table_obj)
    # get available antenna names from antenna id
    ttip_antnames = np.array([antnames[x] for x in ant_id])

    freqs = get_frequencies(table_obj)
    nchan = len(freqs)

    if gtype == 'B':
        spw_id = spw_id.reshape(spw_id.size, 1)
        spw_id = spw_id.repeat(nchan, axis=1)
        scan_no = scan_no.reshape(scan_no.size, 1)
        scan_no = scan_no.repeat(nchan, axis=1)
        ant_id = ant_id.reshape(ant_id.size, 1)
        ant_id = ant_id.repeat(nchan, axis=1)
        # ttip_antnames = ttip_antnames.

    return spw_id, scan_no, ttip_antnames


def get_xaxis_data(table_obj, gtype):
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
    if gtype == 'B':
        xdata = get_frequencies(table_obj)
        xaxis_label = 'Channel'
    elif gtype == 'F' or gtype == 'G':
        xdata = table_obj.getcol('TIME')
        xaxis_label = 'Time[s]'
    elif gtype == 'K':
        xdata = table_obj.getcol('ANTENNA1')
        xaxis_label = 'Antenna'

    return xdata, xaxis_label


def prep_xaxis_data(xdata, gtype):
    """Function to Prepare the x-axis data.
    Inputs
    ------
    xdata: 1-D array
           Data for the xaxis to be prepared
    gtype: str
           gain type of the table to plot
    ptype: str
           Type of plot, whether ap or ri

    Outputs
    -------
    prepdx: 1-D array
            Data for the x-axid of the plots
    """
    if gtype == 'B':
        prepdx = np.arange(xdata.size)
    elif gtype == 'G' or gtype == 'F':
        prepdx = xdata - xdata[0]
    elif gtype == 'K':
        prepdx = xdata
    return prepdx


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
        y2 = np.ma.angle(m_ydata, deg=True)
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


def autofill_gains_fields(t, g, f):
    """Normalise length of f and g lists to the length of
       t list. This function is meant to support  the ability to specify multiple gain tables while only specifying single values for field ids and gain table types. An assumption will be made that for all the specified tables, the same field id and gain table type will be used.

    Inputs
    ------
    t: list
          list of the gain tables.
    f: str
            field id to be plotted.
    g: list 
           type of gain table [B,G,K,F].


    Outputs
    -------
    f, g: list
                    lists of length lengthof(t) containing field ids and gain types.
    """
    ltab = len(t)
    lfields = len(f)
    lgains = len(g)

    if ltab != lgains and lgains == 1:
        g = g * ltab
    if ltab != lfields and lfields == 1:
        f = f * ltab

    return g, f


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
                        help='Field ID(s) / NAME(s) to plot')
    parser.add_argument('-g', '--gaintype', nargs='*', type=str,
                        dest='gain_types', choices=['B', 'G', 'K', 'F'],
                        help='Type of table(s) to be plotted: B, G, K, F',
                        default=[])
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

    return parser


def main(**kwargs):
    """Main function"""
    NB_RENDER = None
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
        gain_types = kwargs.get('gain_types', [])

    if len(mytabs) == 0:
        logger.error('Exiting: No gain table specified.')
        sys.exit(-1)

    mytabs = [x.rstrip("/") for x in mytabs]

    if len(gain_types) == 0:
        logger.error('Exiting: No gain type specified.')
        sys.exit(-1)

    gain_types = [x.upper() for x in gain_types]

    if len(field_ids) == 0:
        logger.error('Exiting: No field id specified.')
        sys.exit(-1)

    if doplot not in ['ap', 'ri']:
        logger.error('Exiting: Plot selection must be ap or ri.')
        sys.exit(-1)

    # for notebook
    for gain_type in gain_types:
        if gain_type not in GAIN_TYPES:
            logger.error("Exiting: gtype {} invalid".format(gain_type))
            sys.exit(-1)

    gain_types, field_ids = autofill_gains_fields(mytabs, gain_types,
                                                  field_ids)
    # array to store final output image
    final_layout = []

    for mytab, gain_type, field in zip(mytabs, gain_types, field_ids):

        # reinitialise plotant list for each table
        if NB_RENDER:
            plotants = kwargs.get('plotants', [-1])
        else:
            plotants = options.plotants

        tt = table(mytab, ack=False)

        field_names = get_fields(tt)
        field_src_ids = dict(enumerate(field_names))
        antnames = get_antennas(tt)
        ants = np.unique(tt.getcol('ANTENNA1'))
        fields = np.unique(tt.getcol('FIELD_ID'))

        frequencies = get_frequencies(tt) / GHZ

        # setting up colors for the antenna plots
        cNorm = colors.Normalize(vmin=0, vmax=len(ants) - 1)
        mymap = cm = pylab.get_cmap(mycmap)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=mymap)

        if field.isdigit():
            field = int(field)
        else:
            field = name_2id(field, field_src_ids)

        if int(field) not in fields.tolist():
            logger.info(
                'Skipping table: {} : Field id {} not found.'.format(mytab, field))
            continue

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
        ax1 = figure(sizing_mode='scale_both', **TOOLS)
        ax2 = figure(sizing_mode='scale_both', x_range=ax1.x_range, **TOOLS)

        stats_text = stats_display(tt, gain_type, doplot, corr, field)

        # list for collecting plot states
        ax1_plots = []
        ax2_plots = []

        # forming Legend object items for data and errors
        legend_items_ax1 = []
        legend_items_ax2 = []
        legend_items_err_ax1 = []
        legend_items_err_ax2 = []

        # setting default maximum and minimum values for the different axes
        xmin = 1e20
        xmax = -1e20
        ylmin = 1e20
        ylmax = -1e20
        yumin = 1e20
        yumax = -1e20

        # for each antenna
        for ant in plotants:
            # creating legend labels
            antlabel = antnames[ant]
            legend = antnames[ant]
            legend_err = "E" + antnames[ant]

            # creating colors for maps
            y1col = scalarMap.to_rgba(float(ant))
            y2col = scalarMap.to_rgba(float(ant))

            # denormalizing the color array
            y1col = color_denormalize(y1col)
            y2col = color_denormalize(y2col)

            mytaql = 'ANTENNA1==' + str(ant)
            mytaql += '&&FIELD_ID==' + str(field)

            # querying the table for the 2 columns

            subtab = tt.query(query=mytaql)

            xdata, xlabel = get_xaxis_data(subtab, gain_type)
            prepd_x = prep_xaxis_data(xdata, gain_type)
            ydata, y1label, y2label = get_yaxis_data(subtab, gain_type,
                                                     doplot)

            # for tooltips
            spw_id, scan_no, ttip_antnames = get_tooltip_data(subtab,
                                                              gain_type)

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

            if gain_type is 'B':
                nchan = get_frequencies(subtab).size
                chans = np.arange(nchan)
                # for tooltips
                ttip_antnames = [antlabel] * nchan

                if ant == plotants[-1]:
                    linax1 = add_axis(ax1, (frequencies[0], frequencies[-1]),
                                      ax_label='Frequency [GHz]')
                    linax2 = add_axis(ax2, (frequencies[0], frequencies[-1]),
                                      ax_label='Frequency [GHz]')
                    ax1.add_layout(linax1, 'above')
                    ax2.add_layout(linax2, 'above')

            if gain_type is 'K':
                if doplot == 'ri':
                    logger.error('Exiting: No complex values to plot')
                    # break #[for when there'r multiple tables to be plotted]
                    sys.exit(-1)

            y1, y1_err, y2, y2_err = prep_yaxis_data(subtab, ydata,
                                                     gain_type,
                                                     ptype=doplot,
                                                     corr=corr,
                                                     flag=True)

            source = ColumnDataSource(data=dict(x=prepd_x, y1=y1, y2=y2,
                                                spw=spw_id, scanid=scan_no,
                                                antname=ttip_antnames))

            p1, p1_err, p2, p2_err = make_plots(
                source=source, color=y1col, ax1=ax1, ax2=ax2, y1_err=y1_err)

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

            if np.min(prepd_x) < xmin:
                xmin = np.min(prepd_x)
            if np.max(prepd_x) > xmax:
                xmax = np.max(prepd_x)
            if np.min(y1) < yumin:
                yumin = np.min(y1)
            if np.max(y1) > yumax:
                yumax = np.max(y1)
            if np.min(y2) < ylmin:
                ylmin = np.min(y2)
            if np.max(y2) > ylmax:
                ylmax = np.max(y2)

        # reorienting the min and max vales for x and y axes
        xmin = xmin - 400
        xmax = xmax + 400

        # setting the axis limits for scaliing
        if yumin < 0.0:
            yumin = -1 * (1.1 * np.abs(yumin))
        else:
            yumin = yumin * 0.9
        yumax = yumax * 1.1
        if ylmin < 0.0:
            ylmin = -1 * (1.1 * np.abs(ylmin))
        else:
            ylmin = ylmin * 0.9
        ylmax = ylmax * 1.1

        if t0 != -1:
            xmin = float(t0)
        if t1 != -1:
            xmax = float(t1)
        if yl0 != -1:
            ylmin = yl0
        if yl1 != -1:
            ylmax = yl1
        if yu0 != -1:
            yumin = yu0
        if yu1 != -1:
            yumax = yu1

        ax1.y_range = Range1d(yumin, yumax)
        ax2.y_range = Range1d(ylmin, ylmax)

        tt.close()

        # configuring titles for the plots
        ax1_title = Title(text=ax1_ylabel + ' vs ' + ax1_xlabel,
                          align='center', text_font_size='25px')
        ax2_title = Title(text=ax2_ylabel + ' vs ' + ax2_xlabel,
                          align='center', text_font_size='25px')

        ax1.add_tools(hover)
        ax2.add_tools(hover2)
        # LEGEND CONFIGURATIONS
        BATCH_SIZE = 16
        # determining the number of legend objects required to be created
        # for each plot
        num_legend_objs = int(np.ceil(len(plotants) / float(BATCH_SIZE)))

        batches_ax1, batches_ax1_err, batches_ax2, batches_ax2_err = \
            create_legend_batches(num_legend_objs, legend_items_ax1,
                                  legend_items_ax2, legend_items_err_ax1,
                                  legend_items_err_ax2, batch_size=BATCH_SIZE)

        legend_objs_ax1, legend_objs_ax1_err, legend_objs_ax2, \
            legend_objs_ax2_err = create_legend_objs(num_legend_objs,
                                                     batches_ax1,
                                                     batches_ax1_err,
                                                     batches_ax2,
                                                     batches_ax2_err)

        # adding legend objects to the layouts
        for i in range(num_legend_objs):
            ax1.add_layout(legend_objs_ax1['leg_%s' % str(i)], 'right')
            ax2.add_layout(legend_objs_ax2['leg_%s' % str(i)], 'right')

            ax1.add_layout(legend_objs_ax1_err['leg_%s' % str(i)], 'left')
            ax2.add_layout(legend_objs_ax2_err['leg_%s' % str(i)], 'left')

        # adding plot titles
        ax2.add_layout(ax2_title, 'above')
        ax1.add_layout(ax1_title, 'above')

        # creating size slider for the plots
        size_slider = Slider(end=10, start=1, step=0.5,
                             value=4, title='Scatter point size')

        # creating and configuring Antenna selection buttons
        ant_select = Toggle(label='Select All Antennas',
                            button_type='success', width=200)

        # configuring toggle button for showing all the errors
        toggle_err = Toggle(label='Show All Error bars',
                            button_type='warning', width=200)

        ant_labs = gen_checkbox_labels(BATCH_SIZE, num_legend_objs)

        batch_select = CheckboxGroup(labels=ant_labs, active=[])

        # Dropdown to hide and show legends
        legend_toggle = Select(title="Showing Legends: ", value="non",
                               options=[("all", "All"), ("alo", "Antennas"),
                                        ("elo", "Errors"), ("non", "None")])

        ant_select.callback = CustomJS(args=dict(glyph1=legend_items_ax1,
                                                 glyph2=legend_items_ax2,
                                                 batchsel=batch_select),
                                       code=ant_select_callback())

        toggle_err.callback = CustomJS(args=dict(err1=legend_items_err_ax1,
                                                 err2=legend_items_err_ax2),
                                       code=toggle_err_callback())

        # BATCH SELECTION
        batch_select.callback = CustomJS.from_coffeescript(
            args=dict(bax1=batches_ax1,
                      bax1_err=batches_ax1_err,
                      bax2=batches_ax2,
                      bax2_err=batches_ax2_err,
                      antsel=ant_select),
            code=batch_select_callback())

        legend_toggle.callback = CustomJS(
            args=dict(
                loax1=legend_objs_ax1.values(),
                loax1_err=legend_objs_ax1_err.values(),
                loax2=legend_objs_ax2.values(),
                loax2_err=legend_objs_ax2_err.values()),
            code=legend_toggle_callback())

        size_slider.callback = CustomJS(args={'slide': size_slider,
                                              'p1': ax1_plots,
                                              'p2': ax2_plots},
                                        code=size_slider_callback())

        plot_widgets = widgetbox([ant_select, batch_select,
                                  toggle_err, legend_toggle,
                                  stats_text, size_slider])

        if gain_type is not 'K':
            layout = gridplot([[plot_widgets, ax1, ax2]],
                              plot_width=700, plot_height=600)
        else:
            layout = gridplot([[plot_widgets, ax1]],
                              plot_width=700, plot_height=600)

        final_layout.append(layout)

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
    if mytabs is None:
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
