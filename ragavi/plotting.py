from bokeh.models import (BasicTicker, DatetimeAxis, DataRange1d, Grid,
                          Legend, LinearAxis, LinearScale, LogAxis, LogScale,
                          Toolbar, Plot, PrintfTickFormatter, Range1d,
                          Scatter, Text, Title, Whisker)
from bokeh.models.tools import (BoxEditTool, BoxSelectTool, BoxZoomTool,
                                EditTool, HoverTool, LassoSelectTool, PanTool,
                                ResetTool, RangeTool, SaveTool, UndoTool,
                                WheelZoomTool)


def create_bk_fig(x=None, xlab=None, x_min=None, x_max=None,
                  ylab=None, fh=None, fw=None,
                  title=None, pw=None, ph=None, x_axis_type="linear",
                  y_axis_type="linear", x_name=None, y_name=None, **kwargs):
    """ Generates a bokeh figure

    Parameters
    ----------
    x :obj:`DataArray`
        Contains x-axis data
    xlab : :obj:`str`
        X-axis label
    x_min : :obj:`float`
        Min x value
    x_max : :obj:`float`
        Max x value
    ylab : :obj:`str`
        Y-axis label
    fh: :obj:`int`
        True height of figure without legends, axes titles etc
    fw: :obj:`int`
        True width of figure without legends, axes etc
    title: :obj:`str`
        Title of plot
    pw: :obj:`int`
        Plot width including legends, axes etc
    ph: :obj:`int`
        Plot height including legends, axes etc
    x_axis_type: :obj:`str`
        Type of x-axis can be linear, log, or datetime
    y_axis_type: :obj:`str`
        Can be linear, log or datetime
    x_name: :obj:`str`
        Name of the column used for the x-axis. Mostly used to form tooltips
    y_name: :obj:`str`
        Name of the column used for the y-axis. Also used for tooltips
    add_grid: :obj:`bool`
        Whether or not to add grid
    add_title: :obj:`bool`
        Whether or not to add title to plot
    add_xaxis: :obj:`bool`
        Whether or not to add x-axis and tick marks
    add_yaxis: :obj:`bool`
        Add y-axis or not
    fix_plotsize: :obj:`bool`
        Enforce certain dimensions on plot. This is useful for ensuring a plot is not obscure by axes and other things. If activated, plot's dimensions will not be responsive. It utilises fw and fh.

    Returns
    -------
    p : :obj:`Plot`
        A bokeh Plot object

    """

    add_grid = kwargs.pop("add_grid", False)
    add_title = kwargs.pop("add_title", True)
    add_xaxis = kwargs.pop("add_xaxis", False)
    add_yaxis = kwargs.pop("add_yaxis", False)
    fix_plotsize = kwargs.pop("fix_plotsize", True)
    # addition plot specs
    pl_specs = kwargs.pop("pl_specs", {})
    # additional axis specs
    ax_specs = kwargs.pop("ax_specs", {})
    # ticker specs
    ti_specs = kwargs.pop("ti_specs", {})

    plot_specs = dict(background="white", border_fill_alpha=0.1,
                      border_fill_color="white", min_border=3,
                      name="plot", outline_line_dash="solid",
                      outline_line_width=2, outline_line_color="#017afe",
                      outline_line_alpha=0.4, output_backend="canvas",
                      sizing_mode="stretch_width", title_location="above",
                      toolbar_location="right")
    plot_specs.update(pl_specs)

    axis_specs = dict(minor_tick_line_alpha=0, axis_label_text_align="center",
                      axis_label_text_font="monospace",
                      axis_label_text_font_size="10px",
                      axis_label_text_font_style="normal",
                      major_label_orientation="horizontal")
    axis_specs.update(ax_specs)

    tick_specs = dict(desired_num_ticks=5)
    tick_specs.update(ti_specs)

    # Define frame width and height
    # This is the actual size of the plot without the titles et al
    if fix_plotsize and not(fh or fw):
        fw = int(0.98 * pw)
        fh = int(0.93 * ph)

    # percentage increase in axis ranges
    inc = 1.00

    # define the axes ranges
    x_range = DataRange1d(name="p_x_range", only_visible=True)

    if x_min != None and x_max != None and x_name.lower() in ["channel", "frequency"]:
        x_range = Range1d(name="p_x_range", start=x_min, end=x_max)

    y_range = DataRange1d(name="p_y_range", only_visible=True)

    # define items to add on the plot
    p_htool = HoverTool(tooltips=[(x_name, "$x"),
                                  (y_name, "$y")],
                        name="p_htool", point_policy="snap_to_data")

    p_toolbar = Toolbar(name="p_toolbar",
                        tools=[p_htool, BoxSelectTool(), BoxZoomTool(),
                               # EditTool(), # BoxEditTool(), # RangeTool(),
                               LassoSelectTool(), PanTool(), ResetTool(),
                               SaveTool(), UndoTool(), WheelZoomTool()])
    p_ticker = BasicTicker(name="p_ticker", **tick_specs)

    # select the axis scales for x and y
    if x_axis_type == "linear":
        x_scale = LinearScale(name="p_x_scale")
        # define the axes and tickers
        p_x_axis = LinearAxis(axis_label=xlab, name="p_x_axis",
                              ticker=p_ticker, **axis_specs)
    elif x_axis_type == "datetime":
        x_scale = LinearScale(name="p_x_scale")
        # define the axes and tickers
        p_x_axis = DatetimeAxis(axis_label=xlab, name="p_x_axis",
                                ticker=p_ticker, **axis_specs)
    elif x_axis_type == "log":
        x_scale = LogScale(name="p_x_scale")
        p_x_axis = LogAxis(axis_label=xlab, name="p_x_axis",
                           ticker=p_ticker, **axis_specs)

    if y_axis_type == "linear":
        y_scale = LinearScale(name="p_y_scale")
        # define the axes and tickers
        p_y_axis = LinearAxis(axis_label=ylab, name="p_y_axis",
                              ticker=p_ticker, **axis_specs)
    elif x_axis_type == "datetime":
        y_scale = LinearScale(name="p_y_scale")
        # define the axes and tickers
        p_y_axis = DatetimeAxis(axis_label=xlab, name="p_y_axis",
                                ticker=p_ticker, **axis_specs)
    elif y_axis_type == "log":
        y_scale = LogScale(name="p_y_scale")
        # define the axes and tickers
        p_y_axis = LogAxis(axis_label=ylab, name="p_y_axis",
                           ticker=p_ticker, **axis_specs)

    # Create the plot object
    p = Plot(plot_width=pw, plot_height=ph, frame_height=fh, frame_width=fw,
             toolbar=p_toolbar, x_range=x_range, x_scale=x_scale,
             y_range=y_range, y_scale=y_scale, **plot_specs)

    if add_title:
        p_title = Title(align="center", name="p_title", text=title,
                        text_font_size="24px",
                        text_font="monospace", text_font_style="bold",)
        p.add_layout(p_title, "above")

    if add_xaxis:
        p.add_layout(p_x_axis, "below")

    if add_yaxis:
        p.add_layout(p_y_axis, "left")

    if add_grid:
        p_x_grid = Grid(dimension=0, ticker=p_ticker)
        p_y_grid = Grid(dimension=1, ticker=p_ticker)
        p.add_layout(p_x_grid)
        p.add_layout(p_y_grid)

    return p


def add_axis(fig, axis_range, ax_label, ax_name=None):
    """Add an extra axis to the current figure

    Parameters
    ----------
    fig: :obj:`bokeh.plotting.figure`
        The figure onto which to add extra axis

    axis_range: :obj:`list`, :obj:`tuple`
        A range of sorted values or a tuple with 2 values containing or the
        order (min, max). This can be any ordered iteraable that can be
        indexed.

    Returns
    -------
    fig : :obj:`bokeh.plotting.figure`
        Bokeh figure with an extra axis added
    """
    if fig.extra_x_ranges is None:
        fig.extra_x_ranges = {}

    if ax_name is None:
        ax_name = "p_extra_xaxis"

    fig.extra_x_ranges[ax_name] = Range1d(start=axis_range[0],
                                          end=axis_range[-1])

    linaxis = LinearAxis(x_range_name=ax_name, axis_label=ax_label,
                         axis_label_text_font="monospace",
                         axis_label_text_font_style="normal",
                         axis_label_text_font_size="8pt",
                         major_tick_out=2,
                         major_tick_in=2,
                         major_label_text_font_size="8pt",
                         minor_tick_line_width=1,
                         major_label_orientation='horizontal',
                         name=ax_name,
                         ticker=BasicTicker(desired_num_ticks=8))
    fig.add_layout(linaxis, 'above')
    return fig
