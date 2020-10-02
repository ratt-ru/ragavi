import logging

from argparse import ArgumentParser
from ragavi import __version__

logger = logging.getLogger(__name__)


class ArgumentParserError(Exception):
    pass


class MyParser(ArgumentParser):

    def error(self, message):
        raise ArgumentParserError(message)


# for ragavi-vis
def vis_argparser():
    """Create command line arguments for ragavi-vis

    Returns
    -------
    parser : :obj:`ArgumentParser()`
        Argument parser object that contains command line argument's values

    """
    x_choices = ["ant1", "antenna1", "ant2", "antenna2",
                 "amp", "amplitude", "chan", "channel", "freq",  "frequency",
                 "imag", "imaginary", "phase", "real", "scan", "time",
                 "uvdist", "UVdist", "uvdistance", "uvdistl", "uvdist_l",
                 "UVwave", "uvwave"]

    y_choices = ["amp", "amplitude", "imag", "imaginary", "phase", "real"]

    iter_choices = ["ant", "antenna", "ant1", "antenna1", "ant2", "antenna2",
                    "bl", "baseline", "corr", "field", "scan", "spw",
                    ]

    parser = MyParser(usage="ragavi-vis [options] <value>",
                      description="A Radio Astronomy Visibilities Inspector")

    infos = parser.add_argument_group("Information")

    infos.add_argument("-v", "--version", action="version",
                       version=f"ragavi {__version__}")

    required = parser.add_argument_group("Required arguments")
    required.add_argument("--ms", dest="mytabs",
                          nargs='+', type=str, metavar='',
                          help="MS to plot. Default is None",
                          default=[])
    required.add_argument('-x', "--xaxis", dest="xaxis", type=str, metavar='',
                          choices=x_choices, help="""X-axis to plot. See
                          https://ragavi.readthedocs.io/en/dev/vis.html#ragavi-vis for the accepted values.""",
                          default=None, required=True)
    required.add_argument("-y", "--yaxis", dest="yaxis", type=str, metavar='',
                          choices=y_choices, help="Y-axis to plot",
                          default=None, required=True)

    pconfig = parser.add_argument_group("Plot settings")

    pconfig.add_argument("-ch", "--canvas-height", dest="c_height", type=int,
                         metavar='',
                         help="""Set height resulting image. Note: This is
                         not the plot height. Default is 720""",
                         default=None)
    pconfig.add_argument("-cw", "--canvas-width", dest="c_width", type=int,
                         metavar='',
                         help="""Set width of the resulting image. Note: This
                        is not the plot width. Default is 1080.""",
                         default=None)
    pconfig.add_argument("--cmap", dest="mycmap", type=str, metavar='',
                         help="""Colour or colour map to use.A list of valid
                         cmap arguments can be found at:
                        https://colorcet.pyviz.org/user_guide/index.html
                        Note that if the argument "colour-axis" is supplied,
                        a categorical colour scheme will be adopted. Default
                        is blue. """,
                         default=None)
    pconfig.add_argument("--cols", dest="n_cols", type=int, metavar='',
                         help="""Number of columns in grid if iteration is
                         active. Default is 9.""",
                         default=None)
    pconfig.add_argument("-ca", "--colour-axis", dest="colour_axis", type=str,
                         metavar='',
                         choices=iter_choices,
                         help="""Select column to colourise by. This will
                         result in a single image. Default is None.""",
                         default=None)
    pconfig.add_argument("--debug", dest="debug",
                         action="store_true",
                         help="""Enable debug messages""")
    pconfig.add_argument("-ia", "--iter-axis", dest="iter_axis", type=str,
                         metavar='', choices=iter_choices,
                         help="""Select column to iterate over. This will
                        result in a grid. Default is None.""",
                         default=None)
    pconfig.add_argument("-lf", "--logfile", dest="logfile", type=str,
                         metavar="",
                         help="""The name of resulting log file (with
                         preferred extension) If no file extension is
                         provided, a '.log' extension is appended. The
                         default log file name is ragavi.log""",
                         default=None)
    pconfig.add_argument("-lp", "--link-plots", dest="link_plots",
                         action="store_true",
                         help="""Lock axis ranges for the iterated plots. All 
                        the plots will share the x- and y-axes. This is 
                        disabled by default""")
    pconfig.add_argument("-o", "--htmlname", dest="html_name", type=str,
                         metavar='',
                         help="Output HTML file name (without '.html')",
                         default=None)

    d_config = parser.add_argument_group("Data Selection")
    d_config.add_argument("-a", "--ant", dest="ants", type=str,
                          metavar='',
                          help="""Select baselines where ANTENNA1 corresponds
                          to the supplied antenna(s). "Can be
                        specified as e.g. "4", "5,6,7", "5~7" (inclusive
                        range), "5:8" (exclusive range), 5:(from 5 to last).
                        Default is all.""",
                          default=None)
    d_config.add_argument("--chan", dest="chan", type=str, metavar='',
                          help="""Channels to select. Can be specified using
                        syntax i.e "0:5" (exclusive range) or "20" for
                        channel 20 or "10~20" (inclusive range) (same as
                        10:21) "::10" for every 10th channel or "0,1,3" etc.
                        Default is all.""",
                          default=None)
    d_config.add_argument("-c", "--corr", dest="corr", type=str, metavar='',
                          help="""Correlation index or subset to plot. Can be
                        specified using normal python slicing syntax i.e
                        "0:5" for 0<=corr<5 or "::2" for every 2nd corr or
                        "0" for corr 0  or "0,1,3". Can also be specified
                        using comma separated corr labels e.g 'xx,yy' or
                        specifying 'diag' / 'diagonal' for diagonal
                        correlations and 'off-diag' / 'off-diagonal' for of
                        diagonal correlations. Default is all.""",
                          default=None)
    d_config.add_argument("-dc", "--data-column", dest="data_column",
                          type=str, metavar='',
                          help="""MS column to use for data.
                          Default is DATA.""", default="DATA")
    d_config.add_argument("--ddid", dest="ddid", type=str,
                          metavar='',
                          help="""DATA_DESC_ID(s) /spw to select. Can be
                        specified as e.g. "5", "5,6,7", "5~7" (inclusive
                        range), "5:8" (exclusive range), 5:(from 5 to last).
                        Default is all.""",
                          default=None)
    d_config.add_argument("-f", "--field", dest="fields", type=str,
                          metavar='',
                          help="""Field ID(s) / NAME(s) to plot. Can be
                        specified as "0", "0,2,4", "0~3" (inclusive range),
                        "0:3" (exclusive range), "3:" (from 3 to last) or
                        using a field name or comma separated field names.
                        Default is all""",
                          default=None)
    d_config.add_argument("-if", "--include-flagged", dest="flag",
                          action="store_false",
                          help="""Include flagged data in the plot (Plots
                          both flagged and unflagged data.)""")
    d_config.add_argument("-s", "--scan", dest="scan", type=str, metavar='',
                          help="Scan Number to select. Default is all.",
                          default=None)
    d_config.add_argument("--taql", dest="where", type=str, metavar='',
                          help="TAQL where", default=None)

    d_config.add_argument("--xmin", dest="xmin", type=float, metavar='',
                          help="Minimum x value to plot", default=None)
    d_config.add_argument("--xmax", dest="xmax", type=float, metavar='',
                          help="Maximum x value to plot", default=None)
    d_config.add_argument("--ymin", dest="ymin", type=float, metavar='',
                          help="Minimum y value to plot", default=None)
    d_config.add_argument("--ymax", dest="ymax", type=float, metavar='',
                          help="Maximum y value to plot", default=None)

    avconfig = parser.add_argument_group("Averaging settings")
    avconfig.add_argument("--cbin", dest="cbin", type=int, metavar="",
                          help="""Size of channel bins over which to average
                        .e.g setting this to 50 will average over every 5
                        channels""",
                          default=None)
    avconfig.add_argument("--tbin", dest="tbin", type=float, metavar='',
                          help="""Time in seconds over which to average .e.g
                        setting this to 120.0 will average over every 120.0
                        seconds""",
                          default=None)

    r_config = parser.add_argument_group("Resource configurations")
    r_config.add_argument("-cs", "--chunks", dest="chunks", type=str,
                          metavar='',
                          help="""Chunk sizes to be applied to the dataset.
                          Can be an integer e.g "1000", or a comma separated
                         string e.g "1000,100,2" for multiple dimensions.
                         The available dimensions are (row, chan, corr)
                         respectively. If an integer, the specified chunk
                         size will be applied to all dimensions. If comma
                         separated string, these chunk sizes will be applied
                         to each dimension respectively. Default is 5,000
                         in the row axis.""",
                          default=None)
    r_config.add_argument("-ml", "--mem-limit", dest="mem_limit",
                          type=str, metavar='',
                          default=None,
                          help="""Memory limit per core e.g '1GB' or '128MB'.
                         Default is 1GB""")
    r_config.add_argument("-nc", "--num-cores", dest="n_cores", type=int,
                          metavar='',
                          help="""Number of CPU cores to be used by Dask.
                        Default is 10 cores. Unless specified, however, this
                        value may change depending on the amount of RAM on
                        this machine to ensure that:
                        num-cores * mem-limit < total RAM available""",
                          default=None)
    return parser


def gains_argparser():
    """Create command line arguments for ragavi-gains

    Returns
    -------
    parser : :obj:`ArgumentParser()`
        Argument parser object that contains command line argument's values

    """
    parser = MyParser(usage="%(prog)s [options] <value>",
                      description="Radio Astronomy Gains Inspector")
    parser.add_argument("-v", "--version", action="version",
                        version=f"ragavi {__version__}")

    required = parser.add_argument_group("Required arguments")
    required.add_argument("-t", "--table", dest="mytabs",
                          nargs='+', type=str, metavar=(' '), required=True,
                          help="""Table(s) to plot. Multiple tables can be
                          specified as a space separated list""",
                          default=[])

    d_config = parser.add_argument_group("Data Selection")
    d_config.add_argument("-a", "--ant", dest="plotants", type=str,
                          metavar='',
                          help="""Plot only a specific antenna, or
                        comma-separated list of antennas. Defaults to all.""",
                          default=None)
    d_config.add_argument("-c", "--corr", dest="corr", type=str, metavar='',
                          help="""Correlation index to plot. Can be a single
                        integer or comma separated integers e.g '0,2'.
                        Defaults to all.""",
                          default=None)
    d_config.add_argument("--ddid", dest="ddid", type=str, metavar='',
                          help="""SPECTRAL_WINDOW_ID or ddid number.
                         Defaults to all""",
                          default=None)
    d_config.add_argument("-f", "--field", dest="fields", type=str, nargs='+',
                          metavar='',
                          help="""Field ID(s) / NAME(s) to plot. Can be
                        specified as "0", "0,2,4", "0~3" (inclusive range),
                        "0:3" (exclusive range), "3:" (from 3 to last) or
                        using a field name or comma separated field names.
                        Defaults to all""",
                          default=None)
    d_config.add_argument("--t0", dest="t0", type=float, metavar='',
                          help="""Minimum time to plot [in seconds].
                        Defaults to full range]""",
                          default=None)
    d_config.add_argument("--t1", dest="t1", type=float, metavar='',
                          help="""Maximum time to plot [in seconds].
                        Defaults to full range""",
                          default=None)
    d_config.add_argument("--taql", dest="where", type=str, metavar='',
                          help="TAQL where clause",
                          default=None)

    pconfig = parser.add_argument_group("Plot settings")
    pconfig.add_argument("--cmap", dest="mycmap", type=str, metavar='',
                         help="""Bokeh or Colorcet colour map to use for
                         antennas. List of available colour maps can be
                        found at: https://docs.bokeh.org/en/latest/docs/reference/palettes.html or
                        https://colorcet.holoviz.org/user_guide/index.html.
                        Defaults to coolwarm""",
                         default="coolwarm")

    pconfig.add_argument("-y", "--yaxis", "-d", "--doplot", dest="doplot",
                         type=str, metavar='',
                         help="""Plot complex values as any amplitude (a),
                         phase (p), real (r), imaginary (i). For a combination
                         of multiple plots, specify as a single string. e.g.
                         To plot both amplitude and phase, set this to 'ap'.
                         To plot all at once, set this to 'all'.
                         Defaults to ap.""",
                         default="ap")
    pconfig.add_argument("--debug", dest="debug",
                         action="store_true",
                         help="""Enable debug messages""")
    pconfig.add_argument("-x", "--xaxis", dest="kx", type=str, metavar='',
                         help="""Choose an x-xaxis Valid choices are: 
                         time, antenna, channel. If this is not supplied, an
                         appropriate one will be selected automatically
                         depending on the type of gains being plotted.""",
                         default=None)
    pconfig.add_argument("-lf", "--logfile", dest="logfile", type=str,
                         metavar="",
                         help="""The name of resulting log file (with
                         preferred extension) If no file extension is
                         provided, a '.log' extension is appended. The
                         default log file name is ragavi.log""",
                         default=None)
    pconfig.add_argument("-o", "--htmlname", dest="html_name", type=str,
                         metavar='',
                         help="""Name of the resulting HTML file. The '.html'
                        prefix will be appended automatically.""",
                         default=None)
    pconfig.add_argument("-p", "--plotname", dest="image_name", type=str,
                         metavar='', help="""Static image name. The suffix of
                         this name determines the type of plot. If
                         foo.png, the output will be PNG, else if foo.svg,
                         the output will be of the SVG format. PDF is also
                         accepable""",
                         default=None)

    return parser
