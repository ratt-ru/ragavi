from argparse import ArgumentParser
import argparse
from ipdb import set_trace

__version__ = "dao"
class ArgumentParserError(Exception):
    pass

class RagParser(ArgumentParser):
    def error(self, message):
        raise ArgumentParserError(message)

def vis_argparser():
    """
    Create command line arguments for ragavi-vis

    Returns
    -------
    parser : :obj:`ArgumentParser()`
        Argument parser object that contains command line argument's values

    """
    x_choices = [
        "ant1", "antenna1", "ant2", "antenna2", "amp", "amplitude", "chan",
        "channel", "freq",  "frequency", "imag", "imaginary", "phase", "real",
        "scan", "time", "uvdist", "UVdist", "uvdistance", "uvdistl",
        "uvdist_l", "UVwave", "uvwave"]

    y_choices = ["amp", "amplitude", "imag", "imaginary", "phase", "real"]

    iter_choices = [
        "ant", "antenna", "ant1", "antenna1", "ant2", "antenna2", "bl",
        "baseline", "corr", "field", "scan", "spw"]

    parser = RagParser(usage="ragavi-vis [options] <value>",
                      description="A Radio Astronomy Visibilities Inspector")

    infos = parser.add_argument_group("Information")

    infos.add_argument("-v", "--version", action="version",
                       version=f"ragavi {__version__}")

    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "--ms", dest="msnames", nargs='+', type=str, metavar='', default=[None],
        help="MS to plot. Default is None")
    required.add_argument(
        '-x', "--xaxis", dest="xaxes", nargs='+', type=str, metavar='',
        default=None, required=True,
        help="""X-axis to plot. See
        https://ragavi.readthedocs.io/en/dev/vis.html#ragavi-vis
        for the accepted values.""")
    required.add_argument(
        "-y", "--yaxis", dest="yaxes", nargs='+', type=str, metavar='',
        choices=y_choices, default=None, required=True,
        help="Y-axis to plot")

    pconfig = parser.add_argument_group("Plot settings")

    pconfig.add_argument(
        "-ch", "--canvas-height", dest="c_height", nargs='+', type=int,
        metavar='', default=None,
        help="""Set height resulting image. Note: This is not the plot
        height. Default is 720""")
    pconfig.add_argument(
        "-cw", "--canvas-width", dest="c_width", nargs='+', type=int,
        metavar='', default=None,
        help="""Set width of the resulting image. Note: This is not the
        plot width. Default is 1080.""")
    pconfig.add_argument(
        "--cmap", dest="cmaps", type=str, metavar='', nargs='+', default=[],
        help="""Colour or colour map to use.A list of valid cmap arguments
        can be found at: https://colorcet.pyviz.org/user_guide/index.html
        Note that if the argument "colour-axis" is supplied, a categorical
        colour scheme will be adopted. Default is blue. """)
    pconfig.add_argument(
        "--cols", dest="grid_cols", nargs='+', type=int, metavar='',
        default=None,
        help="""Number of columns in grid if iteration is active.
        Default is 9.""")
    pconfig.add_argument(
        "-ca", "--colour-axis", dest="c_axes", nargs='+', type=str, metavar='',
        choices=iter_choices, default=[None],
        help="""Select column to colourise by. This will result in a single
        image. Default is None.""")
    pconfig.add_argument(
        "--debug", dest="debug", action="store_true",
        help="""Enable debug messages""")
    pconfig.add_argument(
        "-ia", "--iter-axis", dest="i_axes", nargs='+', type=str, metavar='',
        choices=iter_choices, default=[None],
        help="""Select column to iterate over. This will result in a grid.
        Default is None.""")
    pconfig.add_argument(
        "-lf", "--logfile", dest="logfile", nargs='+', type=str, metavar="",
        default=None,
        help="""The name of resulting log file (with preferred extension)
        If no file extension is provided, a '.log' extension is appended.
        The default log file name is ragavi.log""")
    pconfig.add_argument(
        "-lp", "--link-plots", dest="link_plots", action="store_true",
        help="""Lock axis ranges for the iterated plots. All the plots will
        share the x- and y-axes. This is disabled by default""")
    pconfig.add_argument(
        "-o", "--htmlname", dest="html_names", nargs='+', type=str, metavar='',
        default=[None],
        help="Output HTML file name (without '.html')")

    d_config = parser.add_argument_group("Data Selection")
    d_config.add_argument(
        "-a", "--ant", dest="antennas", nargs='+', type=str, metavar='',
        default=[None],
        help="""Select baselines where ANTENNA1 corresponds to the supplied
        antenna(s). "Can be specified as e.g. "4", "5,6,7", "5~7" (inclusive
        range), "5:8" (exclusive range), 5:(from 5 to last). Default is all.""")
    d_config.add_argument(
        "-b", "--baseline", dest="baselines", nargs='+', type=str, metavar='',
        default=[None],
        help="""Plot only a specific baseline, or comma-separated list of
        baselines e.g m001-m000, m002-m005. Defaults to all.""")
    d_config.add_argument(
        "--chan", dest="channels", nargs='+', type=str, metavar='',
        default=[None],
        help="""Channels to select. Can be specified using syntax i.e "0:5"
        (exclusive range) or "20" for channel 20 or "10~20" (inclusive range)
        (same as 10:21) "::10" for every 10th channel or "0,1,3" etc.
        Default is all.""")
    d_config.add_argument(
        "-c", "--corr", dest="corrs", nargs='+', type=str, metavar='',
        default=[None],
        help="""Correlation index or subset to plot. Can be specified using
        normal python slicing syntax i.e "0:5" for 0<=corr<5 or "::2" for
        every 2nd corr or "0" for corr 0  or "0,1,3". Can also be specified
        using comma separated corr labels e.g 'xx,yy' or specifying
        'diag' / 'diagonal' for diagonal correlations and 'off-diag'
        / 'off-diagonal' for of diagonal correlations. Default is all.""")
    d_config.add_argument(
        "-dc", "--data-column", dest="data_columns", nargs='+', type=str,
        metavar='', default=[None],
        help="""MS column to use for data. Default is DATA.""")
    d_config.add_argument(
        "--ddid", dest="ddids", nargs='+', type=str, metavar='',
        default=[None],
        help="""DATA_DESC_ID(s) /spw to select. Can be specified as e.g. "5",
        "5,6,7", "5~7" (inclusive range), "5:8" (exclusive range),
        5:(from 5 to last). Default is all.""")
    d_config.add_argument(
        "-f", "--field", dest="fields", nargs='+', type=str, metavar='',
        default=[None],
        help="""Field ID(s) / NAME(s) to plot. Can be specified as "0",
        "0,2,4", "0~3" (inclusive range), "0:3" (exclusive range), "3:"
        (from 3 to last) or using a field name or comma separated field names.
        Default is all""")
    d_config.add_argument(
        "-if", "--include-flagged", dest="flag", action="store_false",
        help="""Include flagged data in the plot (Plots both flagged and
        unflagged data.)""")
    d_config.add_argument(
        "-s", "--scan", dest="scans", nargs='+', type=str, metavar='',
        default=[None],
        help="Scan Number to select. Default is all.")
    d_config.add_argument(
        "--taql", dest="taqls", nargs='+', type=str, metavar='', default=[None],
        help="TAQL where")
    d_config.add_argument(
        "--xmin", dest="xmin", type=float, metavar='', default=None,
        help="Minimum x value to plot")
    d_config.add_argument(
        "--xmax", dest="xmax", type=float, metavar='', default=None,
        help="Maximum x value to plot")
    d_config.add_argument(
        "--ymin", dest="ymin", type=float, metavar='', default=None,
        help="Minimum y value to plot")
    d_config.add_argument(
        "--ymax", dest="ymax", type=float, metavar='', default=None,
        help="Maximum y value to plot")

    avconfig = parser.add_argument_group("Averaging settings")
    avconfig.add_argument(
        "--cbin", dest="cbins", nargs='+', type=int, metavar="",
        default=None,
        help="""Size of channel bins over which to average .e.g setting this
        to 50 will average over every 5 channels""")
    avconfig.add_argument(
        "--tbin", dest="tbins", nargs='+', type=float, metavar='',
        default=None,
        help="""Time in seconds over which to average .e.g setting this to
        120.0 will average over every 120.0 seconds""")

    r_config = parser.add_argument_group("Resource configurations")
    r_config.add_argument(
        "-cs", "--chunks", dest="chunk_size", type=str, metavar='',
        default=None,
        help="""Chunk sizes to be applied to the dataset. Can be an integer
        e.g "1000", or a comma separated string e.g "1000,100,2" for multiple
        dimensions. The available dimensions are (row, chan, corr)
        respectively. If an integer, the specified chunk size will be applied
        to all dimensions. If comma separated string, these chunk sizes will
        be applied to each dimension respectively.
        Default is 5,000 in the row axis.""")
    r_config.add_argument(
        "-ml", "--mem-limit", dest="mem_limit", type=str, metavar='',
        default=None,
        help="""Memory limit per core e.g '1GB' or '128MB'. Default is 1GB""")
    r_config.add_argument(
        "-nc", "--num-cores", dest="ncores", type=int, metavar='',
        default=None,
        help="""Number of CPU cores to be used by Dask. Default is 10 cores.
        Unless specified, however, this value may change depending on the
        amount of RAM on this machine to ensure that:
        num-cores * mem-limit < total RAM available""")
    return parser


def gains_argparser():
    """
    Create command line arguments for ragavi-gains

    Returns
    -------
    parser : :obj:`ArgumentParser()`
        Argument parser object that contains command line argument's values

    """
    parser = RagParser(usage="%(prog)s [options] <value>",
                      description="Radio Astronomy Gains Inspector")
    parser.add_argument("-v", "--version", action="version",
                        version=f"ragavi {__version__}")
    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "-t", "--table", dest="msnames", nargs='+', type=str, metavar='',
        default=[None],
        help="""Table(s) to plot. Multiple tables can be specified as a space
        separated list""")

    d_config = parser.add_argument_group("Data Selection")
    d_config.add_argument(
        "-a", "--ant", dest="antennas", nargs='+', type=str, metavar='',
        default=[None],
        help="""Plot only a specific antenna, or comma-separated list of
        antennas. Defaults to all.""")
    d_config.add_argument(
        "-b", "--baseline", dest="baselines", nargs='+', type=str, metavar='',
        default=[None],
        help="""Plot only a specific antenna, or comma-separated list of
        antennas. Defaults to all.""")
    d_config.add_argument(
        "-chan", "--channel", dest="channels", type=str, metavar='',
        nargs='+', default=[None],
        help="""Channels to include. Defaults to all.""")
    d_config.add_argument(
        "-c", "--corr", dest="corrs", type=str, metavar='', nargs='+',
        default=[None],
        help="""Correlation index to plot. Can be a single
        integer or comma separated integers e.g '0,2'. Defaults to all.""")
    d_config.add_argument(
        "--ddid", dest="ddids", type=str, metavar='', nargs='+', default=[None],
        help="""SPECTRAL_WINDOW_ID or ddid number. Defaults to all""")
    d_config.add_argument(
        "-f", "--field", dest="fields", type=str, nargs='+', metavar='',
        default=[None],
        help="""Field ID(s) / NAME(s) to plot. Can be specified as
        "0", "0,2,4", "0~3" (inclusive range), "0:3" (exclusive range), "3:" 
        (from 3 to last) or using a field name or comma separated field names.
        Defaults to all""")
    d_config.add_argument(
        "--t0", dest="t0s", nargs='+', type=float, metavar='', default=[None],
        help="""Minimum time to plot [in seconds]. Defaults to full range]""")
    d_config.add_argument(
        "--t1", dest="t1s", nargs='+', type=float, metavar='', default=[None],
        help="""Maximum time to plot [in seconds]. Defaults to full range""")
    d_config.add_argument(
        "--taql", dest="taqls", nargs='+', type=str, metavar='', default=[None],
        help="TAQL where clause")

    pconfig = parser.add_argument_group("Plot settings")
    pconfig.add_argument(
        "--cmap", dest="cmaps", nargs='+', type=str, metavar='',
        default=["coolwarm"],
        help="""Bokeh or Colorcet colour map to use for antennas. List of
        available colour maps can be found at: 
        https://docs.bokeh.org/en/latest/docs/reference/palettes.html or
        https://colorcet.holoviz.org/user_guide/index.html. 
        Defaults to coolwarm""")
    pconfig.add_argument(
        "-y", "--yaxis", "-d", "--doplot", dest="yaxes", nargs='+', type=str,
        metavar='', default=["ap"],
        help="""Plot complex values as any amplitude (a), phase (p), real (r),
        imaginary (i). For a combination of multiple plots, specify as a
        single string. e.g. To plot both amplitude and phase, set this to
        'ap'. To plot all at once, set this to 'all'. Defaults to ap.""")
    pconfig.add_argument(
        "--debug", dest="debug", action="store_true",
        help="""Enable debug messages""")
    pconfig.add_argument(
        "-x", "--xaxis", dest="xaxes", nargs='+', type=str, metavar='',
        default=[None],
        help="""Choose an x-xaxis Valid choices are: time, antenna, channel.
        If this is not supplied, an appropriate one will be selected
        automatically depending on the type of gains being plotted.""")
    pconfig.add_argument(
        "-lf", "--logfile", dest="logfile", type=str, metavar="",
        default=[None],
        help="""The name of resulting log file (with preferred extension)
        If no file extension is provided, a '.log' extension is appended.
        The default log file name is ragavi.log""")
    pconfig.add_argument(
        "-o", "--htmlname", dest="html_names", nargs='+', type=str, metavar='',
        default=[None],
        help="""Name of the resulting HTML file. The '.html' prefix will be
        appended automatically.""")
    pconfig.add_argument(
        "-p", "--plotname", dest="image_names", nargs='+', type=str, 
        metavar='', default=[None],
        help="""Static image name. The suffix of this name determines the type
        of plot. If foo.png, the output will be PNG, else if foo.svg, the
        output will be of the SVG format. PDF is also accepable""")
    return parser


def cubical_gains_parser():
    """
    Create command line arguments for ragavi-gains

    Returns
    -------
    parser : :obj:`ArgumentParser()`
        Argument parser object that contains command line argument's values

    """
    parser = RagParser(usage="%(prog)s [options] <value>",
                      description="Radio Astronomy Gains Inspector")
    parser.add_argument("-v", "--version", action="version",
                        version=f"ragavi {__version__}")
    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "-t", "--table", dest="msnames", nargs='+', type=str, metavar='',
        default=[None],
        help="""Table(s) to plot. Multiple tables can be specified as a space
        separated list""")
    d_config = parser.add_argument_group("Data Selection")
    d_config.add_argument(
        "-a", "--ant", dest="antennas", nargs='+', type=str, metavar='',
        default=[None],
        help="""Plot only a specific antenna, or comma-separated list of
        antennas. Defaults to all.""")
    d_config.add_argument(
        "-chan", "--channel", dest="channels", type=str, metavar='',
        nargs='+', default=[None],
        help="""Channels to include. Defaults to all.""")
    d_config.add_argument(
        "-c", "--corr", dest="corrs", type=str, metavar='', nargs='+',
        default=[None],
        help="""Correlation index to plot. Can be a single
        integer or comma separated integers e.g '0,2'. Defaults to all.""")
    d_config.add_argument(
        "-f", "--field", dest="fields", type=str, nargs='+', metavar='',
        default=[None],
        help="""Field ID(s) / NAME(s) to plot. Can be specified as
        "0", "0,2,4", "0~3" (inclusive range), "0:3" (exclusive range), "3:" 
        (from 3 to last) or using a field name or comma separated field names.
        Defaults to all""")
    d_config.add_argument(
        "--t0", dest="t0s", nargs='+', type=float, metavar='', default=[None],
        help="""Minimum time to plot [in seconds]. Defaults to full range]""")
    d_config.add_argument(
        "--t1", dest="t1s", nargs='+', type=float, metavar='', default=[None],
        help="""Maximum time to plot [in seconds]. Defaults to full range""")

    pconfig = parser.add_argument_group("Plot settings")
    pconfig.add_argument(
        "--cmap", dest="cmaps", nargs='+', type=str, metavar='',
        default=["coolwarm"],
        help="""Bokeh or Colorcet colour map to use for antennas. List of
        available colour maps can be found at: 
        https://docs.bokeh.org/en/latest/docs/reference/palettes.html or
        https://colorcet.holoviz.org/user_guide/index.html. 
        Defaults to coolwarm""")
    pconfig.add_argument(
        "-y", "--yaxis", "-d", "--doplot", dest="yaxes", nargs='+', type=str,
        metavar='', default=["ap"],
        help="""Plot complex values as any amplitude (a), phase (p), real (r),
        imaginary (i). For a combination of multiple plots, specify as a
        single string. e.g. To plot both amplitude and phase, set this to
        'ap'. To plot all at once, set this to 'all'. Defaults to ap.""")
    pconfig.add_argument(
        "--debug", dest="debug", action="store_true",
        help="""Enable debug messages""")
    pconfig.add_argument(
        "-x", "--xaxis", dest="xaxes", nargs='+', type=str, metavar='',
        default=[None],
        help="""Choose an x-xaxis Valid choices are: time, antenna, channel.
        If this is not supplied, an appropriate one will be selected
        automatically depending on the type of gains being plotted.""")
    pconfig.add_argument(
        "-lf", "--logfile", dest="logfile", type=str, metavar="",
        default=[None],
        help="""The name of resulting log file (with preferred extension)
        If no file extension is provided, a '.log' extension is appended.
        The default log file name is ragavi.log""")
    pconfig.add_argument(
        "-o", "--htmlname", dest="html_names", nargs='+', type=str, metavar='',
        default=[None],
        help="""Name of the resulting HTML file. The '.html' prefix will be
        appended automatically.""")
    pconfig.add_argument(
        "-p", "--plotname", dest="image_names", nargs='+', type=str, 
        metavar='', default=[None],
        help="""Static image name. The suffix of this name determines the type
        of plot. If foo.png, the output will be PNG, else if foo.svg, the
        output will be of the SVG format. PDF is also accepable""")
    return parser
