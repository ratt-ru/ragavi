import argparse
from ragavi import version


class ArgumentParserError(Exception):
    pass


class RagParser(argparse.ArgumentParser):
    def error(self, message):
        raise ArgumentParserError(message)


def base_parser():
    """
    Create command line arguments for ragavi. This is the base template

    Returns
    -------
    parser:: obj: `ArgumentParser()`
    Argument parser object that contains command line argument's values
    """
    parent = argparse.ArgumentParser(
        usage="%(prog)s -x time -y amp --ms test.ms",
        description="Radio Astronomy Gains and Visibilities Inspector (RAGaVI)",
        epilog=(
            "Incase of any issues or queries: "
            + "https://github.com/ratt-ru/ragavi/issues"
        ),
        parents=[],
        prefix_chars="-",
        argument_default=None,
        conflict_handler="resolve",
        add_help=False,
    )

    req_group = parent.add_argument_group("Required arguments")
    gen_group = parent.add_argument_group("General arguments")
    gen_group.add_argument(
        "-h", "--help", action="help", help="show this help message and exit"
    )
    gen_group.add_argument("-v", "--version", action="version", version=version)
    gen_group.add_argument(
        "--debug", dest="debug", action="store_true", help="""Enable debug messages"""
    )
    gen_group.add_argument(
        "-lf",
        "--logfile",
        dest="logfile",
        # nargs="+",
        type=str,
        metavar="",
        default=None,
        help="""The name of resulting log file (with preferred extension)
        If no file extension is provided, a '.log' extension is appended.
        The default log file name is ragavi.log""",
    )
    gen_group.add_argument(
        "-od",
        "--out-dir",
        dest="out_dir",
        type=str,
        metavar="",
        default=None,
        help="""Where to dump ragavi outputs. Default is 'ragavi_out'.
        If a file is already contained there, it will be overwritten""",
    )

    ds_group = parent.add_argument_group("Data selection settings")
    ds_group.add_argument(
        "-a",
        "--ant", "--antenna",
        dest="antennas",
        nargs="+",
        type=str,
        metavar="",
        default=[None],
        help="""Select baselines where ANTENNA1 corresponds to the supplied
        antenna(s). "Can be specified as e.g. "4", "5,6,7", "5~7" (inclusive
        range), "5:8" (exclusive range), 5:(from 5 to last). Default is all.""",
    )
    ds_group.add_argument(
        "--chan", "--channel",
        dest="channels",
        nargs="+",
        type=str,
        metavar="",
        default=[None],
        help="""Channels to select. Can be specified using syntax i.e "0:5"
        (exclusive range) or "20" for channel 20 or "10~20" (inclusive range)
        (same as 10:21) "::10" for every 10th channel or "0,1,3" etc.
        Default is all.""",
    )
    ds_group.add_argument(
        "-c",
        "--corr",
        dest="corrs",
        nargs="+",
        type=str,
        metavar="",
        default=[None],
        help="""Correlation index or subset to plot. Can be specified using
        normal python slicing syntax i.e "0:5" for 0<=corr<5 or "::2" for
        every 2nd corr or "0" for corr 0  or "0,1,3". Can also be specified
        using comma separated corr labels e.g 'xx,yy' or specifying
        'diag' / 'diagonal' for diagonal correlations and 'off-diag'
        / 'off-diagonal' for of diagonal correlations. Default is all.""",
    )
    ds_group.add_argument(
        "--ddid", "--spectral-window",
        dest="ddids",
        nargs="+",
        type=str,
        metavar="",
        default=[None],
        help="""DATA_DESC_ID(s) /spw to select. Can be specified as e.g. "5",
        "5,6,7", "5~7" (inclusive range), "5:8" (exclusive range),
        5:(from 5 to last). Default is all.""",
    )
    ds_group.add_argument(
        "-f",
        "--field",
        dest="fields",
        nargs="+",
        type=str,
        metavar="",
        default=[None],
        help="""Field ID(s) / NAME(s) to plot. Can be specified as "0",
        "0,2,4", "0~3" (inclusive range), "0:3" (exclusive range), "3:"
        (from 3 to last) or using a field name or comma separated field names.
        Default is all""",
    )

    plt_group = parent.add_argument_group("Plot settings")
    plt_group.add_argument(
        "--cmap",
        dest="cmaps",
        type=str,
        metavar="",
        nargs="+",
        default=[],
        help="""Colour or colour map to use.A list of valid cmap arguments
        can be found at: https://colorcet.pyviz.org/user_guide/index.html
        Note that if the argument "colour-axis" is supplied, a categorical
        colour scheme will be adopted. Default is blue. """,
    )
    plt_group.add_argument(
        "-o",
        "--htmlname",
        dest="html_names",
        nargs="+",
        type=str,
        metavar="",
        default=[None],
        help="Output HTML file name (without '.html')",
    )

    return parent


def vis_argparser():
    """
    Create command line arguments for ragavi-vis
    """
    x_choices = [
        "ant1",
        "antenna1",
        "ant2",
        "antenna2",
        "amp",
        "amplitude",
        "chan",
        "channel",
        "freq",
        "frequency",
        "imag",
        "imaginary",
        "phase",
        "real",
        "scan",
        "time",
        "uvdist",
        "UVdist",
        "uvdistance",
        "uvdistl",
        "uvdist_l",
        "UVwave",
        "uvwave",
    ]

    y_choices = ["amp", "amplitude", "imag", "imaginary", "phase", "real"]

    iter_choices = [
        "ant",
        "antenna",
        "ant1",
        "antenna1",
        "ant2",
        "antenna2",
        "bl",
        "baseline",
        "corr",
        "field",
        "scan",
        "spw",
    ]

    parser = RagParser(
        usage="%(prog)s [options] <value>",
        description="A Radio Astronomy Visibilities Inspector",
        parents=[base_parser()],
    )

    for grp in parser._action_groups:
        if "required" in grp.title.lower():
            grp.add_argument(
                "--ms",
                dest="msnames",
                nargs="+",
                type=str,
                metavar="",
                default=[None],
                help="MS to plot. Default is None",
            )
            grp.add_argument(
                "-x",
                "--xaxis",
                dest="xaxes",
                nargs="+",
                type=str,
                metavar="",
                default=None,
                required=True,
                help="""X-axis to plot. See
                https://ragavi.readthedocs.io/en/dev/vis.html#ragavi-vis
                for the accepted values.""",
            )
            grp.add_argument(
                "-y",
                "--yaxis",
                dest="yaxes",
                nargs="+",
                type=str,
                metavar="",
                default=None,
                required=True,
                help="Y-axis to plot",
            )

        if "plot" in grp.title.lower():
            grp.add_argument(
                "-ch",
                "--canvas-height",
                dest="c_height",
                type=int,
                metavar="",
                default=None,
                help="""Set height resulting image. Note: This is not the plot
                height. Default is 720""",
            )
            grp.add_argument(
                "-cw",
                "--canvas-width",
                dest="c_width",
                type=int,
                metavar="",
                default=None,
                help="""Set width of the resulting image. Note: This is not the
                plot width. Default is 1080.""",
            )
            grp.add_argument(
                "--cols",
                dest="grid_cols",
                type=int,
                metavar="",
                default=None,
                help="""Number of columns in grid if iteration is active.
                Default is 9.""",
            )
            grp.add_argument(
                "-ca",
                "--colour-axis",
                dest="c_axes",
                nargs="+",
                type=str,
                metavar="",
                default=[None],
                help="""Select column to colourise by. This will result in a
                single image. Default is None.""",
            )
            grp.add_argument(
                "-ia",
                "--iter-axis",
                dest="i_axes",
                nargs="+",
                type=str,
                metavar="",
                default=[None],
                help="""Select column to iterate over. This will result in a 
                grid. Default is None.""",
            )
            grp.add_argument(
                "-lp",
                "--link-plots",
                dest="link_plots",
                action="store_true",
                help="""Lock axis ranges for the iterated plots. All the plots 
                will share the x- and y-axes. This is disabled by default""",
            )
            grp.add_argument(
                "-o",
                "--htmlname",
                dest="html_names",
                nargs="+",
                type=str,
                metavar="",
                default=[None],
                help="Output HTML file name (without '.html')",
            )

        if "selection" in grp.title.lower():
            grp.add_argument(
                "-b",
                "--baseline",
                dest="baselines",
                nargs="+",
                type=str,
                metavar="",
                default=[None],
                help="""Plot only a specific baseline, or comma-separated list 
                of baselines e.g m001-m000, m002-m005. Defaults to all.""",
            )
            grp.add_argument(
                "-c",
                "--corr",
                dest="corrs",
                nargs="+",
                type=str,
                metavar="",
                default=[None],
                help="""Correlation index or subset to plot. Can be specified 
                using normal python slicing syntax i.e "0:5" for 0<=corr<5 or 
                "::2" for every 2nd corr or "0" for corr 0  or "0,1,3". Can also 
                be specified using comma separated corr labels e.g 'xx,yy' or 
                specifying 'diag' / 'diagonal' for diagonal correlations and 
                'off-diag' / 'off-diagonal' for of diagonal correlations. 
                Default is all.""",
            )
            grp.add_argument(
                "-dc",
                "--data-column",
                dest="data_columns",
                nargs="+",
                type=str,
                metavar="",
                default=[None],
                help="""MS column to use for data. Default is DATA.""",
            )
            grp.add_argument(
                "-if",
                "--include-flagged",
                dest="flag",
                action="store_false",
                help="""Include flagged data in the plot (Plots both flagged and
                unflagged data.)""",
            )
            grp.add_argument(
                "-s",
                "--scan",
                dest="scans",
                nargs="+",
                type=str,
                metavar="",
                default=[None],
                help="Scan Number to select. Default is all.",
            )
            grp.add_argument(
                "--taql",
                dest="taqls",
                nargs="+",
                type=str,
                metavar="",
                default=[None],
                help="TAQL where",
            )
            grp.add_argument(
                "--xmin",
                dest="xmin",
                type=float,
                metavar="",
                default=None,
                help="Minimum x value to plot",
            )
            grp.add_argument(
                "--xmax",
                dest="xmax",
                type=float,
                metavar="",
                default=None,
                help="Maximum x value to plot",
            )
            grp.add_argument(
                "--ymin",
                dest="ymin",
                type=float,
                metavar="",
                default=None,
                help="Minimum y value to plot",
            )
            grp.add_argument(
                "--ymax",
                dest="ymax",
                type=float,
                metavar="",
                default=None,
                help="Maximum y value to plot",
            )

    avconfig = parser.add_argument_group("Averaging settings")
    avconfig.add_argument(
        "--cbin",
        dest="cbins",
        nargs="+",
        type=int,
        metavar="",
        default=None,
        help="""Size of channel bins over which to average .e.g setting this
        to 50 will average over every 5 channels""",
    )
    avconfig.add_argument(
        "--tbin",
        dest="tbins",
        nargs="+",
        type=float,
        metavar="",
        default=None,
        help="""Time in seconds over which to average .e.g setting this to
        120.0 will average over every 120.0 seconds""",
    )

    r_config = parser.add_argument_group("Resource configurations")
    r_config.add_argument(
        "-cs",
        "--chunks",
        dest="chunk_size",
        type=str,
        metavar="",
        default=None,
        help="""Chunk sizes to be applied to the dataset. Can be an integer
        e.g "1000", or a comma separated string e.g "1000,100,2" for multiple
        dimensions. The available dimensions are (row, chan, corr)
        respectively. If an integer, the specified chunk size will be applied
        to all dimensions. If comma separated string, these chunk sizes will
        be applied to each dimension respectively. Default is 5,000 in the 
        row axis.""",
    )
    r_config.add_argument(
        "-ml",
        "--mem-limit",
        dest="mem_limit",
        type=str,
        metavar="",
        default=None,
        help="""Memory limit per core e.g '1GB' or '128MB'. Default is 1GB""",
    )
    r_config.add_argument(
        "-nc",
        "--num-cores",
        dest="ncores",
        type=int,
        metavar="",
        default=None,
        help="""Number of CPU cores to be used by Dask. Default is 10 cores.
        Unless specified, however, this value may change depending on the
        amount of RAM on this machine to ensure that:
        num-cores * mem-limit < total RAM available""",
    )
    return parser


def gains_argparser():
    """Create command line arguments for ragavi-gains"""
    parser = RagParser(
        usage="%(prog)s [options] <value>",
        description="Radio Astronomy Gains Inspector for CASA gain tables",
        parents=[base_parser()],
        add_help=False,
    )

    for grp in parser._action_groups:
        if "required" in grp.title.lower():
            grp.add_argument(
                "-t",
                "--table",
                dest="msnames",
                nargs="+",
                type=str,
                metavar="",
                default=[None],
                help="""Table(s) to plot. Multiple tables can be specified as a
                space separated list""",
            )

        if "selection" in grp.title.lower():
            grp.add_argument(
                "-c",
                "--corr",
                dest="corrs",
                type=str,
                metavar="",
                nargs="+",
                default=[None],
                help="""Correlation index to plot. Can be a single integer or 
                comma separated integers e.g '0,2'. Defaults to all.""",
            )
            grp.add_argument(
                "--t0",
                dest="t0s",
                nargs="+",
                type=float,
                metavar="",
                default=[None],
                help="""Minimum time to plot [in seconds]. 
                Defaults to full range]""",
            )
            grp.add_argument(
                "--t1",
                dest="t1s",
                nargs="+",
                type=float,
                metavar="",
                default=[None],
                help="""Maximum time to plot [in seconds]. 
                Defaults to full range""",
            )
            grp.add_argument(
                "--taql",
                dest="taqls",
                nargs="+",
                type=str,
                metavar="",
                default=[None],
                help="TAQL where clause",
            )

        if "plot" in grp.title.lower():
            grp.add_argument(
                "-y",
                "--yaxis",
                "-d",
                "--doplot",
                dest="yaxes",
                nargs="+",
                type=str,
                metavar="",
                default=["ap"],
                help="""Plot complex values as any amplitude (a), phase (p), 
                real (r), imaginary (i). For a combination of multiple plots, 
                specify as a single string. e.g. To plot both amplitude and 
                phase, set this to 'ap'. To plot all at once, set this to 'all'. 
                Defaults to ap.""",
            )
            grp.add_argument(
                "-x",
                "--xaxis",
                dest="xaxes",
                nargs="+",
                type=str,
                metavar="",
                default=[None],
                help="""Choose an x-xaxis Valid choices are: time, antenna, 
                channel. If this is not supplied, an appropriate one will be 
                selected automatically depending on the type of gains being 
                plotted.""",
            )
            grp.add_argument(
                "-p",
                "--plotname",
                dest="image_names",
                nargs="+",
                type=str,
                metavar="",
                default=[None],
                help="""Static image name. The suffix of this name determines 
                the type of plot. If foo.png, the output will be PNG, else if 
                foo.svg, the output will be of the SVG format. PDF is also 
                accepable""",
            )
    return parser


def cubical_gains_parser():
    """Create command line arguments for cubical gains"""
    parser = RagParser(
        usage="%(prog)s [options] <value>",
        description="Radio Astronomy Gains Inspector: Cubical gain tables",
        parents=[gains_argparser()],
        add_help=False,
        conflict_handler="resolve",
    )

    for grp in parser._action_groups:
        if "selection" in grp.title.lower():
            grp.add_argument(
                "-c",
                "--corr",
                dest="corrs",
                type=str,
                metavar="",
                nargs="+",
                default=[None],
                help="""Correlation index to plot. Can be a single integer or 
                comma separated integers e.g '0,2'. Defaults to all.""",
            )
            
            # Overriding and Suppressing the following arguments as defined in
            # the parent parser, this is done using argparse.SUPPRESS
            grp.add_argument(
                "--t0",
                dest="t0s",
                nargs="+",
                type=float,
                metavar="",
                default=argparse.SUPPRESS,
                help=argparse.SUPPRESS,
            )
            grp.add_argument(
                "--t1",
                dest="t1s",
                nargs="+",
                type=float,
                metavar="",
                default=argparse.SUPPRESS,
                help=argparse.SUPPRESS,
            )
            grp.add_argument(
                "--taql",
                dest="taqls",
                nargs="+",
                type=str,
                metavar="",
                default=argparse.SUPPRESS,
                help=argparse.SUPPRESS,
            )

    return parser


def quartical_gains_parser():
    """Create command line arguments for Quartical gains"""
    parser = RagParser(
        usage="%(prog)s [options] <value>",
        description="Radio Astronomy Gains Inspector: Quartical gain tables",
        parents=[cubical_gains_parser()],
        add_help=False,
    )

    for grp in parser._action_groups:
        if "required" in grp.title.lower():
            grp.add_argument(
                "-gt",
                "--gtype",
                dest="gtypes",
                nargs="+",
                type=str,
                metavar="",
                default=[None],
                required=False,
                help="""Gain table type(s) being plotted. Multiple tables can 
                    be specified as a space separated list""",
            )
        if "selection" in grp.title.lower():
            grp.add_argument(
                "--scan",
                dest="scans",
                nargs="+",
                type=str,
                metavar="",
                default=[None],
                help="""Which scans to select, Same specification as for field selection
                """
            )

    return parser
