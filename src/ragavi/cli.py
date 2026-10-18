"""Console entry points for Ragavi's gain and visibility plotters."""

from ragavi.arguments import gains_argparser, vis_argparser


def _catch_empty(options):
    """Convert empty command-line values to ``None``."""
    for key, value in vars(options).items():
        if value in ("", " "):
            setattr(options, key, None)
    return options


def gains():
    """Run the gain plotter from the command line."""
    options = _catch_empty(gains_argparser().parse_args())
    from ragavi.ragavi import main

    main(options=options)


def vis():
    """Run the visibility plotter from the command line."""
    options = _catch_empty(vis_argparser().parse_args())
    from ragavi.visibilities import main

    main(options=options)
