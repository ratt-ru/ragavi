import sys
import glob
import pylab
import numpy as np

import matplotlib.cm as cmx
from pyrap.tables import table
from optparse import OptionParser
import matplotlib.colors as colors

from bokeh.plotting import figure
from bokeh.models.widgets import Div
from bokeh.layouts import row, column, gridplot, widgetbox
from bokeh.io import output_file, show, output_notebook, export_svgs
from bokeh.models import (Range1d, HoverTool, ColumnDataSource, LinearAxis,
                          FixedTicker, Legend, Toggle, CustomJS, Title,
                          CheckboxGroup, Select, Text)


def get_argparser():
    """Get argument parser"""
    parser = OptionParser(usage='%prog [options] tablename')
    parser.add_option('-f', '--field', dest='field',
                      help='Field ID to plot (default = 0)', default=0)
    parser.add_option('-d', '--doplot', dest='doplot',
                      help='Plot complex values as amp and phase (ap)'
                      'or real and imag (ri) (default = ap)', default='ap')
    parser.add_option('-a', '--ant', dest='plotants',
                      help='Plot only this antenna, or comma-separated list of antennas',
                      default=[-1])
    parser.add_option('-c', '--corr', dest='corr',
                      help='Correlation index to plot (usually just 0 or 1, default = 0)',
                      default=0)
    parser.add_option('--t0', dest='t0',
                      help='Minimum time to plot (default = full range)', default=-1)
    parser.add_option('--t1', dest='t1',
                      help='Maximum time to plot (default = full range)', default=-1)
    parser.add_option('--yu0', dest='yu0',
                      help='Minimum y-value to plot for upper panel (default=full range)',
                      default=-1)
    parser.add_option('--yu1', dest='yu1',
                      help='Maximum y-value to plot for upper panel (default=full range)',
                      default=-1)
    parser.add_option('--yl0', dest='yl0',
                      help='Minimum y-value to plot for lower panel (default=full range)',
                      default=-1)
    parser.add_option('--yl1', dest='yl1',
                      help='Maximum y-value to plot for lower panel (default=full range)',
                      default=-1)
    parser.add_option('--cmap', dest='mycmap',
                      help='Matplotlib colour map to use for antennas (default=coolwarm)',
                      default='coolwarm')
    parser.add_option('--ms', dest='myms',
                      help='Measurement Set to consult for proper antenna names',
                      default='')
    parser.add_option('-p', '--plotname', dest='pngname',
                      help='Output PNG name(default = something sensible)', default='')

    return parser


def main():
    """Main function"""
    parser = get_argparser()
    args = parser.parse_args()
