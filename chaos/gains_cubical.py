from itertools import product
import cubical.param_db as  db
import numpy as np

import dask.array as da

from bokeh.layouts import grid, gridplot, column, row
from bokeh.io import save, output_file

from holy_chaos.chaos.exceptions import InvalidCmap, InvalidColumnName, EmptyTable
from holy_chaos.chaos.arguments import gains_argparser
from holy_chaos.chaos.plotting import FigRag, Circle, Scatter
from holy_chaos.chaos.ragdata import Axargs
from holy_chaos.chaos.gains import get_colours
from holy_chaos.chaos.processing import Processor
from holy_chaos.chaos.widgets_cubical import make_widgets

from collections import namedtuple
from ipdb import set_trace

class TableData:
    def __init__(self, ms_name, ants=None, fields=None,
                 corr1s=None, corr2s=None):
        self.ms_name = ms_name
        self.ants = ants
        self.fields = fields
        self.corr1s = corr1s
        self.corr2s = corr2s
        self.active_corrs = []
        self.active_spws = [0]

    @property
    def num_corr1s(self):
        return len(self.corr1s)

    @property
    def num_corr2s(self):
        return len(self.corr2s)
    
    @property
    def num_ants(self):
        return len(self.ants)

    @property
    def num_fields(self):
        return len(self.fields)


def add_extra_xaxis(channels, figrag, sargs):
    chans = chans[[0, -1]] / 1e9
    figrag.add_axis(chans[0], chans[-1], "x", "linear", "Frequency GHz",
                    "above")


cb_name = "/home/lexya/Documents/test_gaintables/cubical/reduction02-cubical-G-field_0-ddid_None.parmdb"
# cb_name = "/home/lexya/Documents/test_gaintables/cubical/reduction02-cubical-BBC-field_0-ddid_None.parmdb"
table = db.load(cb_name)

syns = {("error" if "err" in _ else "gain"): _ for _ in table.names()}


gain_data = table[syns["gain"]]
gain_err = table[syns["error"]]

tdata = TableData(cb_name, ants=gain_data.grid[gain_data.ax.ant],
                           corr1s=gain_data.grid[gain_data.ax.corr1],
                           corr2s=gain_data.grid[gain_data.ax.corr2],
                        #    fields=gain_data.grid[gain_data.ax.field]
                        )

cmap = get_colours(tdata.num_ants)


yaxes = "a,p,r,i"
xaxis = "time"

all_figs = []
for yaxis in yaxes.split(","):
  
    print(f"Axis: {yaxis}")

    figrag = FigRag(add_toolbar=True,
                    x_scale=xaxis if xaxis == "time" else "linear")
    Axargs = namedtuple("Axargs", ["flags", "errors", "xaxis", "yaxis"])
  
    for ant, corr1, corr2 in product(tdata.ants[:3], tdata.corr1s, tdata.corr2s):
        print(f"Antenna: {ant}, corr: {corr1} {corr2}")

        masked_data, (time, freq) = gain_data.get_slice(ant=ant, corr1=corr1, corr2=corr2)
        masked_err = gain_err.get_slice(ant=ant, corr1=corr1, corr2=corr2)[0]

        if masked_data is None or corr1 != corr2:
            continue
        
        if f"{corr1}{corr2}" not in tdata.active_corrs:
            tdata.active_corrs.append(f"{corr1}{corr2}")


        masked_data, masked_err = masked_data.flatten(), masked_err.flatten()
        
        ax_info = Axargs(flags=~masked_data.mask, errors=masked_err.data,
                         xaxis=xaxis, yaxis=yaxis)
        
        xaxes = {
            "time": Processor.unix_timestamp(time),
            "channel": np.arange(freq.size)
            }
        data = {
            "x": xaxes[xaxis],
            "y": Processor(masked_data.data).calculate(yaxis),
            "data": ax_info
        }
        
        figrag.add_glyphs("circle", data=data, legend=ant,
                          fill_color=cmap[tdata.ants.index(ant)],
                          line_color=cmap[tdata.ants.index(ant)],
                          tags=[f"a{tdata.ants.index(ant)}",
                                f"c{corr1}{corr2}"], "s0")

    figrag.update_xlabel(ax_info.xaxis)
    figrag.update_ylabel(ax_info.yaxis)

    # if "chan" in ax_info.xaxis:
    #     add_extra_xaxis(msdata, figrag, sel_args)

    figrag.add_legends(group_size=16, visible=True)
    figrag.update_title(f"{ax_info.yaxis} vs {ax_info.xaxis}")
    figrag.show_glyphs(selection="b0")

    all_figs.append(figrag.fig)

output_file(filename="ghost.html")
widgets = make_widgets(tdata, all_figs[0], group_size=8)
save(column(row(widgets), *all_figs), filename="ghost.html", title="oster")
set_trace()
