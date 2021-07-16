from itertools import product
import cubical.param_db as  db
import numpy as np

import dask.array as da

from bokeh.layouts import grid, gridplot, column, row
from bokeh.io import save, output_file

from exceptions import InvalidCmap, InvalidColumnName, EmptyTable
from arguments import gains_argparser
from plotting import FigRag, Circle, Scatter
from ragdata import Axargs
from gains import get_colours
from processing import Processor
from widgets_cubical import make_widgets

from collections import namedtuple

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
    figrag.add_axis(*chans, "x", "linear", "Frequency GHz",
                    "above")


def main(parser, gargs):
    ps = parser().parse_args(gargs)

    for (msname, antennas, channels, corrs, fields, t0, t1, cmap, yaxes,
        xaxis, html_name, image_name) in zip_longest(ps.msnames, ps.antennas,
        ps.channels, ps.corrs, ps.fields, ps.t0s, ps.t1s, ps.cmaps, ps.yaxes,
        ps.xaxes, ps.html_names, ps.image_names):
        
        table = db.load(msname)
        syns = {("error" if "err" in _ else "gain"): _ for _ in table.names()}

        gain_data = table[syns["gain"]]
        gain_err = table[syns["error"]]

        tdata = TableData(cb_name, ants=gain_data.grid[gain_data.ax.ant],
                        corr1s=gain_data.grid[gain_data.ax.corr1],
                        corr2s=gain_data.grid[gain_data.ax.corr2],
                        #    fields=gain_data.grid[gain_data.ax.field]
                        )

        cmap = get_colours(tdata.num_ants)
        
        if yaxes is None:
            yaxes = ["a", "p"]
        elif yaxes == "all":
            yaxes = [_ for _ in "apri"]
        elif set(yaxes).issubset(set("apri")):
            yaxes = [_ for _ in yaxes]
        yaxes = [Axargs.translate_y(_) for _ in yaxes]

        all_figs = []
        for yaxis in yaxes.split(","):
        
            print(f"Axis: {yaxis}")

            figrag = FigRag(add_toolbar=True,
                            x_scale=xaxis if xaxis == "time" else "linear")
            Axargs = namedtuple("Axargs", ["flags", "errors", "xaxis", "yaxis"])
        
            for ant, corr1, corr2 in product(tdata.ants[:3], tdata.corr1s,
                tdata.corr2s):
                print(f"Antenna: {ant}, corr: {corr1} {corr2}")

                masked_data, (time, freq) = gain_data.get_slice(ant=ant,
                                                    corr1=corr1, corr2=corr2)
                masked_err = gain_err.get_slice(ant=ant, corr1=corr1, corr2=corr2)[0]

                if masked_data is None or corr1 != corr2:
                    continue
                
                if f"{corr1}{corr2}" not in tdata.active_corrs:
                    tdata.active_corrs.append(f"{corr1}{corr2}")


                masked_data, masked_err = masked_data.flatten(), masked_err.flatten()
                
                axes = Axargs(flags=~masked_data.mask, errors=masked_err.data,
                                xaxis=xaxis, yaxis=yaxis)
               # TODO: are this necessary, if not delete. Lookgin superfluous
                axes.set_axis_data("xaxis")
                axes.set_axis_data("yaxis")
                xaxes = {
                    "time": Processor.unix_timestamp(time),
                    "channel": np.arange(freq.size)
                    }
                data = {
                    "x": xaxes[xaxis],
                    "y": Processor(masked_data.data).calculate(yaxis),
                    "data": axes
                }
                
                figrag.add_glyphs("circle", data=data, legend=ant,
                                fill_color=cmap[tdata.ants.index(ant)],
                                line_color=cmap[tdata.ants.index(ant)],
                                tags=[f"a{tdata.ants.index(ant)}",
                                        f"c{corr1}{corr2}"], "s0")

            figrag.update_xlabel(axes.xaxis)
            figrag.update_ylabel(axes.yaxis)

            # if "chan" in axes.xaxis:
            #     add_extra_xaxis(msdata, figrag, sel_args)

            figrag.add_legends(group_size=16, visible=True)
            figrag.update_title(f"{axes.yaxis} vs {axes.xaxis}")
            figrag.show_glyphs(selection="b0")

            all_figs.append(figrag.fig)

        output_file(filename="ghost.html")
        widgets = make_widgets(tdata, all_figs[0], group_size=8)
        save(column(row(widgets), *all_figs), filename="ghost.html", title="oster")





if __name__ == "__main__":
    cb_name = "/home/lexya/Documents/test_gaintables/cubical/reduction02-cubical-G-field_0-ddid_None.parmdb"
    # cb_name = "/home/lexya/Documents/test_gaintables/cubical/reduction02-cubical-BBC-field_0-ddid_None.parmdb"


    #synonyms for the the tables available here

    yaxes = "a,p,r,i"
    xaxis = "time"
