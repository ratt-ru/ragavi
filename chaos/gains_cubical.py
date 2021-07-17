import cubical.param_db as db
import dask.array as da
import numpy as np

from itertools import product, zip_longest
from collections import namedtuple
from bokeh.layouts import column, grid, gridplot, layout, row
from bokeh.io import output_file, save

from arguments import gains_argparser
from exceptions import EmptyTable, InvalidCmap, InvalidColumnName
from gains import get_colours
from lograg import logging, get_logger
from plotting import Circle, FigRag, Scatter
from processing import Chooser, Processor
from ragdata import Axargs, Genargs, Selargs
from widgets_cubical import make_table_name, make_widgets

snitch = get_logger(logging.getLogger(__name__))
_GROUP_SIZE_ = 16

class TableData:
    def __init__(self, ms_name, ants=None, fields=None, corr1s=None):
        self.ms_name = ms_name
        self.ant_names = ants
        self.fields = fields
        self.corr1s = [c.upper() for c in corr1s]
        self.active_corrs = []
        self.active_corr1s = None
        self.active_corr2s = None
        self.active_spws = [0]

    @property
    def ant_map(self):
        return {a: self.ant_names.index(a) for a in self.ant_names}

    @property
    def reverse_ant_map(self):
        return {v: k for k, v in self.ant_map.items()}

    @property
    def field_map(self):
        return {a: self.field_names.index(a) for a in self.field_names}

    @property
    def reverse_field_map(self):
        return {v: k for k, v in self.field_map.items()}

    @property
    def corr_names(self):
        return [f"{c1}{c2}".upper() for c1, c2 in product(self.corr1s, self.corr1s)]

    @property
    def corr_map(self):
        return {a: self.corr_names.index(a) for a in self.corr_names}

    @property
    def reverse_corr_map(self):
        return {v: k for k, v in self.corr_map.items()}

    @property
    def num_corr1s(self):
        return len(self.corr1s)
    
    @property
    def num_ants(self):
        return len(self.ant_names)

    @property
    def num_fields(self):
        return len(self.fields)


def add_extra_xaxis(channels, figrag, sargs):
    chans = chans[[0, -1]] / 1e9
    figrag.add_axis(*chans, "x", "linear", "Frequency GHz",
                    "above")


def organise_data(sels, tdata):
    if sels.antennas is not None:
        antennas = sels.antennas.replace(" ", "").split(",")
        if all([a.isdigit() for a in antennas]):
            sels.antennas = [int(a) for a in antennas]
        else:
            sels.antennas = [tdata.ant_map[a] for a in antennas]
    else:
        sels.antennas = list(tdata.ant_map.values())
    tdata.active_ants = sels.antennas
    
    if sels.corrs is not None:
        # change to xx etc labels and then translate from corr1s
        corrs = sels.corrs.replace(" ", "").upper().split(",")
        if all([c.isdigit() for c in corrs]):
            corrs = [tdata.reverse_corr_map[int(c)] for c in corrs]
        else:
            corrs = [c for c in corrs if c in tdata.corr_map]
        tdata.active_corr1s, tdata.active_corr2s = (
            [tdata.corr1s.index(c[0]) for c in corrs],
            [tdata.corr1s.index(c[1]) for c in corrs])
    else:
        tdata.active_corr1s, tdata.active_corr2s = (
            [*range(len(tdata.corr1s))], [*range(len(tdata.corr1s))])
    
    return tdata

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

        # initialise table data container
        tdata = TableData(cb_name, ants=gain_data.grid[gain_data.ax.ant],
                        corr1s=gain_data.grid[gain_data.ax.corr1],
                        # corr2s=gain_data.grid[gain_data.ax.corr2],
                        #    fields=gain_data.grid[gain_data.ax.field]
                        )
        generals = Genargs(msname=msname, version="testwhatever")
        selections = Selargs(
            antennas=antennas, corrs=corrs,
            baselines=None, channels=Chooser.get_knife(channels),
            ddids=None)

        tdata = organise_data(selections, tdata)
        cmap = get_colours(len(tdata.active_ants))
        
        if yaxes is None:
            yaxes = ["a", "p"]
        elif yaxes == "all":
            yaxes = [_ for _ in "apri"]
        elif set(yaxes).issubset(set("apri")):
            yaxes = [_ for _ in yaxes]
        yaxes = [Axargs.translate_y(_) for _ in yaxes]

        all_figs = []
        for yaxis in yaxes:
            snitch.info(f"Axis: {yaxis}")
            figrag = FigRag(
                add_toolbar=True, width=900, height=710,
                x_scale=xaxis if xaxis == "time" else "linear",
                plot_args={"frame_height": None, "frame_width": None})

            Axes = namedtuple("Axes", ["flags", "errors", "xaxis", "yaxis"])
        
            for ant, corr1, corr2 in product(tdata.active_ants,
                tdata.active_corr1s, tdata.active_corr2s):

                _corr = f"{tdata.corr1s[corr1]}{tdata.corr1s[corr2]}"
                snitch.info(f"Antenna: {ant}, corr: {_corr}")

                masked_data, (time, freq) = gain_data.get_slice(ant=ant,
                                                    corr1=corr1, corr2=corr2)
                masked_err = gain_err.get_slice(ant=ant, corr1=corr1, corr2=corr2)[0]

                if masked_data is None:
                    continue

                if _corr not in tdata.active_corrs:
                    tdata.active_corrs.append(f"{_corr}")

                masked_data, masked_err = masked_data.flatten(), masked_err.flatten()
                axes = Axes(flags=~masked_data.mask, errors=masked_err.data,
                                xaxis=xaxis, yaxis=yaxis)
            
                xaxes = {
                    "time": Processor.unix_timestamp(time),
                    "channel": np.arange(freq.size)
                    }
                data = {
                    "x": xaxes[xaxis],
                    "y": Processor(masked_data.data).calculate(yaxis),
                    "data": axes
                }
                
                figrag.add_glyphs("circle", data=data, 
                    legend=tdata.reverse_ant_map[ant], fill_color=cmap[ant],
                    line_color=cmap[ant],
                    tags=[f"a{ant}", "s0", f"c{_corr}"])

            figrag.update_xlabel(axes.xaxis)
            figrag.update_ylabel(axes.yaxis)

            # if "chan" in axes.xaxis:
            #     add_extra_xaxis(msdata, figrag, sel_args)

            figrag.add_legends(group_size=_GROUP_SIZE_, visible=True)
            figrag.update_title(f"{axes.yaxis} vs {axes.xaxis}")
            figrag.show_glyphs(selection="b0")

            all_figs.append(figrag.fig)

        widgets = make_widgets(tdata, all_figs[0], group_size=_GROUP_SIZE_)
        all_widgets = grid([widgets[0], column(widgets[1:]+[])],
                           sizing_mode="fixed", nrows=1)
        plots = gridplot([all_figs], toolbar_location="right",
                         sizing_mode="stretch_width")
        final_layout = layout([
            [make_table_name(generals.version, tdata.ms_name)],
            [all_widgets], [plots]], sizing_mode="stretch_width")

        output_file(filename="ghost.html")
        save(final_layout, filename="ghost.html", title="oster")


if __name__ == "__main__":
    cb_name = "/home/lexya/Documents/test_gaintables/cubical/reduction02-cubical-G-field_0-ddid_None.parmdb"
    # cb_name = "/home/lexya/Documents/test_gaintables/cubical/reduction02-cubical-BBC-field_0-ddid_None.parmdb"
    #synonyms for the the tables available here

    yaxes = "a"
    xaxis = "time"
    main(gains_argparser, ["-t", cb_name, "-y", yaxes, "-x", xaxis, "--ant",
                           "0", "--corr", "0,1"])
