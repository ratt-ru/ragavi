import os
import cubical.param_db as db
import dask.array as da
import numpy as np

from concurrent import futures
from functools import partial
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
_NOTEBOOK_ = False

class TableData:
    def __init__(self, ms_name, ants=None, fields=None, corr1s=None):
        self.ms_name = ms_name
        self.ant_names = ants
        self.field_names = [str(f) for f in fields]
        self.corr1s = [c.upper() for c in corr1s]
        self.active_antennas = None
        self.active_corrs = []
        self.active_corr1s = None
        self.active_corr2s = None
        self.active_fields = None
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


def add_extra_xaxis(chans, figrag, sargs):
    chans = chans[[0, -1]] / 1e9
    figrag.add_axis(*chans, "x", "linear", "Frequency GHz",
                    "above")

def add_hover_data(fig, axes):
    """
    Add spw, field, scan, corr, antenna into the hover tooltips

    Parameters
    ----------
    fig : :obj:`Plot`
        The figure itself
    """
    #format unix time in tooltip
    if axes.xaxis == "time":
        tip0 = (f"({axes.xaxis:.4}, {axes.yaxis:.4})",
                "(@x{%F %T}, @y)")
        fig.fig.tools[0].formatters = {"@x": "datetime"}
    else:
        tip0 = (f"({axes.xaxis:.4}, {axes.yaxis:.4})", f"(@x, @y)")

    # fig.select_one({"tags": "hover"}).tooltips
    fig.fig.tools[0].tooltips = [
        tip0,
        ("field", "@field"), ("ant", "@ant"), ("corr", "@corr")
    ]
    return fig

def calculate_points(iters, size_per_iter):
    return iters * size_per_iter

def organise_data(sels, tdata):
    if sels.antennas is not None:
        antennas = sels.antennas.replace(" ", "").split(",")
        if all([a.isdigit() for a in antennas]):
            sels.antennas = [int(a) for a in antennas]
        else:
            sels.antennas = [tdata.ant_map[a] for a in antennas]
    else:
        sels.antennas = list(tdata.ant_map.values())
    tdata.active_antennas = sels.antennas
    
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
                        fields=gain_data.grid[gain_data.ax.dir]
                        )
        
        # set active fields by default to whatever is is in direction
        tdata.active_fields = gain_data.grid[gain_data.ax.dir]      
        generals = Genargs(msname=msname, version="testwhatever")
        selections = Selargs(
            antennas=antennas, corrs=corrs,
            baselines=None, channels=Chooser.get_knife(channels),
            ddids=None)

        if html_name is None and image_name is None:
            html_name = tdata.ms_name + f"_{''.join(yaxes)}" + ".html"
        
        tdata = organise_data(selections, tdata)
        cmap = get_colours(len(tdata.active_antennas), cmap)
        
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
        
            for ant, corr1, corr2 in product(tdata.active_antennas,
                tdata.active_corr1s, tdata.active_corr2s):

                _corr = f"{tdata.corr1s[corr1]}{tdata.corr1s[corr2]}"
                snitch.info(
                    f"Antenna: {tdata.reverse_ant_map[ant]}, corr: {_corr}")

                masked_data, (time, freq) = gain_data.get_slice(
                    ant=ant, corr1=corr1, corr2=corr2)
                
                # Select item 0 cause tuple containin array is returned
                masked_err = gain_err.get_slice(
                    ant=ant, corr1=corr1,corr2=corr2)[0]

                if masked_data is None:
                    # remove antenna, messes up with plotting fn
                    tdata.active_antennas.remove(ant)
                    snitch.info(f"Antenna {tdata.reverse_ant_map[ant]} " +
                            f"corr {_corr} contains no data.")
                    continue
                else:
                    # perform channel selection
                    masked_data = masked_data[:,selections.channels]
                    masked_err = masked_err[:,selections.channels]
                    if xaxis != "time":
                        masked_data, masked_err.T = masked_data.T, masked_err.T

                if _corr not in tdata.active_corrs:
                    tdata.active_corrs.append(f"{_corr}")

                masked_data, masked_err = masked_data.flatten(), masked_err.flatten()
                axes = Axes(flags=~masked_data.mask, errors=masked_err.data,
                            xaxis=xaxis, yaxis=yaxis)
                
                total_reps = masked_data.size

                xaxes = {
                    "time": np.repeat(Processor.unix_timestamp(time),
                                        total_reps//time.size),
                    "channel": np.repeat(np.arange(freq.size),
                                            total_reps//freq.size)
                    }
                data = {
                    "x": xaxes[xaxis],
                    "y": Processor(masked_data.data).calculate(yaxis),
                    "ant": np.repeat(tdata.reverse_ant_map[ant], total_reps),
                    "corr": np.repeat(_corr, total_reps),
                    "field": np.repeat(0, total_reps),
                    "data": axes
                }
                figrag.add_glyphs("circle", data=data, 
                    legend=tdata.reverse_ant_map[ant], fill_color=cmap[ant],
                    line_color=cmap[ant],
                    tags=[f"a{ant}", "s0", f"c{_corr}", f"f{0}"])

            figrag.update_xlabel(axes.xaxis)
            figrag.update_ylabel(axes.yaxis)

            if "chan" in axes.xaxis:
                add_extra_xaxis(freqs, figrag, selections)

            figrag.add_legends(group_size=_GROUP_SIZE_, visible=True)
            figrag.update_title(f"{axes.yaxis} vs {axes.xaxis}")
            figrag.show_glyphs(selection="b0")
            figrag = add_hover_data(figrag, axes)

            all_figs.append(figrag)

        points = calculate_points(
            len(tdata.active_antennas)*len(tdata.active_corr1s)*
            len(tdata.active_corr2s), total_reps)
        if points > 30000 and image_name is None:
            image_name = tdata.ms_name + ".png"
        if image_name:
            statics = lambda func, _x, **kwargs: getattr(_x, func)(**kwargs)
            with futures.ThreadPoolExecutor() as executor:
                stores = executor.map(
                    partial(statics, mdata=tdata, filename=image_name,
                            group_size=_GROUP_SIZE_),
                    *zip(*product(["write_out_static", "potato"], all_figs)))
        if html_name:
            data_column = "gains"
            all_figs[0].link_figures(*all_figs[1:])
            all_figs = [fig.fig for fig in all_figs]
            widgets = make_widgets(tdata, all_figs[0], group_size=_GROUP_SIZE_)
            all_widgets = grid([widgets[0], column(widgets[1:]+[])],
                            sizing_mode="fixed", nrows=1)
            plots = gridplot([all_figs], toolbar_location="right",
                            sizing_mode="stretch_width")
            final_layout = layout([
                [make_table_name(generals.version, tdata.ms_name)],
                [all_widgets], [plots]], sizing_mode="stretch_width")
            if _NOTEBOOK_:
                return final_layout
            output_file(filename=html_name)
            save(final_layout, filename=html_name,
                 title=os.path.splitext(os.path.basename(html_name))[0])
        snitch.info("Plotting Done")


if __name__ == "__main__":
    cb_name = "/home/lexya/Documents/test_gaintables/cubical/reduction02-cubical-G-field_0-ddid_None.parmdb"
    # cb_name = "/home/lexya/Documents/test_gaintables/cubical/reduction02-cubical-BBC-field_0-ddid_None.parmdb"
    #synonyms for the the tables available here

    yaxes = "a"
    xaxis = "time"
    main(gains_argparser, ["-t", cb_name, "-y", yaxes, "-x", xaxis, "--corr", "0",
        "--cmap", "glasbey"
                        #    "--ant", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"
                        ])
