import os
import cubical.param_db as db
import dask.array as da
import numpy as np

from concurrent import futures
from functools import partial
from itertools import product, zip_longest
from bokeh.layouts import column, grid, gridplot, layout, row
from bokeh.io import output_file, save

from ragavi import version
from ragavi.arguments import cubical_gains_parser
from ragavi.exceptions import EmptyTable, InvalidCmap, InvalidColumnName
from ragavi.gains import get_colours
from ragavi.lograg import logging, update_log_levels, update_logfile_name
from ragavi.plotting import Circle, FigRag, Scatter
from ragavi.processing import Chooser, Processor
from ragavi.ragdata import Axargs, Genargs, Selargs
from ragavi.ragdata import CubicalTableData as TableData
from ragavi.widgets_cubical import make_table_name, make_widgets
from ragavi.widgets import F_MARKS

snitch = logging.getLogger(__name__)
_GROUP_SIZE_ = 16
_NOTEBOOK_ = False

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
    _, actives = Chooser.form_taql_string(
        tdata, antennas=sels.antennas, baselines=sels.baselines,
        fields=sels.fields, spws=sels.ddids, return_ids=True)
    if sels.antennas is not None:
       sels.antennas = actives.get("antennas")
    else:
        sels.antennas = list(tdata.reverse_ant_map.keys())
    tdata.active_antennas = sels.antennas
    
    if sels.corrs is not None:
        # get in string format for conversion to xx,xy,yx,yy format
        if any([char in sels.corrs for char in ": ~".split()]):
            sels.corrs = Chooser.get_knife(sels.corrs, get_slicer=False)
        elif sels.corrs.startswith("diag"):
            sels.corrs = ",".join([*map(tdata.reverse_corr_map.get, [0,3])])
        elif sels.corrs.startswith("off-"):
            sels.corrs = ",".join([*map(tdata.reverse_corr_map.get, [1,2])])
    
        # change to xx etc labels and then translate from corr1s
        # Doing this coz data stored as corr1:[0,1] and corr2:[0,1]
        # so xx will be corr1[0]corr2[0]
        corrs = sels.corrs.replace(" ", "").upper().split(",")
        if all([c.isdigit() for c in corrs]):
            corrs = [tdata.reverse_corr_map[int(c)] for c in corrs]
        else:
            corrs = [c for c in corrs if c in tdata.corr_map]
        # this list will contain tuples of the form (corr1, corr2)
        sels.corrs = [(tdata.corr1s.index(c[0]), tdata.corr1s.index(c[1]))
                    for c in corrs]
    else:
        # this list will contain tuples of the form (corr1, corr2)
        sels.corrs = [*product(range(len(tdata.corr1s)), repeat=2)]
        # select only diagonals mm[0] and nn[-1]
        sels.corrs = [sels.corrs[0], sels.corrs[-1]]
    tdata.corrs = sels.corrs
    return tdata

def set_xaxis(gain):
    gains = {k: "channel"  for k in "b xf df".split()}
    gains.update({k: "time" for k in "d g f k kcross".split()})
    return gains.get(gain.lower(), "time")


def main(parser, gargs=None):
    if gargs is None:
        ps = parser().parse_args()
    else:
        ps = parser().parse_args(gargs)

    if ps.debug:
        update_log_levels(snitch.parent, 10)
    update_logfile_name(
        snitch.parent,
        ps.logfile if ps.logfile else "ragains-cubical.log")

    for (msname, antennas, channels, corrs, fields, cmap, yaxes,
        xaxis, html_name, image_name) in zip_longest(ps.msnames, ps.antennas,
        ps.channels, ps.corrs, ps.fields, ps.cmaps, ps.yaxes,
        ps.xaxes, ps.html_names, ps.image_names):
        
        table = db.load(msname)
        snitch.info(f"Reading {msname}")

        # get table type and appropriate data location names
        syns = {("error" if "err" in _ else "gain"): _ for _ in table.names()}
        
        # get required data
        gain_data, gain_err = table[syns["gain"]], table[syns["error"]]

        # initialise table data container
        # This is done after reading the table cause data isn't contained elsewhere
        tdata = TableData(msname, ants=gain_data.grid[gain_data.ax.ant],
                        corr1s=gain_data.grid[gain_data.ax.corr1],
                        # corr2s=gain_data.grid[gain_data.ax.corr2],
                        fields=gain_data.grid[gain_data.ax.dir],
                        table_type=table.names()[0][0], colnames=""
                        )
        
        # set active fields by default to whatever is is in direction
        tdata.active_fields = gain_data.grid[gain_data.ax.dir]      
        generals = Genargs(msname=msname, version=version)
        selections = Selargs(
            antennas=antennas, corrs=corrs,baselines=None,
            channels=Chooser.get_knife(channels), ddids=None)

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
        if tdata.table_type.lower().startswith("k"):
            yaxes = ["delay"]

        if xaxis is None:
            xaxis = set_xaxis(tdata.table_type)

        all_figs = []
        for yaxis in yaxes:
            snitch.info(f"Axis: {yaxis}")
            figrag = FigRag(
                add_toolbar=True, width=900, height=710,
                x_scale=xaxis if xaxis == "time" else "linear",
                plot_args={"frame_height": None, "frame_width": None})
        
            for fid, ant, (corr1, corr2) in product(
                tdata.active_fields, tdata.active_antennas, tdata.corrs):

                _corr = f"{tdata.corr1s[corr1]}{tdata.corr1s[corr2]}"
                snitch.info(f"Antenna: {tdata.reverse_ant_map[ant]}, " +
                            f"Dir: {fid} " + f"corr: {_corr}")

                masked_data, (time, freq) = gain_data.get_slice(
                    ant=ant, corr1=corr1, corr2=corr2, dir=fid)
                
                # Select item 0 cause tuple containin array is returned
                masked_err = gain_err.get_slice(
                    ant=ant, corr1=corr1, corr2=corr2, dir=fid)[0]

                if masked_data is None:
                    # remove antenna, messes up with plotting fn
                    if ant in tdata.active_antennas:
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
               
                axes = Axargs(xaxis=xaxis, yaxis=yaxis, data_column=None,
                             msdata=tdata, flags=~masked_data.mask, 
                             errors=masked_err.data)   
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
                    "field": np.repeat(fid, total_reps),
                    "data": axes
                }
                colour = cmap[tdata.active_antennas.index(ant)]
                figrag.add_glyphs(F_MARKS[fid], data=data, 
                    legend=tdata.reverse_ant_map[ant], fill_color=colour,
                    line_color=colour,
                    tags=[f"a{ant}", "s0", f"c{_corr}", f"f{fid}"])

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
            len(tdata.active_antennas)*len(tdata.corrs), total_reps)
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
            snitch.info(f"HTML file at: {html_name}")
        snitch.info("Plotting Done")

def console():
    """A console run entry point for setup.cfg"""
    main(cubical_gains_parser)

if __name__ == "__main__":
    console()