import os
from concurrent import futures
from functools import partial

import daskms as xm
import numpy as np
from dask import compute
from itertools import product, zip_longest
from bokeh.layouts import column, grid, gridplot, layout, row
from bokeh.io import output_file, output_notebook, save

from chaos import version
from chaos.arguments import gains_argparser
from chaos.exceptions import EmptyTable, InvalidCmap, InvalidColumnName
from chaos.lograg import logging, update_log_levels, update_logfile_name
from chaos.plotting import FigRag, Circle, Scatter
from chaos.processing import Chooser, Processor
from chaos.ragdata import (dataclass, field, Axargs, Genargs, MsData, Plotargs,
                     Selargs)
from chaos.utils import get_colours, timer
from chaos.widgets import F_MARKS, make_stats_table, make_table_name, make_widgets

snitch = logging.getLogger(__name__)

_GROUP_SIZE_ = 16
_NOTEBOOK_ = False
    
def get_table(msdata, sargs, group_data):
    # perform  TAQL selections and chan and corr selection
    super_taql = Chooser.form_taql_string(
        msdata, antennas=sargs.antennas, baselines=sargs.baselines,
        fields=sargs.fields, spws=sargs.ddids, taql=sargs.taql,
        time=(sargs.t0, sargs.t1))
    msdata.taql_selector = super_taql

    snitch.debug(f"Selection string: {super_taql}")
    mss = xm.xds_from_table(
        msdata.ms_name, taql_where=super_taql, group_cols=group_data,
        table_schema={"CPARAM": {"dims": ("chan", "corr")},
                                 "FLAG": {"dims": ("chan", "corr")},
                                 "FPARAM": {"dims": ("chan", "corr")},
                                 "PARAMERR": {"dims": ("chan", "corr")},
                                 "SNR": {"dims": ("chan", "corr")},
                                 })

    for i, sub in enumerate(mss):
        mss[i] = sub.sel(corr=sargs.corrs)
    return mss

def add_extra_xaxis(msdata, figrag, sargs):
    channels = msdata.freqs.sel(chan=sargs.channels)
    for sp in msdata.active_spws:
        if channels.ndim > 1:
            chans = channels.sel(row=sp)
        else:
            chans = channels
        chans = chans[[0, -1]].values/1e9
        figrag.add_axis(*chans, "x",
                        "linear", "Frequency GHz", "above")

def iron_data(axes):
    """Flatten these data to 1D"""
    def flat(att, obj): 
        if getattr(obj, att) is not None:
            return setattr(obj, att, getattr(obj, att).flatten())

    axes.xdata, axes.ydata, axes.flags,  axes.errors = compute(
        axes.xdata, axes.ydata, axes.flags, axes.errors)
   
    for dat in ["ydata", "flags", "errors"]:
        flat(dat, axes)
    axes.xdata = np.tile(axes.xdata, 
            axes.ydata.size//axes.xdata.size)
    return axes

def add_hover_data(fig, sub, msdata, data):
    """
    Add spw, field, scan, corr, antenna into main CDS for the hover tooltips

    Parameters
    ----------
    fig : :obj:`Plot`
        The figure itself
    sub: :obj:`xr.Dataset`
        Dataset contining data for  sub ms
    msdata: :obj:`MsData`
        Class containing data for all things MS
    data: :obj:`dict`
        Dictionary containing data for the plots, onto which htool 
        data will be added
    
    Returns
    -------
    Updated data dictionary (with htool data)
    """
    axes = data["data"]
    ant = msdata.reverse_ant_map[sub.ANTENNA1]
    field = msdata.reverse_field_map[sub.FIELD_ID]

    #format unix time in tooltip
    if axes.xaxis == "time":
        tip0 = (f"({axes.xaxis:.4}, {axes.yaxis:.4})",
                "(@x{%F %T}, @y)")
        # fig.select_one({"tags": "hover"}).formatters = {"@x": "datetime"}
        fig.tools[0].formatters = {"@x": "datetime"}
    else:
        tip0 = (f"({axes.xaxis:.4}, {axes.yaxis:.4})", f"(@x, @y)")

    # fig.select_one({"tags": "hover"}).tooltips 
    fig.tools[0].tooltips = [
        tip0,
        ("spw", "@spw"), ("field", "@field"), ("scan", "@scan"),
        ("ant", "@ant"), ("corr", "@corr"), ("% flagged", "@pcf")
    ]
    data.update({
        "spw": np.full(axes.xdata.shape, sub.SPECTRAL_WINDOW_ID, dtype="int8"),
        "field": np.full(axes.xdata.shape, field),
        "scan": np.tile(sub.SCAN_NUMBER.values,
                        axes.xdata.size//sub.row.size),
        "ant": np.full(axes.xdata.shape, ant, ),
        "corr": np.full(axes.xdata.shape, data["corr"], dtype="int8"),
        "pcf": np.full(axes.xdata.shape,
                        (sub.FLAG.sum()/sub.FLAG.size).values * 100)
    })
    return data

def calculate_points(subms, nsubs):
    return subms.FLAG.size * nsubs

def set_xaxis(gain):
    gains = {k: "channel"  for k in "b xf df".split()}
    gains.update({k: "time" for k in "d g f k kcross".split()})
    return gains.get(gain.lower(), "time")
    
def gen_plot(yaxis, xaxis, cmap, msdata, subs, selections, static_name):
    snitch.debug(f"Axis: {yaxis}")
    figrag = FigRag(add_toolbar=True, width=900, height=710,
                    x_scale=xaxis if xaxis == "time" else "linear",
                    plot_args={"frame_height": None, "frame_width": None})
    for it, (sub, corr) in enumerate(product(subs, msdata.active_corrs)):

        snitch.debug(f"spw: {sub.SPECTRAL_WINDOW_ID}, " +
              f"field: {msdata.reverse_field_map[sub.FIELD_ID]}, " +
              f"corr: {corr}, yaxis: {yaxis}")
        colour = cmap[msdata.active_antennas.index(
            msdata.reverse_ant_map[sub.ANTENNA1])]
        msdata.active_channels = msdata.freqs.sel(chan=selections.channels,
                                                  row=sub.SPECTRAL_WINDOW_ID)

        axes = Axargs(xaxis=xaxis, yaxis=yaxis, data_column="CPARAM",
                      msdata=msdata)
        axes.set_axis_data("xaxis", ms_obj=sub)
        axes.set_axis_data("yaxis", ms_obj=sub)
        axes.xdata = Processor(axes.xdata).calculate(axes.xaxis).data

        sel_corr = {"corr": corr} if "corr" in axes.ydata.dims else {}
        # setting ydata here for reinit. Otherwise, autoset in axargs
        axes.ydata = Processor(sub[axes.ydata_col]).calculate(
            axes.yaxis).sel(sel_corr).data

        axes.flags = ~sub.FLAG.sel(sel_corr).data
        axes.errors = sub.PARAMERR.sel(sel_corr).data
        
        axes = iron_data(axes)

        #pass inverted flags for the view as is
        #the view only shows true mask data
        data = {"x": axes.xdata, "y": axes.ydata, "data": axes, "corr": corr}
        data = add_hover_data(figrag.fig, sub, msdata, data)
            
        figrag.add_glyphs(F_MARKS[sub.FIELD_ID], data=data,
            legend=msdata.reverse_ant_map[sub.ANTENNA1], fill_color=colour,
            line_color=colour, tags=[f"a{sub.ANTENNA1}",
                f"s{sub.SPECTRAL_WINDOW_ID}", f"c{corr}", f"f{sub.FIELD_ID}"])

    figrag.update_xlabel(axes.xaxis)
    figrag.update_ylabel(axes.yaxis)

    if "chan" in axes.xaxis:
        add_extra_xaxis(msdata, figrag, selections)

    figrag.add_legends(group_size=_GROUP_SIZE_, visible=True)
    figrag.update_title(f"{axes.yaxis} vs {axes.xaxis}")
    figrag.show_glyphs(selection="b0")

    #attach this data column here to be collected
    figrag.data_column = axes.data_column
    return figrag

def main(parser, gargs=None):
    if gargs is None:
        ps = parser().parse_args()
    else:
        ps = parser().parse_args(gargs)

    if ps.debug:
        update_log_levels(snitch.parent, 10)
    update_logfile_name(snitch.parent,
                        ps.logfile if ps.logfile else "ragains.log")

    for (msname, antennas, channels, corrs, ddids, fields, t0, t1,
        taql, cmap, yaxes, xaxis, html_name, image_name) in zip_longest(
        ps.msnames, ps.antennas, ps.channels, ps.corrs, ps.ddids,
        ps.fields, ps.t0s, ps.t1s, ps.taqls, ps.cmaps, ps.yaxes, ps.xaxes,
        ps.html_names, ps.image_names):
        
        #we're grouping the arguments into 4
        generals = Genargs(msname=msname, version=version)
        selections = Selargs(
            antennas=antennas, corrs=Chooser.get_knife(corrs),
            baselines=None, channels=Chooser.get_knife(channels),
            ddids=ddids, fields=fields, taql=taql, t0=t0, t1=t1)

        #initialise data ssoc with ms
        msdata = MsData(msname)
        snitch.info(f"MS: {msname}")

        subs = get_table(msdata, selections, group_data=["SPECTRAL_WINDOW_ID", 
                                                        "FIELD_ID", "ANTENNA1"])
        if len(subs) > 0:
            if corrs is None:
                msdata.active_corrs = subs[0].corr.values
            else:
                if type(selections.corrs) == int:
                    msdata.active_corrs = [selections.corrs]
                else:
                    msdata.active_corrs = selections.corrs
        else:
            raise EmptyTable(
                "Table came up empty. Please your check selection.")
    
        for sub in subs:
            msdata.active_fields.append(sub.FIELD_ID)
            msdata.active_spws.append(sub.SPECTRAL_WINDOW_ID)
            msdata.active_antennas.append(
                        msdata.reverse_ant_map[sub.ANTENNA1])
        msdata.active_fields = sorted(list(set(msdata.active_fields)))
        msdata.active_spws = sorted(list(set(msdata.active_spws)))
        msdata.active_antennas = sorted(list(set(msdata.active_antennas)))
        
        if yaxes is None:
            yaxes = ["a", "p"]
        elif yaxes == "all":
            yaxes = [_ for _ in "apri"]
        elif set(yaxes).issubset(set("apri")):
            yaxes = [_ for _ in yaxes]

        if msdata.table_type.lower().startswith("k"):
            yaxes = ["delay"]
        
        if xaxis is None:
            xaxis = set_xaxis(msdata.table_type)

        if html_name is None and image_name is None:
            html_name = msdata.ms_name + f"_{''.join(yaxes)}" +".html"
        
        if cmap is None:
            cmap = "coolwarm"
        cmap = get_colours(len(msdata.active_antennas), cmap)
        points = calculate_points(subs[0], len(subs))

        if points > 30000 and image_name is None:
            image_name = msdata.ms_name + ".png"

        with futures.ThreadPoolExecutor() as executor:            
            all_figs = executor.map(partial(gen_plot, xaxis=xaxis, cmap=cmap,
                msdata=msdata, subs=subs, selections=selections,
                static_name=image_name), yaxes)

        all_figs = list(all_figs)
        
        if image_name:           
            statics = lambda func, _x, **kwargs: getattr(_x, func)(**kwargs)
            with futures.ThreadPoolExecutor() as executor:
                executor.map(
                    partial(statics, mdata=msdata, filename=image_name,
                        group_size=_GROUP_SIZE_),
                        *zip(*product(["write_out_static", "potato"], 
                    all_figs)))
                #generate all differnt combinations of all_figs and the name of
                #the static functions and then split them into individual lists
                # by unpacking the output of zip

        if html_name:
            data_column = all_figs[0].data_column
            all_figs[0].link_figures(*all_figs[1:])
            all_figs = [fig.fig for fig in all_figs]
            widgets = make_widgets(msdata, all_figs[0], group_size=_GROUP_SIZE_)
            stats = make_stats_table(msdata, data_column, yaxes,
                    get_table(msdata, selections,
                        group_data=["SPECTRAL_WINDOW_ID", "FIELD_ID"]))
            # Set up my layouts
            all_widgets = grid([widgets[0], column(widgets[1:]+[stats])],
                                sizing_mode="fixed", nrows=1)
            plots = gridplot([all_figs], toolbar_location="right",
                                sizing_mode="stretch_width")
            final_layout = layout([
                [make_table_name(generals.version, msdata.ms_name)],
                [all_widgets], [plots]], sizing_mode="stretch_width")
            
            if _NOTEBOOK_:
                return final_layout
        
            output_file(filename=html_name)
            save(final_layout, filename=html_name,
                title=os.path.splitext(os.path.basename(html_name))[0])
            snitch.info(f"HTML file at: {html_name}")
        snitch.info("Plotting Done")
    return 0

def plot_table(**kwargs):
    """
    Plot gain tables within Jupyter notebooks. Parameter names correspond
    to the long names of each argument (i.e those with --) from the
    `ragavi-vis` command line help

    Parameters
    ----------
    table : :obj:`str` or :obj:`list`
        The table (list of tables) to be plotted.

    ant : :obj:`str, optional`
        Plot only specific antennas, or comma-separated list of antennas.
    corr : :obj:`int, optional`
        Correlation index to plot. Can be a single integer or comma separated
        integers e.g "0,2". Defaults to all.
    cmap : `str, optional`
        Matplotlib colour map to use for antennas. Default is coolwarm
    ddid : :obj:`int`
        SPECTRAL_WINDOW_ID or ddid number. Defaults to all
    doplot : :obj:`str, optional`
        Plot complex values as amp and phase (ap) or real and imag (ri).
        Default is "ap".
    field : :obj:`str, optional`
        Field ID(s) / NAME(s) to plot. Can be specified as "0", "0,2,4",
        "0~3" (inclusive range), "0:3" (exclusive range), "3:" (from 3 to
        last) or using a field name or comma separated field names. Defaults
        to all.
    k-xaxis: :obj:`str`
        Choose the x-xaxis for the K table. Valid choices are: time or
        antenna. Defaults to time.
    taql: :obj:`str, optional`
        TAQL where clause
    t0  : :obj:`int, optional`
        Minimum time [in seconds] to plot. Default is full range
    t1 : :obj:`int, optional`
        Maximum time [in seconds] to plot. Default is full range
    """
    global _NOTEBOOK_
    _NOTEBOOK_ = True
    nargs = []

    #collect function arguments and pass them to parser
    for key, value in kwargs.items():
        nargs.append(f"--{key}")
        if isinstance(value, list):
            nargs.extend(value)
        else:
            nargs.append(value)

    output_notebook()
    main_layout = main(gains_argparser, nargs)
    show(main_layout)
    print("Notebook plots are ready")
    return 0

def console():
    """A console run entry point for setup.cfg"""
    main(gains_argparser)