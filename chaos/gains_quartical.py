import os
import numpy as np
import dask.array as da
import xarray as xr

from concurrent import futures
from bokeh.layouts import column, grid, gridplot, layout, row
from bokeh.io import save, output_file
from collections import namedtuple
from dask import compute
from daskms.experimental.zarr import xds_from_zarr
from functools import partial
from itertools import product, zip_longest

from chaos.arguments import gains_argparser
from chaos.exceptions import InvalidCmap, InvalidColumnName, EmptyTable
from chaos.gains import get_colours
from chaos.lograg import logging
from chaos.plotting import Circle, Scatter, FigRag
from chaos.processing import Chooser, Processor
from chaos.ragdata import Axargs, Selargs, stokes_types, dataclass, Genargs
from chaos.ragdata import QuarticalTableData as TableData
from chaos.utils import new_darray
from chaos.widgets import F_MARKS, make_widgets, make_stats_table, make_table_name

snitch = logging.getLogger(__name__)

_GROUP_SIZE_ = 16
_NOTEBOOK_ = False  

@dataclass
class Axes:
    xaxis: str
    yaxis: str
    flags: np.array = None
    errors: da.array = None

def calculate_points(subms, nsubs):
    return subms.gains.size * nsubs

def add_extra_xaxis(channels, figrag, sargs):
    channels = channels[sargs][[0, -1]] /1e9
    figrag.add_axis(*channels, "x", "linear", "Frequency GHz", "above")

def init_table_data(msname, sub_list):
    fields, scans, spws={}, [], []
    for ms in sub_list:
        fields[ms.FIELD_ID]=ms.FIELD_NAME
        scans.append(ms.SCAN_NUMBER)
        spws.append(ms.DATA_DESC_ID)
    fields = [fields[k] for k in sorted(fields.keys())]
    scans = sorted(np.unique(scans))
    spws = sorted(np.unique(spws))
    ants = sub_list[0].ant.values
    corrs = sub_list[0].corr.values

    return TableData(msname, ant_names=ants, corr_names=corrs,
        field_names=fields, spws=spws)

def populate_fig_data(subms, axes, cmap, figrag, msdata):
    time, freq = subms.gain_t.values, subms.gain_f.values / 1e9
    ants_corrs = product(msdata.active_antennas, msdata.active_corrs)
    for ant, corr in ants_corrs:
        sub = subms.sel(ant=ant, corr=corr)
        gains, sdict = sub.gains, {}

        snitch.info(f"Antenna {ant} corr: {ant}")
        if axes.xaxis == "time":
            original = ("gain_t", "gain_f", "dir")
        else:
            original = ("gain_f", "gain_t", "dir")
    
        gains = gains.transpose(*original)
        total_reps = gains.size
        data_reps = gains.size // freq.size

        xaxes={
            "time": np.repeat(Processor.unix_timestamp(time),
                                gains.size // time.size),
            "channel": np.repeat(np.arange(freq.size), gains.size // freq.size)
        }
        xaxes["chan"]=xaxes["freq"]=xaxes["frequency"] = xaxes["channel"]

        sdict["y"], sdict["field"], sdict["scan"], sdict["ddid"]= \
            compute(Processor(gains).calculate(axes.yaxis).data,
                    sub.fields.data.flatten(),
                    sub.scans.data.flatten(), sub.ddids.data.flatten(),
        )
        sdict["y"] = np.ravel(sdict["y"])
        sdict["x"] = xaxes[axes.xaxis]
        sdict["ant"] = np.repeat(ant, total_reps)
        # sdict["colors"] = np.repeat(cmap[ant], total_reps)
        sdict["corr"] = np.repeat(corr, total_reps)
        sdict["markers"] = np.repeat(F_MARKS[sub.FIELD_ID], total_reps)
        sdict["data"] = axes
        
        # TODO: change from ant to actual ant name

        # aidx: antenna item number in theactive antennas array
        aidx = msdata.active_antennas.tolist().index(ant)
        figrag.add_glyphs(
            F_MARKS[sub.FIELD_ID], data=sdict,
            legend=ant, 
            fill_color=cmap[aidx],
            line_color=cmap[aidx],
            tags=[f"a{msdata.ant_map[ant]}", f"c{msdata.corr_map[corr]}",
                  f"s{sub.DATA_DESC_ID}", f"f{sub.FIELD_ID}"])
    return figrag


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
        ("spw", "@ddid"), ("field", "@field"), ("scan", "@scan"),
        ("ant", "@ant"), ("corr", "@corr")
        ]
    return fig

def organise_table(ms, sels, tdata):
    """
    Group table into different ddids and fields
    """
    ms = sorted(ms, key=lambda x: x.SCAN_NUMBER)
    dd_fi = sorted(list({(m.DATA_DESC_ID, m.FIELD_ID) for m in ms}))
    variables = "gains fields scans ddids FLAG".split()

    if sels.fields is not None:
        fields = sels.fields.replace(" ", "").split(",")
        if all([f.isdigit() for f in fields]):
            sels.fields = [int(f) for f in fields]
        else:
            sels.fields = [tdata.field_maps[f] for f in fields]
    else:
        sels.fields = tdata.field_map.values()

    if sels.antennas is not None:
        antennas = sels.antennas.replace(" ", "").split(",")
        if all([a.isdigit() for a in antennas]):
            sels.antennas = [tdata.reverse_ant_map[int(a)] for a in antennas]
        else:
            sels.antennas = [a for a in antennas]
    else:
        sels.antennas = slice(None, None)


    new_order = []
    for (dd, fi) in dd_fi:
        sub_order = []
        for i, sub in enumerate(ms):
            if (sub.DATA_DESC_ID, sub.FIELD_ID) == (dd, fi) and fi in sels.fields:
                sub = sub.sel(corr=sels.corrs, ant=sels.antennas,
                             gain_f=sels.channels)
                sub["fields"] = new_darray(sub.gains, "fields", sub.FIELD_ID)
                sub["scans"] = new_darray(sub.gains, "scans", sub.SCAN_NUMBER)
                sub["ddids"] = new_darray(sub.gains, "ddids", sub.DATA_DESC_ID)
                sub["FLAG"] = new_darray(sub.gains, "flags", False)
                ms[i] = sub[variables]
                ms[i].attrs = {var: sub.attrs[var]
                               for var in ["DATA_DESC_ID", "FIELD_ID"]}
                sub_order.append(ms[i])
        new_order.append(xr.concat(sub_order, "gain_t",
                         combine_attrs="drop_conflicts"))
    

    tdata.active_fields = sels.fields
    tdata.active_antennas = new_order[0].ant.values
    tdata.active_corrs = new_order[0].corr.values
    # TODO: antenna slection needs to be fixed properly
    return new_order

def main(parser, gargs):
    ps = parser().parse_args(gargs)

    for (msname, antennas, channels, corrs, ddids, fields, cmap, yaxes,
         xaxis, html_name, image_name) in zip_longest(
            ps.msnames, ps.antennas, ps.channels, ps.corrs, ps.ddids, ps.fields,
            ps.cmaps, ps.yaxes, ps.xaxes, ps.html_names, ps.image_names):

        ms = xds_from_zarr(msname + "::G")
        msdata = init_table_data(msname + "G", ms)

        generals = Genargs(msname=msname, version="testwhatever")

        selections = Selargs(
            antennas=antennas, corrs=Chooser.get_knife(corrs),
            baselines=None, channels=Chooser.get_knife(channels),
            ddids=Chooser.get_knife(ddids))

        ms = organise_table(ms, selections, msdata)
        cmap = get_colours(ms[0].ant.size, cmap)
        points = calculate_points(ms[0], len(ms))

        if html_name is None and image_name is None:
            html_name = msdata.ms_name + f"_{''.join(yaxes)}" + ".html"

        if points > 30000 and image_name is None:
            snitch.info("This data contains more than 30K points. ")
            snitch.info("A png counterpart(s) of the plot will also be generated")
            snitch.info("because the HTML file generated may be slow to load.")
            snitch.info("Consider plotting antennas in groups for a better "+
                "interactive experience.")
            image_name = msdata.ms_name + ".png"
        
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

            for sub in ms:
                axes = Axes(flags=None, errors=None,
                    xaxis=xaxis, yaxis=yaxis)
                figrag = populate_fig_data(sub, axes=axes,
                    cmap=cmap, figrag=figrag, msdata=msdata)

            figrag.update_xlabel(axes.xaxis)
            figrag.update_ylabel(axes.yaxis)

            if "chan" in axes.xaxis:
                add_extra_xaxis(ms[0].gain_f.values, figrag, selections.channels)

            figrag.add_legends(group_size=_GROUP_SIZE_, visible=True)
            figrag.update_title(f"{axes.yaxis} vs {axes.xaxis}")
            figrag.show_glyphs(selection="b0")
            figrag = add_hover_data(figrag, axes)

            all_figs.append(figrag)
        
        if image_name:
            statics = lambda func, _x, **kwargs: getattr(_x, func)(**kwargs)
            with futures.ThreadPoolExecutor() as executor:
                stores = executor.map(
                    partial(statics, mdata=msdata, filename=image_name,
                            group_size=_GROUP_SIZE_),
                    *zip(*product(["write_out_static", "potato"], all_figs)))
                #generate all differnt combinations of all_figs and the name of
                #the static functions and then split them into individual lists
                # by unpacking the output of zip            
        if html_name:
            data_column = "gains"
            all_figs[0].link_figures(*all_figs[1:])
            all_figs = [fig.fig for fig in all_figs]
            widgets = make_widgets(
                msdata, all_figs[0], group_size=_GROUP_SIZE_)
            stats = make_stats_table(msdata, data_column, yaxes, ms)
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
            snitch.info(f"HTML at: {html_name}")
        snitch.info("Plotting Done")


if __name__ == "__main__":
    ms_name = "/home/lexya/Documents/test_gaintables/quartical/kgb_mad.qc"
    #synonyms for the the tables available here
    yaxes = "a"
    xaxis = "time"
    main(gains_argparser, ["-t", ms_name, "-y", yaxes, "-x", xaxis,
    "--cmap", "glasbey"])
