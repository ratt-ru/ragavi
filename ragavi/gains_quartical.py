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

from ragavi import version
from ragavi.arguments import quartical_gains_parser
from ragavi.exceptions import InvalidCmap, InvalidColumnName, EmptyTable
from ragavi.gains import get_colours
from ragavi.lograg import logging, update_log_levels, update_logfile_name
from ragavi.plotting import Circle, Scatter, FigRag
from ragavi.processing import Chooser, Processor
from ragavi.ragdata import Axargs, Selargs, stokes_types, dataclass, Genargs
from ragavi.ragdata import QuarticalTableData as TableData
from ragavi.utils import new_darray, update_output_dir
from ragavi.widgets import F_MARKS, make_widgets, make_stats_table, make_table_name


snitch = logging.getLogger(__name__)

_GROUP_SIZE_ = 16
_NOTEBOOK_ = False  


def calculate_points(subms, nsubs):
    return subms.gains.size * nsubs

def add_extra_xaxis(channels, figrag, sargs):
    channels = channels[sargs][[0, -1]] /1e9
    figrag.add_axis(*channels, "x", "linear", "Frequency GHz", "above")

def init_table_data(msname, sub_list):
    snitch.debug("Initialising table data container")
    fields, scans, spws={}, [], []
    for sub_id, ms in enumerate(sub_list):
        fields[ms.FIELD_ID]=ms.FIELD_NAME
        spws.append(ms.DATA_DESC_ID)
        if "SCAN_NUMBER" not in sub_list[0].attrs:
            snitch.warning("SCAN_NUMBER column was not found")
            snitch.warning(f"assigning arbitrary scan ID {sub_id}")
            ms.attrs.update(SCAN_NUMBER=0)
        scans.append(ms.SCAN_NUMBER)

    fields = [fields[k] for k in sorted(fields.keys())]
    scans = sorted(np.unique(scans))
    spws = sorted(np.unique(spws))
    ants = sub_list[0].ant.values
    corrs = sub_list[0].corr.values

    return TableData(msname, ant_names=ants, corr_names=corrs,
        field_names=fields, spws=spws, scans=scans)

def populate_fig_data(subms, axes, cmap, figrag, msdata):
    time, freq = subms.gain_t.values, subms.gain_f.values / 1e9
    ants_corrs = product(msdata.active_antennas, msdata.active_corrs)
    for ant, corr in ants_corrs:
        snitch.info(f"Antenna {ant}, corr: {corr}")
        
        sub = subms.sel(ant=ant, corr=corr)
        gains, sdict = sub.gains, {}
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
    # sort sub mss by scans
    ms = sorted(ms, key=lambda x: x.SCAN_NUMBER)

    # get ddid and field ids. I use these for the selection of data
    dd_fi = sorted(list({(m.DATA_DESC_ID, m.FIELD_NAME, m.SCAN_NUMBER) for m in ms}))
    variables = "gains fields scans ddids FLAG".split()

    _, actives = Chooser.form_taql_string(
        tdata, antennas=sels.antennas, baselines=sels.baselines,
        fields=sels.fields, spws=sels.ddids, scans=sels.scans,
        taql=sels.taql, return_ids=True)

    if sels.fields is not None:
        sels.fields = [tdata.reverse_field_map[_]
                        for _ in actives.get("fields")]
    else:
        sels.fields = list(tdata.field_map.keys())

    if sels.antennas is not None:
       sels.antennas = [tdata.reverse_ant_map[_] 
                         for _ in actives.get("antennas")]
    else:
        sels.antennas = list(tdata.ant_map.keys())

    if sels.ddids is not None:
        sels.ddids = actives.get("spws")
    else:
        sels.ddids = tdata.spws.values

    if sels.scans is not None:
        sels.scans = actives.get("scans")
    else:
        sels.scans = tdata.scans

    if sels.corrs is not None:
        # get in string format for conversion to xx,xy,yx,yy format
        if any([char in sels.corrs for char in ":~"]):
            sels.corrs = Chooser.get_knife(sels.corrs, get_slicer=False)
            sels.corrs = ",".join([tdata.reverse_corr_map[_] for _ in sels.corrs])
        elif sels.corrs.startswith("diag"):
            sels.corrs = ",".join([*map(tdata.reverse_corr_map.get, [0, 3])])
        elif sels.corrs.startswith("off-"):
            sels.corrs = ",".join([*map(tdata.reverse_corr_map.get, [1, 2])])
        sels.corrs = sels.corrs.replace(" ", "").upper().split(",")
        if any([c.isdigit() for c in sels.corrs]):
            sels.corrs = [tdata.reverse_corr_map[int(c)] if c.isdigit() else c 
                            for c in sels.corrs]
        sels.corrs = [c for c in sels.corrs if c in tdata.corr_map]

    else:
        if tdata.num_corrs >2:
            sels.corrs = [*map(tdata.reverse_corr_map.get, [0, 3])]
        else:
            sels.corrs = [*map(tdata.reverse_corr_map.get, [0, 1])]

    new_order = []
    for (dd, fi, sc) in dd_fi:
        sub_order = []
        for i, sub in enumerate(ms):
            if (sub.DATA_DESC_ID, sub.FIELD_NAME, sub.SCAN_NUMBER) == (dd, fi, sc) \
                and (fi in sels.fields) and dd in sels.ddids \
                and sc in sels.scans:
                snitch.debug(f"Selecting ddid {dd} and field {fi}")
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
    
    # leave active fields as numbers because of plotting
    tdata.active_fields = [tdata.field_map[_] for _ in sels.fields]
    tdata.active_antennas = new_order[0].ant.values
    tdata.active_corrs = new_order[0].corr.values
    return new_order

def set_xaxis(gain):
    gains = {k: "channel"  for k in "b xf df".split()}
    gains.update({k: "time" for k in "d g f k kcross".split()})
    return gains.get(gain.lower(), "time")


def main(parser, gargs=None):
    if gargs is None:
        ps = parser().parse_args()
    else:
        ps = parser().parse_args(gargs)

    if ps.out_dir:
        out_dir = ps.out_dir
    else:
        out_dir = os.path.join(os.path.curdir, "ragavi_out")

    if ps.debug:
        update_log_levels(snitch.parent, 10)

    logfile = ps.logfile if ps.logfile else "ragains-quartical.log"
    update_logfile_name(snitch.parent, update_output_dir(logfile, out_dir))


    for (msname, antennas, channels, corrs, ddids, fields, scans, cmap, yaxes,
         xaxis, html_name, image_name, gtype) in zip_longest(ps.msnames,
         ps.antennas, ps.channels, ps.corrs, ps.ddids, ps.fields, ps.scans,
         ps.cmaps, ps.yaxes, ps.xaxes, ps.html_names, ps.image_names, ps.gtypes):
        
        msname = msname.rstrip("/")
        subs = xds_from_zarr(f"{msname}::{gtype}")
        msdata = init_table_data(f"{msname}::{gtype}", subs)
        snitch.info(f"Loading {msdata.ms_name}")
        msdata.table_type = gtype

        generals = Genargs(msname=msname, version=version)

        selections = Selargs(
            antennas=antennas, corrs=corrs, baselines=None,
            channels=Chooser.get_knife(channels), ddids=ddids, scans=scans)

        subs = organise_table(subs, selections, msdata)

        if cmap is None:
            cmap = "coolwarm"
        cmap = get_colours(subs[0].ant.size, cmap)
        points = calculate_points(subs[0], len(subs))

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
        if msdata.table_type.lower().startswith("k"):
            yaxes = ["delay"]
        if xaxis is None:
            xaxis = set_xaxis(msdata.table_type)

        all_figs = []
        for yaxis in yaxes:
            snitch.info(f"Axis: {yaxis}")
            figrag = FigRag(
                add_toolbar=True, width=900, height=710,
                x_scale=xaxis if xaxis == "time" else "linear",
                plot_args={"frame_height": None, "frame_width": None})

            for sub in subs:
                axes = Axargs(xaxis=xaxis, yaxis=yaxis, msdata=msdata,
                             data_column=None, flags=None, errors=None,)
                figrag = populate_fig_data(sub, axes=axes,
                    cmap=cmap, figrag=figrag, msdata=msdata)

            figrag.update_xlabel(axes.xaxis)
            figrag.update_ylabel(axes.yaxis)

            if "chan" in axes.xaxis:
                add_extra_xaxis(subs[0].gain_f.values, figrag, selections.channels)

            figrag.add_legends(group_size=_GROUP_SIZE_, visible=True)
            figrag.update_title(f"{axes.yaxis} vs {axes.xaxis}")
            figrag.show_glyphs(selection="b0")
            figrag = add_hover_data(figrag, axes)

            all_figs.append(figrag)
        
        if image_name:
            image_name = update_output_dir(image_name, out_dir)
            if "::" in image_name:
                image_name = image_name.replace("::", "-")
            statics = lambda func, _x, **kwargs: getattr(_x, func)(**kwargs)

            """   
            Generate all differnt combinations of all_figs and the name of
            the static functions and then split them into individual lists
            by unpacking the output of zip
            """
            ################### Normal program #############################
           
            with futures.ThreadPoolExecutor() as executor:
                stores = executor.map(
                    partial(statics, mdata=msdata, filename=image_name,
                            group_size=_GROUP_SIZE_),
                    *zip(*product(["write_out_static", "potato"], all_figs)))
           
           
            ########### only uncommment when debugging executor ############
           
            # for figa in all_figs:
            #     figa.write_out_static(mdata=msdata, filename=image_name,
            #                 group_size=_GROUP_SIZE_)
            #     figa.potato(mdata=msdata, filename=image_name,
            #                 group_size=_GROUP_SIZE_)
            
            ################################################################

        if html_name:
            data_column = "gains"
            html_name = update_output_dir(html_name, out_dir)

            if "::" in html_name:
                html_name = html_name.replace("::", "-")
            all_figs[0].link_figures(*all_figs[1:])
            all_figs = [fig.fig for fig in all_figs]
            widgets = make_widgets(
                msdata, all_figs[0], group_size=_GROUP_SIZE_)
            stats = make_stats_table(msdata, data_column, yaxes, subs)
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


def console():
    """A console run entry point for setup.cfg"""
    main(quartical_gains_parser)

if __name__ == "__main__":
    console()