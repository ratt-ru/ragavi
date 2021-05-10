import os
import numpy as np

import daskms as xm
from itertools import zip_longest, product
from dask import compute

from bokeh.layouts import grid, gridplot, column, row, layout
from bokeh.io import save, output_file

from holy_chaos.chaos.exceptions import (InvalidCmap, InvalidColumnName,
    EmptyTable)
from holy_chaos.chaos.ragdata import (dataclass, field, MsData, Axargs,
    Genargs, Selargs, Plotargs)
from holy_chaos.chaos.arguments import gains_argparser
from holy_chaos.chaos.plotting import FigRag, Circle, Scatter
from holy_chaos.chaos.processing import Chooser, Processor
from holy_chaos.chaos.widgets import (F_MARKS, make_widgets, make_stats_table,
    make_table_name)

from holy_chaos.chaos.utils import get_colours

from ipdb import set_trace

_GROUP_SIZE_ = 16

def get_table(msdata, sargs, group_data):
    # perform  TAQL selections and chan and corr selection
    super_taql = Chooser.form_taql_string(
        msdata, antennas=sargs.antennas, baselines=sargs.baselines,
        fields=sargs.fields, spws=sargs.ddids, taql=sargs.taql,
        time=(sargs.t0, sargs.t1))
    msdata.taql_selector = super_taql

    print(f"Selection string: {super_taql}")
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
        figrag.add_axis(chans[0], chans[-1], "x",
                        "linear", "Frequency GHz", "above")
    
def iron_data(ax_info):
    """Flatten these data to 1D"""
    def flat(att, obj): 
        if getattr(obj, att) is not None:
            return setattr(obj, att, getattr(obj, att).flatten())

    ax_info.xdata, ax_info.ydata, ax_info.flags,  ax_info.errors = compute(
        ax_info.xdata, ax_info.ydata, ax_info.flags, ax_info.errors)
   
    for dat in ["ydata", "flags", "errors"]:
        flat(dat, ax_info)
    ax_info.xdata = np.tile(ax_info.xdata, 
            ax_info.ydata.size//ax_info.xdata.size)
    return ax_info

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
    ax_info = data["data"]
    ant = msdata.reverse_ant_map[sub.ANTENNA1]
    field = msdata.reverse_field_map[sub.FIELD_ID]

    #format unix time in tooltip
    if ax_info.xaxis == "time":
        tip0 = (f"({ax_info.xaxis:.4}, {ax_info.yaxis:.4})",
                "(@x{%F %T}, @y)")
        fig.select_one({"tags": "hover"}).formatters = {"@x": "datetime"}
    else:
        tip0 = (f"({ax_info.xaxis:.4}, {ax_info.yaxis:.4})", f"(@x, @y)")

    fig.select_one({"tags": "hover"}).tooltips = [
        tip0,
        ("spw", "@spw"), ("field", "@field"), ("scan", "@scan"),
        ("ant", "@ant"), ("corr", "@corr"), ("% flagged", "@pcf")
    ]
    data.update({
        "spw": np.full(ax_info.xdata.shape, sub.SPECTRAL_WINDOW_ID, dtype="int8"),
        "field": np.full(ax_info.xdata.shape, field),
        "scan": np.tile(sub.SCAN_NUMBER.values,
                        ax_info.xdata.size//sub.row.size),
        "ant": np.full(ax_info.xdata.shape, ant, ),
        "corr": np.full(ax_info.xdata.shape, data["corr"], dtype="int8"),
        "pcf": np.full(ax_info.xdata.shape,
                        (sub.FLAG.sum()/sub.FLAG.size).values * 100)
    })
    return data

def calculate_points(subms, nsubs):
    return subms.FLAG.size * nsubs

def set_xaxis(gain):
    gains = {k: "channel"  for k in "b xf df".split()}
    gains.update({k: "time" for k in "d g f k kcross".split()})
    return gains.get(gain.lower(), "time")
    
def main(parser, gargs):
    ps = parser().parse_args(gargs)

    for (msname, antennas, baselines, channels, corrs, ddids, fields, t0, t1,
        taql, cmap, yaxes, xaxis, html_name, image_name) in zip_longest(ps.msnames,
        ps.antennas, ps.baselines, ps.channels, ps.corrs, ps.ddids, ps.fields, ps.t0s,
        ps.t1s, ps.taqls, ps.cmaps, ps.yaxes, ps.xaxes, ps.html_names, ps.image_names):
        
        #we're grouping the arguments into 4
        gen_args = Genargs(msname=msname, version="testwhatever")
        sel_args = Selargs(antennas=antennas, corrs=Chooser.get_knife(corrs),
                        baselines=baselines, channels=Chooser.get_knife(channels),
                        ddids=ddids, fields=fields, taql=taql, t0=t0, t1=t1)
        
        #initialise data ssoc with ms
        msdata = MsData(msname)
        print(f"MS: {msname}")

        subs = get_table(msdata, sel_args, group_data=["SPECTRAL_WINDOW_ID", 
                                                        "FIELD_ID", "ANTENNA1"])
        if len(subs) > 0:
            msdata.active_corrs = subs[0].corr.values
        else:
            raise EmptyTable(
                "Table came up empty. Please your check selection.")
    
        for sub in subs:
            if sub.FIELD_ID not in msdata.active_fields:
                msdata.active_fields.append(sub.FIELD_ID)
            if sub.SPECTRAL_WINDOW_ID not in msdata.active_spws:
                msdata.active_spws.append(sub.SPECTRAL_WINDOW_ID)
            if sub.ANTENNA1 not in msdata.active_antennas:
                msdata.active_antennas.append(sub.ANTENNA1)
        
        if yaxes is None:
            yaxes = ["a", "p"]
        elif yaxes == "all":
            yaxes = [_ for _ in "apri"]
        elif set(yaxes).issubset(set("apri")):
            yaxes = [_ for _ in yaxes]
        yaxes = [Axargs.translate_y(_) for _ in yaxes]

        if msdata.table_type.lower().startswith("k"):
            yaxes = ["delay"]
        
        if xaxis is None:
            xaxis = set_xaxis(msdata.table_type)

        if html_name is None and image_name is None:
            html_name = msdata.ms_name + f"_{''.join(yaxes)}" +".html"
        
        if cmap is None:
            cmap, = ps.cmaps
        cmap = get_colours(len(msdata.active_antennas), cmap)
        points = calculate_points(subs[0], len(subs))

        if points > 30000 and image_name is None:
            image_name = msdata.ms_name + ".png"
        
        all_figs = []
            
        for yaxis in yaxes:
            print(f"Axis: {yaxis}")

            figrag = FigRag(add_toolbar=True, width=900, height=710,
                x_scale=xaxis if xaxis == "time" else "linear",
                plot_args={"frame_height": None, "frame_width": None})

            for sub, corr in product(subs, msdata.active_corrs):
                # print(f"Antenna {sub.ANTENNA1}")
                colour = cmap[msdata.active_antennas.index(sub.ANTENNA1)]
                msdata.active_channels = msdata.freqs.sel(
                    chan=sel_args.channels,
                    row=sub.SPECTRAL_WINDOW_ID)

                ax_info = Axargs(xaxis=xaxis, yaxis=yaxis, data_column="CPARAM",
                    ms_obj=sub, msdata=msdata)
                ax_info.xdata = Processor(ax_info.xdata).calculate(
                    ax_info.xaxis).data

                
                # setting ydata here for reinit. Otherwise, autoset in axargs
                ax_info.ydata = Processor(sub[ax_info.ydata_col]).calculate(
                    ax_info.yaxis).sel(corr=corr).data
            
                ax_info.flags = ~sub.FLAG.sel(corr=corr).data
                ax_info.errors = sub.PARAMERR.sel(corr=corr).data
                ax_info = iron_data(ax_info)

                #pass inverted flags for the view as is
                #the view only shows true mask data
                data = {
                    "x":ax_info.xdata,
                    "y": ax_info.ydata,
                    "data": ax_info,
                    "corr": corr
                }
                data = add_hover_data(figrag.fig, sub, msdata, data)
                figrag.add_glyphs(F_MARKS[sub.FIELD_ID], data=data,
                    legend=msdata.reverse_ant_map[sub.ANTENNA1],
                    fill_color=colour,
                    line_color=colour,
                    tags=[f"a{sub.ANTENNA1}", f"s{sub.SPECTRAL_WINDOW_ID}",
                            f"c{corr}", f"f{sub.FIELD_ID}"])

            figrag.update_xlabel(ax_info.xaxis)
            figrag.update_ylabel(ax_info.yaxis)
            
            if "chan" in ax_info.xaxis:
                add_extra_xaxis(msdata, figrag, sel_args)
            
            figrag.add_legends(group_size=_GROUP_SIZE_, visible=True)
            figrag.update_title(f"{ax_info.yaxis} vs {ax_info.xaxis}")
            figrag.show_glyphs(selection="b0")

            figrag.write_out_static(msdata, image_name, group_size=_GROUP_SIZE_)
            figrag.potato(msdata, image_name, group_size=_GROUP_SIZE_)
            
            all_figs.append(figrag)
        
        if html_name:
            all_figs[0].link_figures(*all_figs[1:])
            all_figs = [fig.fig for fig in all_figs]
            widgets = make_widgets(msdata, all_figs[0], group_size=_GROUP_SIZE_)
            stats = make_stats_table(msdata,ax_info.data_column, yaxes,
                    get_table(msdata, sel_args,group_data=["SPECTRAL_WINDOW_ID",
                                                            "FIELD_ID"]))
            # Set up my layouts
            all_widgets = grid([widgets[0], column(widgets[1:]+[stats])],
                                sizing_mode="fixed", nrows=1)
            plots = gridplot([all_figs], toolbar_location="right",
                                sizing_mode="stretch_width")
            final_layout = layout([
                [make_table_name(gen_args.version, msdata.ms_name)],
                [all_widgets], [plots]], sizing_mode="stretch_width")
        
            output_file(filename=html_name)
            save(final_layout, filename=html_name,
                title=os.path.splitext(os.path.basename(html_name))[0])
        print("Plotting Done")
    return 0


#overall arg:
# - debug
# - logfile
if __name__ == "__main__":
    main(gains_argparser, [
         "-t", 
        #  "/home/lexya/Documents/chaos_project/holy_chaos/tests/gain_tables/1491291289.B0",
         "/home/lexya/Documents/chaos_project/holy_chaos/tests/gain_tables/1491291289.G0"
        ])
