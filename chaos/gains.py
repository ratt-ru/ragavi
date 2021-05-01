import re
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import numpy as np

import bokeh.palettes as bp
import daskms as xm
from itertools import zip_longest, product, cycle
from dask import compute

from bokeh.layouts import grid, gridplot, column, row, layout
from bokeh.io import save, output_file

from holy_chaos.chaos.exceptions import InvalidCmap, InvalidColumnName, EmptyTable
from holy_chaos.chaos.ragdata import dataclass, field, MsData, Axargs, Genargs, Selargs, Plotargs
from holy_chaos.chaos.arguments import gains_argparser
from holy_chaos.chaos.plotting import FigRag, Circle, Scatter
from holy_chaos.chaos.processing import Chooser, Processor
from holy_chaos.chaos.widgets import F_MARKS, make_widgets, make_stats_table, make_table_name

from ipdb import set_trace

_GROUP_SIZE_ = 8

class Pargs(Plotargs):
    image_name: str = None


def get_table(msdata, sargs, group_data):
    # perform  TAQL selections and chan and corr selection
    super_taql = Chooser.form_taql_string(msdata, antennas=sargs.antennas,
                                          baselines=sargs.baselines,
                                          fields=sargs.fields,
                                          spws=sargs.ddids, taql=sargs.taql,
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
    else:
        tip0 = (f"({ax_info.xaxis:.4}, {ax_info.yaxis:.4})", f"(@x, @y)")

    figrag.fig.select_one({"tags": "hover"}).tooltips = [
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
        "corr": np.full(ax_info.xdata.shape, corr, dtype="int8"),
        "pcf": np.full(ax_info.xdata.shape,
                        (sub.FLAG.sum()/sub.FLAG.size).values * 100)
    })
    return data

def get_colours(n, cmap="coolwarm"):
    """
    cmap: colormap to select
    n: Number colours to return
    """
    #check if the colormap is in mpl
    if cmap in plt.colormaps():
        cmap = plt.get_cmap(cmap)
        norm = mpc.Normalize(vmin=0, vmax=n)
        return [mpc.to_hex(cmap(norm(a))) for a in range(n)]

    elif re.search(f"{cmap}\w*", ",".join(plt.colormaps()), re.IGNORECASE):
        cmap = re.search(f"{cmap}\w*", ",".join(plt.colormaps()),
            re.IGNORECASE).group()
        cmap = plt.get_cmap(cmap)
        norm = mpc.Normalize(vmin=0, vmax=n)
        return [mpc.to_hex(cmap(norm(a))) for a in range(n)]
    
    #check if the colormap is in bokeh
    elif re.search(f"{cmap}\w*", ",".join(bp.all_palettes.keys()), re.IGNORECASE):
        cmaps = re.findall(f"{cmap}\w*", ",".join(bp.all_palettes.keys()), 
            re.IGNORECASE)
        for cm in cmaps:
            if max(bp.all_palettes[cm].keys()) > n:
                cmap = bp.all_palettes[cmap][max(bp.all_palettes[cmap].keys())]
                break
        cmap = bp.linear_palette(cmap, n)
        return cmap
    else:
        raise InvalidCmap(f"cmap {cmap} not found.")


#"/home/lexya/Documents/test_gaintables/1491291289.B0"
#"/home/lexya/Documents/test_gaintables/1491291289.F0"
#"/home/lexya/Documents/test_gaintables/1491291289.K0"
#"/home/lexya/Documents/test_gaintables/1491291289.G0"
#"/home/lexya/Documents/test_gaintables/3c391_ctm_mosaic_10s_spw0.B0"
#"/home/lexya/Documents/test_gaintables/4k_pcal3_1-1576599977_fullpol_1318to1418MHZ-single-3pcal_1.Df"
#"/home/lexya/Documents/test_gaintables/4k_pcal3_1-1576599977_fullpol_1318to1418MHZ-single-3pcal_1.Kcrs"
#"/home/lexya/Documents/test_gaintables/4k_pcal3_1-1576599977_fullpol_1318to1418MHZ-single-3pcal_1.Xf"
#"/home/lexya/Documents/test_gaintables/4k_pcal3_1-1576599977_fullpol_1318to1418MHZ-single-3pcal_1.Xref"
#"/home/lexya/Documents/test_gaintables/J0538_floi-beck-1pcal.Xfparang"
#"/home/lexya/Documents/test_gaintables/short-gps1-1gc1_primary_cal.G0"
#"/home/lexya/Documents/test_gaintables/VLBA_0402_3C84.B0"
#"/home/lexya/Documents/test_gaintables/workflow2-1576687564_sdp_l0-1gc1_primary.B0"


if __name__ == "__main__":
    gargs = ["-t", "/home/lexya/Documents/test_gaintables/1491291289.G0", "b.ms", "c.ms",
             "--cmap", "coolwarm", 
             "-a", "", "m000,m10,m063", "7",
            #  "-c", "", "1", "0,1",
            #  "--ddid", "0",
            #  "-f", "", "DEEP2", "J342-342,X034-123",
            #  "--t0", "0", "1000", "2000",
            #  "--t1", "10000", "7000", "7500",
             "-y", "a", "r,i", "i,a",
             "-x", "time", "time", "chan"]

    ps = gains_argparser().parse_args(gargs)


#overall arg:
# - debug
# - logfile

# DO THE SAME THING FOR PS.IMAGE_NAMES
if len(ps.html_names) == 1:
    #put the plots in a single file:
    pass
else:
    pass
    # put all of them in different files

ps.cmaps = ps.cmaps * len(ps.msnames) if len(ps.cmaps)==1 else ps.cmaps

for (msname, antennas, baselines, channels, corrs, ddids, fields, t0, t1, taql, cmap,
    yaxes, xaxis)in zip_longest(ps.msnames, ps.antennas, ps.baselines, ps.channels, ps.corrs, ps.ddids,
    ps.fields, ps.t0s, ps.t1s, ps.taqls, ps.cmaps, ps.yaxes, ps.xaxes):

    #we're grouping the arguments into 4
    gen_args = Genargs(msname=msname, version="testwhatever")
    sel_args = Selargs(antennas=antennas, corrs=Chooser.get_knife(corrs),
                       baselines=baselines, channels=Chooser.get_knife(channels),
                       ddids=ddids, fields=fields, taql=taql, t0=t0, t1=t1)
    
    #initialise data ssoc with ms
    msdata = MsData(msname)

    subs = get_table(msdata, sel_args, group_data=["SPECTRAL_WINDOW_ID", 
                                                    "FIELD_ID", "ANTENNA1"])
    for sub in subs:
        if sub.FIELD_ID not in msdata.active_fields:
            msdata.active_fields.append(sub.FIELD_ID)
        if sub.SPECTRAL_WINDOW_ID not in msdata.active_spws:
            msdata.active_spws.append(sub.SPECTRAL_WINDOW_ID)
        if sub.ANTENNA1 not in msdata.active_antennas:
            msdata.active_antennas.append(sub.ANTENNA1)

    cmap = cycle(get_colours(len(msdata.active_antennas), cmap))
    
    pl_args = Pargs(cmap=cmap)
    
    if len(subs) > 0:
        msdata.active_corrs = subs[0].corr.values
    else:
        raise EmptyTable("Table came up empty. Please your check selection.")
    
    all_figs = []
        
    for yaxis in yaxes.split(","):
        print(f"Axis: {yaxis}, ")

        figrag = FigRag(add_toolbar=True, width=900, height=710,
            x_scale=xaxis if xaxis == "time" else "linear",
            plot_args={"frame_height": None, "frame_width": None})

        for sub, corr in product(subs, msdata.active_corrs):
            # print(f"Antenna {sub.ANTENNA1}")
            colour = next(cmap)
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
                "data": ax_info
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
        # figrag.write_out_static(msdata, "pst.png", group_size=_GROUP_SIZE_)
        figrag.potato(msdata, "pst.png", group_size=_GROUP_SIZE_)
        
        all_figs.append(figrag.fig)
        widgets = make_widgets(msdata, all_figs[0], group_size=_GROUP_SIZE_)
        stats = make_stats_table(msdata,
                ax_info.data_column, yaxes,
                get_table(msdata, sel_args,group_data=["SPECTRAL_WINDOW_ID",
                                                        "FIELD_ID"]))

        # Set up my layouts
        all_widgets = grid(widgets + [None, stats],
                           sizing_mode="stretch_width", nrows=1)
        plots = gridplot([all_figs], toolbar_location="right",
                         sizing_mode="stretch_width")
        final_layout = layout([
            [make_table_name(gen_args.version, msdata.ms_name)],
            [all_widgets], [plots]], sizing_mode="stretch_width")

        output_file(filename = "ghost.html")
    save(final_layout,filename="ghost.html", title="oster")


        # figrag.write_out()
    # add the widgets at this point. Only need the first figure
    set_trace()
    print("Ends")

