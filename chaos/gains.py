import re
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import bokeh.palettes as bp

from holy_chaos.chaos.exceptions import *
from holy_chaos.chaos.ragdata import dataclass, field, MsData, Axargs, Genargs, Selargs, Plotargs
from holy_chaos.chaos.arguments import gains_argparser
from holy_chaos.chaos.plotting import FigRag, Circle, Scatter
from holy_chaos.chaos.processing import Chooser, Processor
from itertools import cycle, zip_longest, repeat

from dask import compute

import daskms as xm
import numpy as np

from ipdb import set_trace

class Pargs(Plotargs):
    image_name: str = None


def get_table(msdata, sargs, group_data):
    # perform  TAQL selections and chan and corr selection
    super_taql = Chooser.form_taql_string(msdata, antennas=sargs.antennas,
                                          baselines=sargs.baselines, fields=sargs.fields,
                                          spws=sargs.ddids,
                                          taql=sargs.taql, time=f"{sargs.t0}, {sargs.t1}")

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
        mss[i] = sub.sel(chan=Chooser.get_knife(sargs.channels),
                corr=Chooser.get_knife(sargs.corrs))
    return mss

def calculate_stats(msdata, yaxis):
    pass

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


if __name__ == "__main__":
    gargs = ["-t", "/home/lexya/Downloads/radioastro/tutorial_docs/calibration_tutorial/1491291289.G0", "b.ms", "c.ms",
             "--cmap", "coolwarm", 
             "-a", "", "m000,m10,m063", "7",
             "-c", "", "1", "0,1",
             "--ddid", "0",
             "-f", "0", "DEEP2", "J342-342,X034-123",
             "--t0", "0", "1000", "2000",
             "--t1", "10000", "7000", "7500",
             "-y", "a,p,r", "r,i", "i,a",
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
    sel_args = Selargs(antennas=antennas, corrs=corrs, baselines=baselines,
                        channels=channels, ddids=ddids, fields=fields, taql=taql, t0=t0, t1=t1)
    #set user input arguments
    msdata = MsData(msname)
    msdata.initialise_data()
    msdata.active_channels = msdata.freqs[:][Chooser.get_knife(
        sel_args.channels)]

    cmap = get_colours(msdata.num_ants, cmap)
    pl_args = Pargs(cmap=cmap)

    for yaxis in yaxes.split(","):
        figrag = FigRag(add_toolbar=True, 
            x_scale=xaxis if xaxis == "time" else "linear" )
        #iterating over submss
        subs = get_table(msdata, sel_args, group_data=[
                         "SPECTRAL_WINDOW_ID", "FIELD_ID", "ANTENNA1"])
        for a_num, sub in enumerate(subs):
            ax_info = Axargs(xaxis=xaxis, yaxis=yaxis, data_column="CPARAM", ms_obj=sub, msdata=msdata)
            
            ax_info.ydata = Processor(ax_info.ydata).calculate(ax_info.yaxis).data
            #get x-axis data
            ax_info.xdata = Processor(ax_info.xdata).calculate(ax_info.xaxis).data
            set_trace()
            ax_info.flags = ~sub.FLAG.data
            ax_info.errors = sub.PARAMERR.data
            ax_info = iron_data(ax_info)
            #pass inverted flags here so that they are given to the view as is
            #the view only shows the data that is tre in hte boolean mask while hidding the rest
            figrag.add_glyphs("circle", data=ax_info,
                legend=msdata.reverse_ant_map[sub.ANTENNA1], 
                    fill_color=cmap[a_num], line_color=cmap[a_num])
        
        figrag.update_xlabel(ax_info.xaxis.capitalize())
        figrag.update_ylabel(ax_info.yaxis.capitalize())
        figrag.update_title(
            f"{ax_info.yaxis.capitalize()} vs {ax_info.xaxis.capitalize()}")
        figrag.add_legends(group_size=8, visible=True)
        
        figrag.write_out()
        set_trace()
