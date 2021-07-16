import os
import numpy as np

import dask.array as da
import xarray as xr
from dask import compute
from collections import namedtuple
from itertools import product, zip_longest
from daskms.experimental.zarr import xds_from_zarr

from bokeh.layouts import grid, gridplot, column, row, layout
from bokeh.io import save, output_file

from exceptions import InvalidCmap, InvalidColumnName, EmptyTable
from arguments import gains_argparser
from plotting import FigRag, Circle, Scatter
from ragdata import Axargs, Selargs, stokes_types, dataclass, Genargs
from gains import get_colours
from processing import Chooser, Processor
from widgets import (F_MARKS, make_widgets, make_stats_table,
                     make_table_name)

from concurrent import futures 
from functools import partial
from lograg import logging, get_logger

import time as tme

snitch = get_logger(logging.getLogger(__name__))

_GROUP_SIZE_ = 16
_NOTEBOOK_ = False  

class TableData:
    def __init__(self, ms_name, ant_names=None, field_names=None,
                 corr_names=None, scans=None, spws=None):
        self.ms_name = ms_name
        self.ant_names = ant_names.tolist()
        self.field_names = field_names
        self.corr_names = corr_names.tolist()
        self.scans = scans
        self.spws = xr.DataArray(spws)
        self.active_corrs = []
        self.active_spws = [0]
        self.active_fields = []

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
    def corr_map(self):
        return {a: self.corr_names.index(a) for a in self.corr_names}

    @property
    def reverse_corr_map(self):
        return {v: k for k, v in self.corr_map.items()}
    
    @property
    def num_corrs(self):
        return len(self.corr_names)

    @property
    def num_ants(self):
        return len(self.ant_names)

    @property
    def num_fields(self):
        return len(self.field_names)
    
    @property
    def num_spws(self):
        return len(self.spws)

    @property
    def num_scans(self):
        return len(self. scans)

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
    ants = sub_list[0].ant_names.values
    corrs = stokes_types[sub_list[0].corr_type.values]

    return TableData(msname, ant_names=ants, corr_names=corrs,
        field_names=fields, spws=spws)


def populate_fig_data(subms, axes, cmap, figrag):
    time, freq = subms.gain_t.values, subms.gain_f.values / 1e9
    ants_corrs = product(range(subms.ant.size), range(subms.corr.size))
    for ant, corr in ants_corrs:
        start = tme.perf_counter()
        
        sub = subms.sel(ant=ant, corr=corr)
        gains, sdict = sub.gains, {}
        
        # original = ("gain_t", "gain_f", "ant", "dir", "corr")
        gains = np.ravel(gains.transpose("gain_f", "gain_t", "dir").data)
        
        total_reps = gains.size
        data_reps = gains.size // freq.size

        xaxes={
            "time": np.repeat(Processor.unix_timestamp(time), gains.size // time.size),
            "channel": np.repeat(np.arange(freq.size), gains.size // freq.size)
        }

        sdict["y"], sdict["field"], sdict["scan"], sdict["ddid"]= \
            compute(
            Processor(gains).calculate(axes.yaxis), sub.fields.data.flatten(),
            sub.scans.data.flatten(), sub.ddids.data.flatten(),
        )

        sdict["x"] = xaxes[axes.xaxis]
        sdict["ant"] = np.repeat(ant, total_reps)
        sdict["colors"] = np.repeat(cmap[ant], total_reps)
        sdict["corr"] = np.repeat(corr, total_reps)
        sdict["markers"] = np.repeat(F_MARKS[sub.FIELD_ID], total_reps)
        sdict["data"] = axes
        
        # TODO: change from ant to actual ant name
        figrag.add_glyphs(
            F_MARKS[sub.FIELD_ID], data=sdict,
            legend=str(ant), fill_color=cmap[ant],
            line_color=cmap[ant],
            tags=[f"a{ant}", f"c{corr}",
                  f"s{sub.DATA_DESC_ID}", f"f{sub.FIELD_ID}"])

        print(f"Loop Done at {tme.perf_counter() - start} secs")
    return figrag


def new_darray(in_model, out_name, out_value):
    types = {bool: bool, int: np.int8}
    bn = in_model.copy(deep=True, data=da.full(
        in_model.shape, out_value, dtype=types[type(out_value)]))
    bn.name = out_name
    return bn

def organise_table(ms, sels, tdata):
    """
    Group table into different ddids and fields
    """
    ms = sorted(ms, key=lambda x: x.SCAN_NUMBER)
    dd_fi = sorted(list({(m.DATA_DESC_ID, m.FIELD_ID) for m in ms}))
    variables = "gains fields scans ddids flags".split()

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
        if all([f.isdigit() for f in antennas]):
            sels.antennas = [int(f) for f in antennas]
        else:
            sels.antennas = [tdata.ant_map[f] for f in fields]
    else:
        sels.antennas = list(tdata.ant_map.values())

    tdata.active_fields = sels.fields
    tdata.active_antennas = sels.antennas
    # TODO: antenna slection needs to be fixed properly
    new_order = []
    for (dd, fi) in dd_fi:
        sub_order = []
        for i, sub in enumerate(ms):
            if (sub.DATA_DESC_ID, sub.FIELD_ID) == (dd, fi) and fi in sels.fields:
                sub = sub.sel(corr=sels.corrs, ant=sels.antennas, gain_f=sels.channels)
                sub["fields"] = new_darray(sub.gains, "fields", sub.FIELD_ID)
                sub["scans"] = new_darray(sub.gains, "scans", sub.SCAN_NUMBER)
                sub["ddids"] = new_darray(sub.gains, "ddids", sub.DATA_DESC_ID)
                sub["flags"] = new_darray(sub.gains, "flags", False)
                ms[i] = sub[variables]
                ms[i].attrs = {var: sub.attrs[var]
                               for var in ["DATA_DESC_ID", "FIELD_ID"]}
                sub_order.append(ms[i])
        new_order.append(xr.concat(sub_order, "gain_t",
                         combine_attrs="drop_conflicts"))
    return new_order

def main(parser, gargs):
    ps = parser().parse_args(gargs)

    for (msname, antennas, channels, corrs, ddids, fields, cmap, yaxes,
         xaxis, html_name, image_name) in zip_longest(
            ps.msnames, ps.antennas, ps.channels, ps.corrs, ps.ddids, ps.fields,
            ps.cmaps, ps.yaxes, ps.xaxes, ps.html_names, ps.image_names):

        ms = xds_from_zarr(msname + "::B")
        msdata = init_table_data(msname + "B", ms)

        generals = Genargs(msname=msname, version="testwhatever")

        selections = Selargs(
            antennas=antennas, corrs=Chooser.get_knife(corrs),
            baselines=None, channels=Chooser.get_knife(channels),
            ddids=Chooser.get_knife(ddids))

        ms = organise_table(ms, selections, msdata)

        cmap = get_colours(ms[0].ant.size)

        points = calculate_points(ms[0], len(ms))

        if html_name is None and image_name is None:
            html_name = msdata.ms_name + f"_{''.join(yaxes)}" + ".html"

        if points > 30000 and image_name is None:
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
            print(f"Axis: {yaxis}")
            figrag = FigRag(add_toolbar=True,
                            x_scale=xaxis if xaxis == "time" else "linear")

            start = tme.perf_counter()
            for sub in ms:
                axes = Axes(flags=None, errors=None,
                    xaxis=xaxis, yaxis=yaxis)
                figrag = populate_fig_data(sub, axes=axes,
                    cmap=cmap, figrag=figrag)
           
            print(f"Done at {tme.perf_counter() - start} secs")
            
            print("Here")

            figrag.update_xlabel(axes.xaxis)
            figrag.update_ylabel(axes.yaxis)

            if "chan" in axes.xaxis:
                add_extra_xaxis(freqs, figrag, selections.channels)

            figrag.add_legends(group_size=16, visible=True)
            figrag.update_title(f"{axes.yaxis} vs {axes.xaxis}")
            figrag.show_glyphs(selection="b0")

            all_figs.append(figrag)
        
        image_name ="lost.png"
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
            # stats = make_stats_table(msdata, data_column, yaxes,
            #                          get_table(msdata, selections,
            #                                    group_data=["SPECTRAL_WINDOW_ID", "FIELD_ID"]))
            stats = None
            # Set up my layouts
            all_widgets = grid([widgets[0], column(widgets[1:]+[])],
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
        snitch.info("Plotting Done")


if __name__ == "__main__":
    ms_name="/home/lexya/Documents/test_gaintables/quartical/gains.qc"
    #synonyms for the the tables available here

    yaxes = "a"
    xaxis = "time"
    main(gains_argparser, ["-t", ms_name, "-y", yaxes, "-x", xaxis, "--ant", 
    "0,1"])
