import os
import json
from concurrent import futures
from functools import partial

import numpy as np
import dask.array as da
import dask.dataframe as dd
import daskms as xm
import datashader as ds
import datashader.transfer_functions as tf
import xarray as xr

from itertools import cycle, product, zip_longest
from dask import compute, config
from dask.diagnostics import ProgressBar
from daskms.table_schemas import MS_SCHEMA
from bokeh.core.properties import value
from bokeh.io import output_file, output_notebook, save
from bokeh.layouts import column, grid, gridplot, layout, row
from bokeh.models import Div, ImageRGBA, PreText, Text

from chaos.arguments import vis_argparser
from chaos.exceptions import EmptyTable, InvalidCmap, InvalidColumnName, warn
from chaos.lograg import logging, get_logger, update_log_levels
from chaos.plotting import FigRag, Circle, Scatter
from chaos.processing import Chooser, Processor
from chaos.ragdata import (dataclass, field, Axargs, Genargs, MsData, Plotargs,
    Selargs)
from chaos.utils import get_colours, new_darray, timer, bp
from chaos.widgets import F_MARKS, make_stats_table, make_table_name, make_widgets

snitch = get_logger(logging.getLogger(__name__))

def antenna_iter(msdata, columns, **kwargs):
    """
    Return a list containing iteration over antennas in respective SPWs
    Parameters
    ----------
    msdata: :obj:`str`
        Name of the MS

    columns : :obj:`list`
        Columns that should be present in the dataset

    Returns
    -------
    subs: :obj:`list`
        A list containing data for each individual antenna. This list is
        ordered by antenna and SPW.
    """
    snitch.debug("Creating antenna iterable")

    taql_where, table_schema, chunks = [*map(kwargs.get, ["taql_where",
        "table_schema", "chunks"])]
    
    # Shall be prepended to the later antenna selection    
    taql_where = f"{taql_where} && " if taql_where else taql_where
    subs = []

    if "DATA_DESC_ID" in kwargs["group_cols"]:
        ddid_name = "DATA_DESC_ID"  
    else:
        ddid_name = "SPECTRAL_WINDOW_ID"

    for spw, ant in product(kwargs["active_spws"], kwargs["active_ants"]):
        snitch.debug(f"Spw: {spw}, antenna: {ant}")

        sel_str = (taql_where + 
            f"ANTENNA1=={ant} || ANTENNA2=={ant} && {ddid_name}=={spw}")

        # do not group to capture all the data
        sub, = xm.xds_from_ms(msdata.ms_name, taql_where=sel_str,
                    table_schema=table_schema, chunks=chunks, columns=columns,
                    group_cols=[])

        # Add the selection attributes
        sub.attrs = dict(ANTENNA=ant, DATA_DESC_ID=spw)
        subs.append(sub)

    snitch.debug("Done")
    return subs


def corr_iter(subs):
    """ Return a list containing iteration over corrs in respective SPWs

    Parameters
    ----------
    subs: :obj:`list`
        List containing subtables for the daskms grouped data

    Returns
    -------
    outp: :obj:`list`
        A list containing data for each individual corr. This list is
        ordered by corr and SPW.

    # NOTE: can also be used for chan iteration. Will require name change
    """
    snitch.debug("Creating correlation iterable")

    outp, n_corrs = [], subs[0].corr.size if len(subs) > 0 else 0

    for sub in subs:
        for c in range(n_corrs):
            snitch.debug(f"Corr: {c}")
            nsub = sub.copy(deep=True).sel(corr=c)
            nsub.attrs["CORR"] = c
            outp.append(nsub)

    snitch.debug("Done")
    return outp


def get_ms(msdata, selections, axes, cbin=None, chunks=None, tbin=None):
    """
    Get xarray Dataset objects containing Measurement Set columns of the
    selected data

    Parameters
    ----------
    ms_name: :obj:`str`
        Name of your MS or path including its name
    chunks: :obj:`str`
        Chunk sizes for the resulting dataset.
    cbin: :obj:`int`
        Number of channels binned together for channel averaging
    colour_axis: :obj:`str`
        Axis to be used for colouring
    iter_axis: :obj:`str`
        Axis to iterate over
    data_col: :obj:`str`
        Data column to be used. Defaults to 'DATA'
    tbin: :obj:`float`
        Time in seconds to bin for time averaging
    x_axis: :obj:`
        The chosen x-axis

    Returns
    -------
    tab_objs: :obj:`list`
        A list containing the specified table objects as
        :obj:`xarray.Dataset`
    """
    snitch.info(f"Working on {msdata.ms_name}")

    # Nb for i/c axis not in MS e.g baseline, corr, antenna, axes returns None
    # Always group by DDID
    if os.path.splitext(msdata.ms_name)[-1].lower() == ".ms":
        group_cols = ["DATA_DESC_ID"]
    else:
        group_cols = ["SPECTRAL_WINDOW_ID"]
    sel_cols = [*{"FLAG", *axes.active_columns}]
    
    where, actives = Chooser.form_taql_string(msdata, antennas=selections.antennas,
                baselines=selections.baselines, fields=selections.fields,
                spws=selections.ddids, scans=selections.scans,
                taql=selections.taql, return_ids=True)

    if axes.iaxis and axes.idata_col and axes.idata_col not in group_cols:
            group_cols += axes.idata_col.split()
    
    ms_schema = MS_SCHEMA.copy()
    ms_schema["WEIGHT_SPECTRUM"] = ms_schema["SIGMA_SPECTRUM"] = ms_schema["DATA"]

    # defining part of the gain table schema
    if axes.data_column not in ["DATA", "CORRECTED_DATA"]:
        ms_schema[axes.data_column] = ms_schema["DATA"]

    #removed chan and corr chunking    
    xds_inputs = dict(chunks=dict(row=chunks), taql_where=where,
                      columns=sel_cols, group_cols=group_cols)
    if {"antenna"}.issubset({axes.iaxis, axes.caxis}):
        tab_objs = antenna_iter(msdata, 
        **{**xds_inputs, **dict(
            active_ants=actives.get("antennas", range(msdata.num_ants)),
            active_spws=actives.get("spws", range(msdata.num_spws)))}
            )
        if axes.caxis == "antenna":
            # create an antenna column for colouration
            snitch.warn(f"No iterations as colour axis: {axes.caxis} is active")
            for sub in tab_objs:
                sub["ANTENNA"] = new_darray(sub.FLAG.ROWID, "ANTENNA", 
                                            sub.ANTENNA)
            tab_objs = [xr.concat(
                tab_objs, dim="row", combine_attrs="drop_conflicts"
                ).chunk(xds_inputs["chunks"])]
    elif axes.iaxis == "corr":
        tab_objs = corr_iter(xm.xds_from_ms(msdata.ms_name, **xds_inputs))
    else:
        tab_objs = xm.xds_from_ms(msdata.ms_name, **xds_inputs)
    if len(tab_objs) == 0:
        snitch.warning("No Data found in the MS" +
                     "Please check your selection criteria")
        raise EmptyTable(f"MS {msdata.ms_name}")
    else:
        # get some info about the data        
        chunk_sizes = " x ".join(map(str, tab_objs[0].FLAG.data.chunksize))
        snitch.info(f"Chunk size: {chunk_sizes}")
        snitch.info(f"Number of Partitions: {tab_objs[0].FLAG.data.npartitions}")
        if selections.channels is not None:
            tab_objs = [_.sel(chan=selections.channels) for _ in tab_objs]
        
        if selections.corrs is not None and axes.iaxis != "corr":
            tab_objs = [_.sel(corr=selections.corrs) for _ in tab_objs]

        return tab_objs


def repeat_ravel_viz(arr, max_size, chunk_size):
    """
    arr:
        Array to be repeated
    max_size: int
        Size of the corresponding array we want to match the incoming array's 
        size with. Used to determine the number of repetions
    chunk_size: int
        Size of chunks. This is taken to be the total of (row * chan * corr),
        or whatever is available in the MS. The determining size is the row 
        number of chunks, which is determined by the user.
    """
    dims = "ijkl"[:len(arr.shape)]
    arr = da.ravel(arr).rechunk(chunk_size)
    arr = da.blockwise(np.repeat, "i", arr, "i",
                       repeats=max_size//arr.size, axis=0, dtype=arr.dtype)
    return arr


def create_df(axes):
    """
    Create a dask dataframe from input x, y and iterate columns if
    available. This function flattens all the data into 1-D Arrays

    Parameters
    ----------
    x: :obj:`dask.array`
        Data for the x-axis
    y: :obj:`dask.array`
        Data for the y-axis
    iter_ax: :obj:`dask.array`
        iteration axis if possible

    Returns
    -------
    new_ds: :obj:`dask.Dataframe`
        Dataframe containing the required columns
    """
    snitch.info("Creating dataframe")
    max_size = max(axes.xdata.size, axes.ydata.size)
    chunk_size = np.prod(axes.flags.data.chunksize)

    axes.xdata = repeat_ravel_viz(axes.xdata, max_size, chunk_size)
    axes.ydata = repeat_ravel_viz(axes.ydata, max_size, chunk_size)

    candidates = {axes.xaxis: axes.xdata, axes.yaxis: axes.ydata}

    if axes.caxis is not None:
        if axes.caxis in ["corr", "chan"]:
            candidates[axes.caxis] = da.asarray(np.tile(axes.flags[axes.caxis], 
                max_size // axes.flags.corr.size), chunks=chunk_size)
        else:
            axes.cdata = repeat_ravel_viz(axes.cdata, max_size, chunk_size)
            candidates[axes.caxis] = axes.cdata
    
    ddf = dd.concat([dd.from_array(series.compute_chunk_sizes(), chunksize=chunk_size,
        columns=column) for column, series in candidates.items()], axis=1)

    #make the colour axes categorical
    if axes.caxis:
        ddf = ddf.categorize(axes.caxis)

    # new_ds = new_ds.dropna()

    snitch.info(f"DataFrame is ready with {ddf.npartitions} partitions")
    snitch.debug(f"Columns avalilable: {', '.join(ddf.columns)}")
    snitch.debug(f"Paritition size: {chunk_size}")
    return ddf


@timer
def image_callback(xy_df, plargs, axes):
    cvs = ds.Canvas(plot_width=plargs.c_width,
                    plot_height=plargs.c_height, 
                    x_range=(plargs.xmin, plargs.xmax),
                    y_range=(plargs.ymin, plargs.ymax))
    snitch.info("Datashader aggregation starting")
    if axes.caxis is not None:
        with ProgressBar():
            agg = cvs.points(xy_df, axes.xaxis, axes.yaxis,
                             ds.by(axes.caxis, ds.count())
                             )
    else:
        with ProgressBar():
            agg = cvs.points(xy_df, axes.xaxis, axes.yaxis, ds.count())

    snitch.info("Aggregation done")
    # change dtype to uint32 if agg contains a negative number
    agg = agg.astype(np.uint32) if (agg.values < 0).any() else agg
    return agg


def apply_blockwise(arr, func, out_dim=None):
    dims = "ijkl"[:len(arr.shape)]
    out_dim = dims if out_dim is None else out_dim
    arr = da.blockwise(func, out_dim, arr, dims, dtype=arr.dtype, 
        concatenate=True)
    return arr


def sort_the_plot(fig_num, fig, axes, plargs):
    """
    Finalise the plot
    fig: :obj:`bokeh figure`
        Plot containing figure
    axes:
        axis arguments dataclass
    plargs:
        plot arguments dataclass
    """
    """
    # For future use
    https://stackoverflow.com/questions/33037185/how-to-use-a-datetime-axis-with-a-bokeh-image-rgba-element
    https://stackoverflow.com/questions/25134639/how-to-force-python-print-numpy-datetime64-with-specified-timezone
    """
    if axes.xdata.dtype == "datetime64[s]":
        axes.xdata = axes.xdata.astype(np.float64)
    if axes.ydata.dtype == "datetime64[s]":
        axes.ydata = axes.ydata.astype(np.float64)
    if plargs.x_min is None:
        plargs.xmin = apply_blockwise(axes.xdata, np.nanmin, "")
    if plargs.x_max is None:
        plargs.xmax = apply_blockwise(axes.xdata, np.nanmax, "")
    if plargs.y_min is None:
        plargs.ymin = apply_blockwise(axes.ydata, np.nanmin, "")
    if plargs.y_max is None:
        plargs.ymax = apply_blockwise(axes.ydata, np.nanmax, "")
    
    if plargs.link_plots:
        snitch.debug("Locking plot standard x and y scale")
        plargs.x_min, plargs.x_max, plargs.y_min, plargs.y_max = (
            plargs.xmin, plargs.xmax, plargs.ymin, plargs.ymax)
            
    snitch.info("Calculating x and y Min and Max ranges")
    # with ProgressBar():
    plargs.xmin, plargs.xmax, plargs.ymin, plargs.ymax = compute(
        plargs.xmin, plargs.xmax, plargs.ymin, plargs.ymax)

    if plargs.xmin == plargs.xmax:
        snitch.debug("Incrementing x-max + 1 because x-min==x-max")
        plargs.xmax += 1
    if plargs.ymin == plargs.ymax:
        snitch.debug("Incrementing y-max + 1 one because y-min==y-max")
        plargs.ymax += 1
    
    if np.all(np.isnan(plargs.ymin)) or np.all(np.isnan(plargs.ymax)):
        plargs.ymin = plargs.xmin = 0
        plargs.ymax = plargs.xmax = 1

    snitch.info("x: {:.4}, min: {:10.4f}, max: {:10.4f}".format(axes.xaxis,
                    plargs.xmin, plargs.xmax))
    snitch.info("y: {:.4}, min: {:10.4f}, max: {:10.4f}".format(axes.yaxis,
                    plargs.ymin, plargs.ymax))

    df = create_df(axes)
    agg = image_callback(df, plargs, axes)

    if axes.caxis is None:
        img = tf.shade(agg, cmap=plargs.cmap)
    else:
        #slice colour map coz sometimes n_categories is more e.g for baselines
        # Also reduce cat_maps and n_categories
        plargs.n_categories = agg[axes.caxis].size
        plargs.cat_map = {idx: plargs.cat_map[idx] for idx in agg[axes.caxis].values}
        img = tf.shade(agg, 
            color_key=cycle(bp.linear_palette(plargs.cmap, plargs.n_categories)))

    fig.add_glyphs(ImageRGBA,
        dict(
            image=[img.data], x=[plargs.xmin], y=[plargs.ymin],
            dw=[plargs.xmax-plargs.xmin], dh=[plargs.ymax-plargs.ymin], 
            minx=[plargs.xmin], miny=[plargs.ymin],
            maxx=[plargs.xmax], maxy=[plargs.ymax],
            **plargs.i_ttips),
        dilate=False)
    
    if axes.xaxis == "time":
        plargs.xmin = plargs.xmin.astype("datetime64[s]")
        plargs.xmax = plargs.xmax.astype("datetime64[s]")
        fig.fig.renderers[0].data_source.data.update(
            dw=[plargs.xmax-plargs.xmin], x=[plargs.xmin],
            minx=[plargs.xmin], maxx=[plargs.xmax]
            )
        tip0 = (f"{axes.xaxis:.4}, {axes.yaxis:.4}",
                "($x{%F %T}, $y)")
        mmax = ("Min-x, Max-x", "(@minx{%F %T}, @maxx{%F %T})")
        fig.fig.tools[0].formatters = {"$x": "datetime", "@minx": "datetime",
                                        "@maxx": "datetime", }
    else:
        tip0 = (f"{axes.xaxis:.4}, {axes.yaxis:.4}", f"($x, $y)")
        mmax = ("Min-x, Max-x", f"({plargs.xmin:.2f}, {plargs.xmax:.2f})")
    
    fig.fig.tools[0].tooltips = [
        tip0, mmax,
        ("Min-y, Max-y", "(@miny{0.00}, @maxy{0.00})")
    ]
    
    fig.update_xlabel(axes.xaxis)
    fig.update_ylabel(axes.yaxis)
    
    if axes.iaxis is not None:
        # add some small title into the plot
        fig.fig.tools[0].tooltips.extend(
            [(name, f"@{name}") for name in plargs.i_ttips.keys()])
        # use value with text_font to avoid bad column name error
        # https://github.com/bokeh/bokeh/issues/11044
        fig.add_glyphs(
            Text,
            dict(
                x=[plargs.xmin + ((plargs.xmax-plargs.xmin) * 0.5)],
                y=[plargs.ymax * 0.87], text=[plargs.i_title]),
            text_font=value("monospace"), 
            text_font_style="bold",
            text_font_size="10pt", text_align="center")
    else:
        fig.update_title(plargs.title)
    
    if axes.caxis and fig_num%plargs.grid_cols==plargs.grid_cols-1:
        fig.add_categorical_colourbar(axes.caxis, plargs, visible=True, 
            position="right")
    return fig


def get_row_chunk(msd):
    """Get good chunk size for row depending
    
    Parameters
    ----------
    msd: MsData object
        MS data containing object
    """
    max_chunk = 10_000*4096*4
    row_cs = max_chunk // (msd.num_chans * msd.num_corrs)
    row_cs = int(np.floor(row_cs/10_000)*10_000)
    return row_cs


def main(parser, gargs): 
    ps = parser().parse_args(gargs)
    
    # ps.cbin, ps.tbin
    
    if ps.debug:
        update_log_levels(snitch.parent, 10)

    generals = Genargs(chunks=ps.chunk_size, mem_limit=ps.mem_limit,
        ncores=ps.ncores)

    config.set(num_workers=generals.ncores, memory_limit=generals.mem_limit)

    # repeated for all 
    for (msname, xaxis, yaxis, data_column, cmap, c_axis, i_axis, html_name,
        antennas, baselines, channels, corrs, ddids, fields, scans, taql
        ) in zip_longest(ps.msnames, ps.xaxes, ps.yaxes, ps.data_columns,
        ps.cmaps, ps.c_axes, ps.i_axes, ps.html_names, ps.antennas, ps.baselines,
        ps.channels, ps.corrs, ps.ddids, ps.fields, ps.scans, ps.taqls):

        msdata = MsData(msname)
        if generals.chunks is None:
            generals.chunks = get_row_chunk(msdata)

        if data_column is None:
            data_column = "DATA"
            snitch.info(f"Default data column: {data_column}")
        if ps.logfile is None:
            ps.logfile = "ragavi.log"
        if i_axis:
            ps.grid_cols = ps.grid_cols if ps.grid_cols else 5
        if cmap is None:
            cmap = "blues" if c_axis is None else "glasbey_bw"
            snitch.info(f"Colour map: {cmap}")
        if html_name is None:
            html_name = "{}_{}_vs_{}_{}_flagged_{}".format(
                os.path.basename(msdata.ms_name), yaxis, xaxis, data_column,
                ps.flag)
            html_name += f"_coloured-by {c_axis}" if c_axis else ""
            html_name += f"_iterated-by {i_axis}" if i_axis else ""
            html_name += ".html"
        else:
            html_name += "" if ".html" in html_name else ".html"
        
        if corrs is not None:
            corrs = [str(msdata.corr_map[_]) if _ in msdata.corr_map else _ 
                     for _ in corrs.upper().replace(" ", "").split(",")]
            invalid_corrs = [*filter(lambda x: not str.isdigit(x), corrs)]
            corrs = [*filter(str.isdigit, corrs)]
            invalid_corrs.extend([_ for _ in corrs if not int(_)
                              in msdata.reverse_corr_map])

            corrs = ",".join({_ for _ in corrs if int(_) in 
                                    msdata.reverse_corr_map})
            if corrs == "":
                corrs = [0, msdata.num_corrs-1]
                snitch.warn(
                    "Selected corrs {} are not available. Selecting {}".format(
                    ','.join(invalid_corrs), 
                    ','.join(map(msdata.reverse_corr_map.get, corrs))))
            snitch.info(f"Corrs: {corrs}")

        selections = Selargs(antennas=antennas, baselines=baselines,
            corrs=Chooser.get_knife(corrs), channels=Chooser.get_knife(channels),
            ddids=ddids, fields=fields, scans=scans, taql=taql)
        axes = Axargs(xaxis=xaxis, yaxis=yaxis, data_column=data_column,
            msdata=msdata, iaxis=i_axis, caxis=c_axis)
        
        if axes.yaxis == axes.xaxis:
            raise RuntimeError(
                f"x-axis '{axes.xaxis}' and y-axis '{axes.yaxis}' are similar")
        if axes.caxis == axes.iaxis and (axes.caxis and axes.iaxis):
            raise RuntimeError(
                f"x-axis '{axes.xaxis}' and y-axis '{axes.yaxis}' are similar")

        subs = get_ms(msdata, selections, axes, cbin=None,
                     chunks=generals.chunks, tbin=None)

        # set plot arguments
        plargs = Plotargs(cmap=cmap, c_height=ps.c_height, c_width=ps.c_width,
                          grid_cols=ps.grid_cols, x_min=ps.xmin, x_max=ps.xmax,
                          y_max=ps.ymax,y_min=ps.ymin, html_name=html_name,
                          partitions=len(subs), link_plots=ps.link_plots)
        plargs.set_grid_cols_and_rows()
        plargs.form_plot_title(axes)

        if axes.caxis is not None:
            plargs.set_category_ids_and_sizes(axes.caxis, msdata)
            plargs.cmap = get_colours(plargs.n_categories, plargs.cmap)
        else:
            plargs.cmap = get_colours(200, plargs.cmap)

        outp = []
        for isub, sub in enumerate(subs):
            snitch.info("Starting data processing")

            msdata.active_channels = msdata.freqs.sel(
                chan=selections.channels,
                row=[_ for _ in map(sub.attrs.get, ["DATA_DESC_ID", 
                    "SPECTRAL_WINDOW_ID"]) if _ is not None][0])

            print(f"shape:{sub[axes.data_column].shape}")
            if axes.iaxis is not None:
                plargs.set_iter_title(axes, sub, msdata)

            # transpose everything if axis is ever chan
            if ({axes.xaxis, axes.yaxis}.intersection({"chan", "channel", 
                "freq", "frequency"})):
                t_matrix = ["chan", "row"]
                t_matrix += ["corr"] if "corr" in sub.dims else []
                sub = sub.transpose(*t_matrix)

            # Set the axis data
            [axes.set_axis_data(_, sub) for _ in list("xyci")]
            
            axes.flags = sub.FLAG
            if ps.flag:
                snitch.debug(f"'-if is {ps.flag}'. Plotting non-flagged data.")
                sub = sub.where(axes.flags == False)

            axes.xdata = Processor(axes.xdata).calculate(axes.xaxis,
                                    msdata.active_channels).data
            axes.ydata = Processor(axes.ydata).calculate(axes.yaxis,
                                    msdata.active_channels).data
            
            if axes.cdata is not None:
                axes.cdata = Processor(axes.cdata).calculate(axes.caxis).data

            to_fig = dict(width=plargs.plot_width, height=plargs.plot_height,
                          x_scale=xaxis if xaxis == "time" else "linear",
                          plot_args={"frame_height": None, "frame_width": None},
                          add_grid=False, add_xaxis=True, add_yaxis=True,
                          add_toolbar=True)
            
            fig = sort_the_plot(isub, FigRag(**to_fig), axes, plargs)
            outp.append(fig)

        outp = [_.fig for _ in outp]
       
        if axes.iaxis is not None:
            title_div = Div(
                text=plargs.title, align="center", width=plargs.plot_width,
                style={"font-size": "24px", "height": "50px",
                        "text-align": "centre", "font-weight": "bold",
                        "font-family": "monospace"},
                sizing_mode="stretch_width")
            info_text = (f"ragavi   : v{generals.version}\n" + 
                        f"MS       : {msdata.ms_name}\n"+
                        f"Grid size: {plargs.grid_rows} x {plargs.grid_cols}" +
                        f" | Linked: {ps.link_plots}")

            pre = PreText(text=info_text, width=int(plargs.plot_width * 0.961),
                          height=50, align="start", margin=(0, 0, 0, 0),
                          sizing_mode="stretch_width")
            final_plot = gridplot(children=outp, ncols=plargs.grid_cols,
                                  sizing_mode="stretch_width")
            final_plot = column(children=[title_div, pre, final_plot],
                sizing_mode="stretch_width")
        else:
            info_text = (f"ragavi   : v{generals.version}\n" +
                         f"MS       :{msdata.ms_name}")
            pre = PreText(text=info_text, width=int(plargs.plot_width * 0.961),
                          height=50, align="start", margin=(0, 0, 0, 0),
                          sizing_mode="stretch_width")
            final_plot = column([pre, outp[0]], sizing_mode="stretch_width")
        
        output_file(html_name, title=plargs.title)
        save(final_plot, filename=html_name)
        snitch.info(f"Rendered plot to: {html_name}")
        snitch.info("Specified options:")
        parsed_opts = dict()
        for k, v in ps.__dict__.items():
            if type(v) is list and None not in v or v is not None:
                parsed_opts[k] = v
    
        parsed_opts = json.dumps(parsed_opts, indent=2, sort_keys=True)
        for _x in parsed_opts.split('\n'):
            snitch.info(_x)
        snitch.info(">" * 70)
    
    return 0
