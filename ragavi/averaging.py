# -*- coding: utf-8 -*-

"""
This is a module to average the main table of Measurement Sets, as well as the spectral window table. It largely borrows and was adapted from the xova application averager, whose code can be found here:
https://github.com/ska-sa/xova/blob/22614b64f0dd8caca20db74ec29705711e347fd0/xova/apps/xova/averaging.py
Commit #d653b38

"""

import dask.array as da
import daskms as xm
import numpy as np
import xarray as xr

from collections import OrderedDict
from dask.array.reductions import partial_reduce
from daskms import Dataset
from itertools import product, combinations
from africanus.averaging.dask import (time_and_channel,
                                      chan_metadata,
                                      chan_average as dask_chan_avg)

import ragavi.utils as vu

logger = vu.logger


def _id(array, fill_value=0, dtype_=np.int32):
    return np.full_like(array, fill_value, dtype=dtype_)


def id_full_like(exemplar, fill_value, dtype=np.int32):
    """ full_like that handles nan chunk sizes """
    return exemplar.map_blocks(_id, fill_value=fill_value,
                               dtype_=dtype, dtype=dtype)


def _safe_concatenate(*args):
    # Handle list with single numpy array case,
    # tuple unpacking fails on it
    if len(args) == 1 and isinstance(args[0], np.ndarray):
        return args[0]

    return np.concatenate(*args)


def concatenate_row_chunks(array, group_every=4):
    """
    Parameters
    ----------
    array : :class:`dask.array.Array`
        dask array to average.
        First dimension must correspond to the MS 'row' dimension
    group_every : int
        Number of adjust dask array chunks to group together.
        Defaults to 4.
    When averaging, the output array's are substantially smaller, which
    can affect disk I/O since many small operations are submitted.
    This operation concatenates row chunks together so that more rows
    are submitted at once
    """

    # Single chunk already
    if len(array.chunks[0]) == 1:
        return array

    # Restrict the number of chunks to group to the
    # actual number of chunks in the array
    group_every = min(len(array.chunks[0]), group_every)
    data = partial_reduce(_safe_concatenate, array,
                          split_every={0: group_every},
                          reduced_meta=None, keepdims=True)

    # partial_reduce sets the number of rows in each chunk
    # to 1, which is untrue. Correctly set the row chunks to nan,
    # steal the graph and recreate the array
    row_chunks = tuple(np.nan for _ in data.chunks[0])
    chunks = (row_chunks,) + data.chunks[1:]
    graph = data.__dask_graph__()

    return da.Array(graph, data.name, chunks, dtype=data.dtype)


def output_dataset(avg, field_id, data_desc_id, scan_number,
                   group_row_chunks):
    """
    Parameters
    ----------
    avg : namedtuple
        Result of :func:`average`
    field_id : int
        FIELD_ID for this averaged data
    data_desc_id : int
        DATA_DESC_ID for this averaged data
    scan_number : int
        SCAN_NUMBER for this averaged data
    group_row_chunks : int
        Concatenate row chunks group

    Returns
    -------
    Dataset
        Dataset containing averaged data
    """
    # Create ID columns
    fid = field_id
    ddid = data_desc_id
    scn = scan_number
    field_id = id_full_like(avg.time, fill_value=field_id)
    data_desc_id = id_full_like(avg.time, fill_value=data_desc_id)
    scan_number = id_full_like(avg.time, fill_value=scan_number)

    # Single flag category, equal to flags
    flag_cats = avg.flag[:, None, :, :]

    out_ds = {
        "ANTENNA1": (("row",), avg.antenna1),
        "ANTENNA2": (("row",), avg.antenna2),
        "DATA_DESC_ID": (("row",), data_desc_id),
        "FIELD_ID": (("row",), field_id),
        "SCAN_NUMBER": (("row",), scan_number),
        "FLAG_ROW": (("row",), avg.flag_row),
        # "FLAG_CATEGORY": (("row", "flagcat", "chan", "corr"), flag_cats),
        "TIME": (("row",), avg.time),
        "INTERVAL": (("row",), avg.interval),
        "TIME_CENTROID": (("row",), avg.time_centroid),
        "EXPOSURE": (("row",), avg.exposure),
        "UVW": (("row", "[uvw]"), avg.uvw),
        "WEIGHT": (("row", "corr"), avg.weight),
        "SIGMA": (("row", "corr"), avg.sigma),
        "DATA": (("row", "chan", "corr"), avg.vis),
        "FLAG": (("row", "chan", "corr"), avg.flag),
    }

    # Add optionally averaged columns columns
    if avg.weight_spectrum is not None:
        out_ds["WEIGHT_SPECTRUM"] = (("row", "chan", "corr"),
                                     avg.weight_spectrum)

    if avg.sigma_spectrum is not None:
        out_ds["SIGMA_SPECTRUM"] = (("row", "chan", "corr"),
                                    avg.sigma_spectrum)

    # Concatenate row chunks together
    if group_row_chunks > 1:
        grc = group_row_chunks
        # Remove items whose values are None
        out_ds = {k: (dims, concatenate_row_chunks(data, group_every=grc))
                  for k, (dims, data) in out_ds.items() if data is not None}

    return Dataset(out_ds, attrs={"DATA_DESC_ID": ddid,
                                  "FIELD_ID": fid,
                                  "SCAN_NUMBER": scn})


def average_main(main_ds, time_bin_secs, chan_bin_size,
                 group_row_chunks, respect_flag_row=True,
                 viscolumn="DATA", sel_cols=None):
    """ Perform averaging of the main table of the MS
    At this point,the input :attr:`main_ds` is a list containing an MS
    grouped by DDID, FIELD_ID & SCAN_NUMBER.

    Parameters
    ----------
    main_ds : list of Datasets
        Dataset containing Measurement Set columns.
        Should have a DATA_DESC_ID attribute.
    time_bin_secs : float
        Number of time bins to average together
    chan_bin_size : int
        Number of channels to average together
    group_row_chunks : int, optional
        Number of row chunks to concatenate together
    respect_flag_row : bool
        Respect FLAG_ROW instead of using FLAG
        for computing row flags.
    viscolumn: string
        name of column to average
    sel_cols: list
        Columns that need to be present in the dataset
    Returns
    -------
    avg
        tuple containing averaged data
    """
    output_ds = []

    # for each group
    for ds in main_ds:
        if respect_flag_row is False:
            if len(ds.FLAG.dims) > 2:
                ds = ds.assign(
                    FLAG_ROW=(("row",),
                              ds.FLAG.data.all(axis=(1, 2))))
            else:
                ds = ds.assign(FLAG_ROW=(("row",),
                                         ds.FLAG.data.all(axis=1)))

        # store the subgroup's data variables
        dv = ds.data_vars

        # Default kwargs.
        kwargs = {"time_bin_secs": time_bin_secs,
                  "chan_bin_size": chan_bin_size,
                  "vis": dv[viscolumn].data}

        # Other columns with directly transferable names
        # columns = ["FLAG_ROW", "TIME_CENTROID", "EXPOSURE", "WEIGHT",
        #           "SIGMA",
        #            "UVW", "FLAG", "WEIGHT_SPECTRUM", "SIGMA_SPECTRUM"]
        columns = ["TIME_CENTROID"] + sel_cols

        for c in columns:
            if c != viscolumn:
                try:
                    kwargs[c.lower()] = dv[c].data
                except KeyError:
                    pass

        # Set up the average operation
        avg = time_and_channel(**kwargs)

        output_ds.append(output_dataset(avg,
                                        ds.FIELD_ID,
                                        ds.DATA_DESC_ID,
                                        ds.SCAN_NUMBER,
                                        group_row_chunks))

    return output_ds


def average_spw(spw_ds, chan_bin_size):
    """
    Parameters
    ----------
    spw_ds : list of Datasets
        list of Datasets, each describing a single Spectral Window
    chan_bin_size : int
        Number of channels in an averaging bin
    Returns
    -------
    spw_ds : list of Datasets
        list of Datasets, each describing an averaged Spectral Window
    """

    new_spw_ds = []

    for r, spw in enumerate(spw_ds):
        # Get the dataset variables as a mutable dictionary
        dv = dict(spw.data_vars)

        # Extract arrays we wish to average
        chan_freq = dv["CHAN_FREQ"].data[0]
        chan_width = dv["CHAN_WIDTH"].data[0]
        effective_bw = dv["EFFECTIVE_BW"].data[0]
        resolution = dv["RESOLUTION"].data[0]

        # Construct channel metadata
        chan_arrays = (chan_freq, chan_width, effective_bw, resolution)
        chan_meta = chan_metadata((), chan_arrays, chan_bin_size)
        # Average channel based data
        avg = dask_chan_avg(chan_meta, chan_freq=chan_freq,
                            chan_width=chan_width,
                            effective_bw=effective_bw,
                            resolution=resolution,
                            chan_bin_size=chan_bin_size)

        num_chan = da.full((1,), avg.chan_freq.shape[0], dtype=np.int32)

        # These columns change, re-create them
        dv["NUM_CHAN"] = (("row",), num_chan)
        dv["CHAN_FREQ"] = (("row", "chan"), avg.chan_freq[None, :])
        dv["CHAN_WIDTH"] = (("row", "chan"), avg.chan_width[None, :])
        dv["EFFECTIVE_BW"] = (("row", "chan"), avg.effective_bw[None, :])
        dv["RESOLUTION"] = (("row", "chan"), avg.resolution[None, :])

        # But re-use all the others
        new_spw_ds.append(Dataset(dv))

    return new_spw_ds


def get_averaged_ms(ms_name, tbin=None, cbin=None, chunks=None, taql_where='',
                    columns=None, chan=None, corr=None, data_col=None,
                    group_cols=None, iter_axis=None):
    """ Performs MS averaging.

    Before averaging is performed, data selections is already performed 
    during the MS  acquisition process. TAQL (if available) is used to 
    perform selections for FIELD, SPW/DDID & SCAN. This is the first round of 
    selection. The second round involves selections over channels and 
    correlations. This is done via a slicer. With the exception of corr 
    selectino, all the other selections are done before averaging. This is 
    done because the averager requires 3-dimensional data.

    MS is then grouped by DDID, FIELD_ID & SCAN_NUMBER and fed into 
    :meth:`average_main` which actually performs the averaging.

    This function returns to  :meth:`ragavi.visibilities.get_ms` and is therefore grouped and column select similarly


    Parameters
    ----------
    ms_name : :obj:`str`
        Name of the input MS
    tbin : :obj:`float`
        Time bin in seconds
    cbin : :obj:`int`
        Number of channels to bin together
    chunks : :obj:`dict`
        Size of resulting MS chunks.
    taql_where: :obj:`str`
        TAQL clause to pass to xarrayms
    columns: :obj:`list`
        Columns to be present in the data
    chan : :obj:`slicer`
        A slicer to select the channels to be present in the dataset
    corr : :obj:`slicer` or :obj:`int`
        Correlation index of slice to be present in the dataset
    data_col : :obj:`str`
        Column containing data to be used
    group_cols: :obj:`list`
        List containing columns by which to group the data
    iter_axis: :obj:`str`
        Axis over which iteration is done

    Returns
    -------
    x_dataset: :obj:`list`
        List of :obj:`xarray.Dataset` containing averaged MS. The MSs are split by Spectral windows and grouped depending on the type of plots.

    """

    if chunks is None:
        chunks = dict(row=10000)

    # these are the defaults in averager
    if tbin is None:
        tbin = 1
    if cbin is None:
        cbin = 1

    # ensure that these are in the selected columns
    for _ in ["TIME", "ANTENNA1", "ANTENNA2", "INTERVAL", "FLAG", "FLAG_ROW", data_col]:
        if _ not in columns:
            columns.append(_)

    # must be grouped this way because of time averaging
    ms_obj = xm.xds_from_ms(ms_name,
                            group_cols=["DATA_DESC_ID",
                                        "FIELD_ID", "SCAN_NUMBER"],
                            columns=columns,
                            taql_where=taql_where)

    # some channels have been selected
    # corr selection is performed after averaging!!
    if chan is not None:
        ms_obj = [_.sel(chan=chan) for _ in ms_obj]

    logger.info("Averaging MAIN table")

    # perform averaging to the MS
    avg_mss = average_main(main_ds=ms_obj, time_bin_secs=tbin,
                           chan_bin_size=cbin, group_row_chunks=100000,
                           respect_flag_row=False, sel_cols=columns,
                           viscolumn=data_col)
    n_ams = len(avg_mss)

    # writes_ms = xm.xds_to_table(avg_mss, "tesxt", "ALL")
    logger.info("Creating averaged xarray Dataset")

    x_datasets = []
    for _a, ams in enumerate(avg_mss):
        ams = ams.compute()
        logger.info(f"Averaging {_a+1} / {n_ams}")

        datas = {k: (v.dims, v.data, v.attrs)
                 for k, v in ams.data_vars.items() if k != "FLAG_CATEGORY"}

        # create datasets and rename dimension "[uvw]"" to "uvw"
        new_ds = xr.Dataset(datas, attrs=ams.attrs, coords=ams.coords)
        new_ds = new_ds.chunk(chunks)

        if "uvw" in new_ds.dims.keys():
            new_ds = new_ds.rename_dims({"[uvw]": "uvw"})
        x_datasets.append(new_ds)

    # data will always be grouped by SPW unless iterating over antenna
    # the most amount of grouping that will occur will be between to columns
    all_grps = []

    if len(group_cols) == 0:
        # return a single dataset
        subs = xr.combine_nested(x_datasets, concat_dim="row",
                                 compat="no_conflicts", data_vars="all",
                                 coords="different", join="outer")
        subs.attrs = {}
        subs = subs.chunk(chunks)
        all_grps.append(subs)

    elif (set(group_cols) <= {"DATA_DESC_ID", "ANTENNA1", "ANTENNA2"} or
          iter_axis == "antenna"):
        uniques = np.unique([_.attrs["DATA_DESC_ID"] for _ in x_datasets])
        uants = np.arange(vu.get_antennas(ms_name).size)

        for _d in uniques:
            subs = []
            for _ in x_datasets:
                if _.attrs["DATA_DESC_ID"] == _d:
                    subs.append(_)
            subs = xr.combine_nested(subs, concat_dim="row",
                                     compat="no_conflicts", data_vars="all",
                                     coords="different", join="outer")
            subs.attrs = {"DATA_DESC_ID": _d}
            subs = subs.chunk(chunks)

            if {"ANTENNA1", "ANTENNA2"} <= set(group_cols):
                u_bl = combinations(uants, 2)
                for p, q in u_bl:
                    n_subs = subs.where((subs.ANTENNA1 == p) &
                                        (subs.ANTENNA2 == q), drop=True)
                    n_subs.attrs = {"DATA_DESC_ID": _d,
                                    "ANTENNA1": p,
                                    "ANTENNA2": q}
                    all_grps.append(n_subs)
            elif "ANTENNA1" in group_cols:
                for p in uants[:-1]:
                    n_subs = subs.where((subs.ANTENNA1 == p), drop=True)
                    n_subs.attrs = {"DATA_DESC_ID": _d,
                                    "ANTENNA1": p}
                    all_grps.append(n_subs)
            elif "ANTENNA2" in group_cols:
                for q in uants[:-1] + 1:
                    n_subs = subs.where((subs.ANTENNA2 == q), drop=True)
                    n_subs.attrs = {"DATA_DESC_ID": _d,
                                    "ANTENNA2": q}
                    all_grps.append(n_subs)
            elif iter_axis == "antenna":
                for p in uants:
                    n_subs = subs.where((subs.ANTENNA1 == p) |
                                        (subs.ANTENNA2 == p), drop=True)
                    n_subs.attrs = {"DATA_DESC_ID": _d,
                                    "ANTENNA": p}
                    all_grps.append(n_subs)
            else:
                all_grps.append(subs)

    elif set(group_cols) <= {"DATA_DESC_ID", "FIELD_ID", "SCAN_NUMBER"}:
        grps = {}
        # must be ddid + something else
        # if it is something other than fid and scan e.g
        # by default group by ddid
        for grp in group_cols:
            uniques = np.unique([_.attrs[grp] for _ in x_datasets])
            grps[grp] = uniques
            # grps.append(uniques)
        for com in product(*grps.values()):
            subs = []
            natt = {k: v for k, v in zip(group_cols, com)}
            for _ in x_datasets:
                if set(natt.items()) <= set(_.attrs.items()):
                    subs.append(_)
            subs = xr.combine_nested(subs, concat_dim="row",
                                     compat="no_conflicts",
                                     data_vars="all", coords="different",
                                     join="outer")
            subs.attrs = natt
            subs = subs.chunk(chunks)
            all_grps.append(subs)

     # select a corr
    if corr is not None:
        all_grps = [_.sel(corr=corr) for _ in all_grps]

    logger.info("Averaging completed.")

    return all_grps


def get_averaged_spws(ms_name, cbin, chan_select=None):
    """ Average spectral windows

    Parameters
    ----------
    ms_name : :obj:`str`
        Path or name of MS Spectral window Subtable
    cbin : :obj:`int`
        Number of channels to be binned together
    chan_select : :obj:`slicer` or :obj:`int`
        Which channels to select 

    Returns
    -------
    x_datasets : :obj:`list`
        List containing averaged spectral windows as :obj:`xarray.Dataset`
    """
    spw_subtab = list(xm.xds_from_table(ms_name, group_cols='__row__'))

    if chan_select is not None:
        spw_subtab = [_.sel(chan=chan_select) for _ in spw_subtab]

    logger.info("Averaging SPECTRAL_WINDOW subtable")

    av_spw = average_spw(spw_subtab, cbin)

    x_datasets = []

    # convert from daskms datasets to xarray datasets
    for sub_ds in av_spw:
        sub_ds = sub_ds.compute()

        datas = {k: (v.dims, v.data, v.attrs)
                 for k, v in sub_ds.data_vars.items() if k != "FLAG_CATEGORY"}

        new_ds = xr.Dataset(datas, attrs=sub_ds.attrs, coords=sub_ds.coords)
        new_ds = new_ds.chunk(chunks)
        x_datasets.append(new_ds)

    logger.info("Done")
    return x_datasets
