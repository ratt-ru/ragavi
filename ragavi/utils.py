import os
import re
import bokeh.palettes as bp
import colorcet as cc
import dask.array as da
import matplotlib.colors as mpc
import matplotlib.pyplot as plt
import numpy as np

from time import perf_counter
from difflib import get_close_matches
from ragavi.lograg import logging

snitch = logging.getLogger(__name__)

def timer(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        res = func(*args, **kwargs)
        end = perf_counter()
        out = f"{func.__name__}: run in {end-start} sec"
        print(out)
        print("="*len(out))
        return res
    return wrapper

def get_colours(n, cmap="coolwarm"):
    """
    cmap: colormap to select
    n: Number colours to return
    """
    #check if the colormap is in mpl
    if re.search(f"\w*{cmap}\w*", ",".join(plt.colormaps()),
                    re.IGNORECASE):
        cmap = re.search(f"\w*{cmap}\w*", ",".join(plt.colormaps()),
                           re.IGNORECASE).group()
        cmap = plt.get_cmap(cmap)
        norm = mpc.Normalize(vmin=0, vmax=n-1)
        return [mpc.to_hex(cmap(norm(a))) for a in range(n)]

    elif len(get_close_matches(cmap, plt.colormaps())) > 0:
        cmap, = get_close_matches(cmap, plt.colormaps(), n=1)
        cmap = plt.get_cmap(cmap)
        norm = mpc.Normalize(vmin=0, vmax=n)
        return [mpc.to_hex(cmap(norm(a))) for a in range(n)]

    #check if the colormap is in bokeh
    elif re.search(f"\w*{cmap}\w*", ",".join(bp.all_palettes.keys()),
                    re.IGNORECASE):
        cmaps = re.findall(f"\w*{cmap}\w*", ",".join(bp.all_palettes.keys()),
                           re.IGNORECASE)
        for cm in cmaps:
            if max(bp.all_palettes[cm].keys()) > n:
                cmap = bp.all_palettes[cmap][max(bp.all_palettes[cmap].keys())]
                break
        cmap = bp.linear_palette(cmap, n)
        return cmap
    elif len(get_close_matches(cmap, bp.all_palettes.keys())) > 0:
        cmaps = get_close_matches(cmap, bp.all_palettes.keys(), n=4)
        for cm in cmaps:
            if max(bp.all_palettes[cm].keys()) > n:
                cmap = bp.all_palettes[cmap][max(bp.all_palettes[cmap].keys())]
                break
        cmap = bp.linear_palette(cmap, n)
        return cmap
    #check if colormap is in colorcet
    elif len(get_close_matches(cmap, cc.palette.keys())) > 0:
        cmap = cc.palette(get_close_matches(cmap, cc.palette.keys(), n=1)[0])
        return bp.linear_palette(cmap, n)
    else:
        raise InvalidCmap(f"cmap {cmap} not found.")
        return -1

def new_darray(in_model, out_name, out_value):
    """Generate a new data array emulating the sstructure of  another
    This is for custom variables that are not included in some dataset.
    Note that the output value is repeated in the new array.
    
    Parameters
    ----------
    in_model: :obj:`xr.dataArray`
        Data array to be emulated
    out_name: :obj:`str`
        Name of output data array
    out_value: :obj:
        Value to be filled in the new array
    
    Returns
    -------
    bn : :obj:`xr.dataArray`
        New data array containing the repeated value
    """
    types = {bool: bool, int: np.int8}
    bn = in_model.copy(deep=True, data=da.full(
        in_model.shape, out_value, dtype=types[type(out_value)]))
    bn.name = out_name
    return bn

def pair(x, y):
    """
    Get a unique int representation from two distinct non-negative numbers
    This function is an implementation of cantor's pairing function which can
    be found in (page 1127 of A New Kind Of Science). Obtained from
    http://szudzik.com/ElegantPairing.pdf Follow this link to get more:
    https://stackoverflow.com/questions/919612/mapping-two-integers-to-\
        one-in-a-unique-and-deterministic-way/13871379#13871379
    """
    val = np.square(x) + (3*x) + (2*x*y) + y + np.square(y)
    return val//2


def update_output_dir(output_file, output_dir):
    if os.path.isdir(output_dir):
        snitch.warn(f"Directory {output_dir} already exists. " +
                    "Contents will be overwritten")
    else:
        snitch.debug(f"Creating output directory {output_dir}")
        os.makedirs(output_dir)

    current_outdir = os.path.dirname(output_file)
    # Dirname on the output file is preceded by the specified output dir
    if current_outdir == "":
        output_file = os.path.join(output_dir, output_file)
    else:
        output_file = os.path.join(output_dir, os.path.basename(output_file))
    
    return output_file