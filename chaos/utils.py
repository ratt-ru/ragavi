import matplotlib.colors as mpc
import matplotlib.pyplot as plt

import bokeh.palettes as bp
from time import perf_counter


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
    elif re.search(f"{cmap}\w*", ",".join(bp.all_palettes.keys()),
                    re.IGNORECASE):
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
