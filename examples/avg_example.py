import argparse

import dask.array as da
import numpy as np
from xarrayms import xds_from_ms


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms")
    p.add_argument("-r", "--rows", type=int, default=5000)
    p.add_argument("-c", "--chan-chunks", type=int, default=1024)
    return p


args = create_parser().parse_args()

xds = list(xds_from_ms(args.ms, chunks={"row": args.rows}))


def fn(data):
    return np.average(data, axis=1)[:, None, :]


vis = xds[0].DATA.data
visr = vis.rechunk((vis.chunks[0], args.chan_chunks, vis.chunks[2]))

avg = da.blockwise(fn, ("row", "chan", "corr"),
                   visr, ("row", "chan", "corr"),
                   adjust_chunks={"chan": 1},
                   dtype=visr.dtype)

print(avg.compute().shape)

