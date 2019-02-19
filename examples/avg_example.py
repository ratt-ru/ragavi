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
    """ Because we're not reducing over a dimension,
    blockwise passes numpy arrays to fn """
    return np.average(data, axis=1)[:, None, :]


vis = xds[0].DATA.data
visr = vis.rechunk((vis.chunks[0], args.chan_chunks, vis.chunks[2]))

# Average each channel chunk to 1 channel
avg = da.blockwise(fn, ("row", "chan", "corr"),
                   visr, ("row", "chan", "corr"),
                   adjust_chunks={"chan": 1},
                   dtype=visr.dtype)

print(avg.compute().shape)


def fn2(data):
    """ Because we're reducing over dimension chan
    blockwise pass a list of numpy arrays to fn2 """
    res = []

    for element in data:
        res.append(np.average(element, axis=1))

    # numpy unfunc that sums the list of results together
    return np.add.reduce(res)


# Completely reduce over the channel dimension
avg = da.blockwise(fn2, ("row", "corr"),
                   visr, ("row", "chan", "corr"),
                   dtype=visr.dtype)


print(avg.compute().shape)
