import numpy as np

from . cimport _cftvp_color as cftvp_color
cimport numpy as np

from .base import OptMethod

def _prox_tv_color(np.ndarray[double, ndim=3, mode="c"] u not None, double la,
                   epsilon = -1, iters=500, block_size=16, steps=1, gapiter=1, gap_factor=0.25, opt_method = OptMethod.SPLIT_NEWTON):
    sx, sy, sc = u.shape[1], u.shape[2], u.shape[0]
    res = cftvp_color.prox_tv(sx, sy, sc, &u[0,0,0], la, epsilon, iters, block_size, steps, gapiter, gap_factor, cftvp_color.OE_SPLIT_DESCENT)
    return (res.it, res.msec, res.gap, res.rmse)
