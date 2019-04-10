import numpy as np

from . cimport _cftvp as cftvp
cimport numpy as np

from .base import OptMethod, UpdateStrategy

def _prox_tv(np.ndarray[double, ndim=3, mode="c"] u not None, double la,
             epsilon = 0.0, iters=5000, block_size=16, steps=1, gapiter=1, gap_factor=0.25, opt_method = OptMethod.SPLIT_NEWTON, upd_strategy = 0):
    sx, sy, sz = u.shape[0], u.shape[1], u.shape[2]
    if opt_method == OptMethod.SPLIT_NEWTON:
        res = cftvp.prox_tv(sx, sy, sz, &u[0,0,0], la, epsilon, iters, block_size, steps, gapiter, gap_factor, cftvp.OE_SPLIT_NEWTON, upd_strategy)
    elif opt_method == OptMethod.SPLIT_DESCENT:
        res = cftvp.prox_tv(sx, sy, sz, &u[0,0,0], la, epsilon, iters, block_size, steps, gapiter, gap_factor, cftvp.OE_SPLIT_DESCENT, upd_strategy)
    return (res.it, res.msec, res.gap, res.rmse)
