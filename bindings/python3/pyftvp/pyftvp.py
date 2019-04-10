import numpy as np

from . import _ftvp as ftvp
from . import _ftvp_color as ftvp_color

from .base import OptMethod, UpdateStrategy

def prox_tv(u, la, epsilon = 0.0, iters=5000, block_size=16, steps=3, gapiter=1, gap_factor=0.25, rmse_factor=0.1, opt_method = OptMethod.SPLIT_NEWTON, color = False, upd_strategy = UpdateStrategy.UPDATE_VARYING):
    if u.ndim == 1:
        raise Exception()
    elif u.ndim == 2:
        ures = u[:,:,np.newaxis]
    else:
        ures = u
    if color:
        ures = np.swapaxes(ures,0,2)
    ures = ures.astype(np.double).copy(order='C')
    if rmse_factor is not None:
        gap_factor_new = rmse_factor * rmse_factor
    else:
        gap_factor_new = gap_factor

    if color:
        res = ftvp_color._prox_tv_color(ures, la, epsilon=epsilon, iters=iters, block_size=block_size, steps=steps, gapiter=gapiter, gap_factor=gap_factor_new, opt_method=opt_method)
        ures = np.swapaxes(ures,0,2)
    else:
        res = ftvp._prox_tv(ures, la, epsilon=epsilon, iters=iters, block_size=block_size, steps=steps, gapiter=gapiter, gap_factor=gap_factor_new, opt_method=opt_method, upd_strategy=upd_strategy.value)
    return ures, res
