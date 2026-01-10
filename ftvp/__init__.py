"""Fast Total Variation Proximal operator on GPU.

This package provides a PyTorch-compatible implementation of the proximal
operator for isotropic Total Variation regularization on 2D images.

References
----------
[1] Chambolle, A., & Pock, T. (2015). A remark on accelerated block coordinate
    descent for computing the proximity operators of a sum of convex functions.
    SMAI Journal of Computational Mathematics, 1, 29-54.

[2] Chambolle, A., Tan, P., & Vaiter S. (2017). Accelerated Alternating Descent
    Methods for Dykstra-like problems. J. Math. Imag. Vision, 59(3), 48-97.
"""

from ftvp.api import proxtv

__version__ = "2.0.0"
__all__ = ["proxtv"]
