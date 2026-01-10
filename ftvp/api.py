"""Public API for the ftvp package."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

_ext = None


def _get_extension():
    global _ext
    if _ext is not None:
        return _ext

    from torch.utils.cpp_extension import load

    csrc_dir = Path(__file__).parent / "csrc"
    _ext = load(
        name="ftvp_cuda",
        sources=[str(csrc_dir / "kernels.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _ext


def proxtv(
    img: Tensor,
    lam: float,
    *,
    epsilon: float = 1e-3,
    max_iter: int = 100,
    inner_steps: int = 5,
    gap_iter: int = 1,
    gap_tol: float = 1e-4,
    use_newton: bool = True,
) -> Tensor:
    """Compute the proximal operator of isotropic Total Variation.

    This computes:
        prox_{lam * TV}(img) = argmin_x  0.5 * ||x - img||^2 + lam * TV(x)

    where TV(x) is the (smoothed) isotropic Total Variation of x.

    Parameters
    ----------
    img : Tensor
        Input image. Can be:
        - 2D tensor (H, W) for grayscale
        - 3D tensor (C, H, W) for color
        - 4D tensor (B, C, H, W) for batched images
        Must be on a CUDA device. Dimensions must be even.
    lam : float
        Regularization parameter (lambda). Higher values produce smoother results.
    epsilon : float, optional
        Smoothing parameter for Huber-like TV. Set to a small value for
        near-exact TV. Default: 1e-3
    max_iter : int, optional
        Maximum number of outer iterations. Default: 100
    inner_steps : int, optional
        Number of inner descent steps per block. Default: 5
    gap_iter : int, optional
        Check convergence every gap_iter iterations. Default: 1
    gap_tol : float, optional
        RMSE tolerance for early stopping. Default: 1e-4
    use_newton : bool, optional
        Use Newton method for grayscale images (faster). Default: True

    Returns
    -------
    Tensor
        Denoised image with same shape and dtype as input.

    Examples
    --------
    >>> import torch
    >>> from ftvp import proxtv
    >>> img = torch.randn(3, 256, 256, device="cuda")
    >>> denoised = proxtv(img, lam=0.1)

    >>> # Batched processing
    >>> batch = torch.randn(8, 3, 256, 256, device="cuda")
    >>> denoised_batch = proxtv(batch, lam=0.1)

    References
    ----------
    [1] Chambolle, A., & Pock, T. (2015). A remark on accelerated block
        coordinate descent for computing the proximity operators of a sum
        of convex functions. SMAI J. Comp. Math., 1, 29-54.
    """
    if not img.is_cuda:
        raise ValueError("Input must be on a CUDA device")

    original_shape = img.shape
    original_dtype = img.dtype

    if img.dtype not in (torch.float32, torch.float64):
        img = img.float()

    if img.dim() == 2:
        img = img.unsqueeze(0)
        was_2d = True
    else:
        was_2d = False

    if img.dim() == 3:
        img = img.unsqueeze(0)
        was_3d = True
    else:
        was_3d = False

    if img.dim() != 4:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got {len(original_shape)}D")

    B, C, H, W = img.shape

    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(f"Image dimensions must be even, got ({H}, {W})")

    if C > 3:
        raise ValueError(f"Maximum 3 channels supported, got {C}")

    ext = _get_extension()

    result = img.clone()

    for b in range(B):
        ext.prox_tv_cuda(
            result[b],
            lam,
            epsilon,
            max_iter,
            inner_steps,
            gap_iter,
            gap_tol,
            use_newton,
        )

    if was_3d:
        result = result.squeeze(0)
    if was_2d:
        result = result.squeeze(0)

    if result.dtype != original_dtype:
        result = result.to(original_dtype)

    return result
