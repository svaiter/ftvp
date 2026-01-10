# ftvp - Fast Total Variation Proximal

A PyTorch-native CUDA library for computing the proximal operator of isotropic Total Variation on 2D images.

## Installation

```bash
pip install .
```

Requires:
- Python >= 3.10
- PyTorch >= 2.0 with CUDA support
- CUDA toolkit (for JIT compilation)

## Quick Start

```python
import torch
from ftvp import proxtv

# Grayscale image
img = torch.randn(256, 256, device="cuda")
denoised = proxtv(img, lam=0.1)

# Color image (C, H, W)
img_color = torch.randn(3, 256, 256, device="cuda")
denoised_color = proxtv(img_color, lam=0.1)

# Batched images (B, C, H, W)
batch = torch.randn(8, 3, 256, 256, device="cuda")
denoised_batch = proxtv(batch, lam=0.1)
```

## API

```python
proxtv(img, lam, *, epsilon=1e-3, max_iter=100, inner_steps=5, gap_iter=1, gap_tol=1e-4, use_newton=True)
```

Computes:
```
prox_{λ·TV}(img) = argmin_x  0.5 * ||x - img||² + λ * TV(x)
```

**Parameters:**
- `img`: Input tensor (2D, 3D, or 4D). Must be on CUDA. Dimensions must be even.
- `lam`: Regularization parameter. Higher = smoother.
- `epsilon`: Smoothing for Huber-like TV (default: 1e-3)
- `max_iter`: Maximum iterations (default: 100)
- `inner_steps`: Inner descent steps (default: 5)
- `gap_iter`: Gap check frequency (default: 1)
- `gap_tol`: RMSE tolerance (default: 1e-4)
- `use_newton`: Use Newton method for grayscale (default: True)

**Returns:** Denoised tensor with same shape/dtype.

## Algorithm

This implements the accelerated block coordinate descent method with even/odd square decomposition. The algorithm is based on:

1. **Chambolle, A., & Pock, T. (2015)**. A remark on accelerated block coordinate descent for computing the proximity operators of a sum of convex functions. *SMAI Journal of Computational Mathematics*, 1, 29-54.

2. **Chambolle, A., Tan, P., & Vaiter, S. (2017)**. Accelerated alternating descent methods for Dykstra-like problems. *J. Math. Imaging Vision*, 59(3), 48-97.

## License

BSD-3-Clause
