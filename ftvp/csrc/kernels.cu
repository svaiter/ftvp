/*
 * Fast Total Variation Proximal Operator - CUDA Kernels
 *
 * This implements the accelerated block coordinate descent method from:
 * - Chambolle & Pock (2015): SMAI J. Comp. Math., 1, 29-54
 * - Chambolle, Tan & Vaiter (2017): J. Math. Imag. Vision, 59(3), 48-97
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define NITD 5

template <typename scalar_t>
__device__ __forceinline__ scalar_t sq(scalar_t x) {
    return x * x;
}

template <typename scalar_t>
__device__ void optimsquare_descent(
    scalar_t* __restrict__ u,
    scalar_t* __restrict__ xi,
    int sc,
    scalar_t w,
    scalar_t epsilon,
    int steps
) {
    scalar_t r = scalar_t(1) / (scalar_t(1) + epsilon / scalar_t(4));

    for (int c = 0; c < sc; c++) {
        int off = c * 4;
        u[off + 0] -= xi[off + 0] - xi[off + 3];
        u[off + 1] -= xi[off + 1] - xi[off + 0];
        u[off + 2] -= xi[off + 2] - xi[off + 1];
        u[off + 3] -= xi[off + 3] - xi[off + 2];
    }

    scalar_t nxi[12];
    for (int it = 0; it < steps; it++) {
        scalar_t no = 0;
        for (int c = 0; c < sc; c++) {
            int off = c * 4;
            nxi[off + 0] = r * (scalar_t(0.5) * xi[off + 0] + scalar_t(0.25) * (xi[off + 3] + xi[off + 1] + u[off + 1] - u[off + 0]));
            nxi[off + 1] = r * (scalar_t(0.5) * xi[off + 1] + scalar_t(0.25) * (xi[off + 0] + xi[off + 2] + u[off + 2] - u[off + 1]));
            nxi[off + 2] = r * (scalar_t(0.5) * xi[off + 2] + scalar_t(0.25) * (xi[off + 1] + xi[off + 3] + u[off + 3] - u[off + 2]));
            nxi[off + 3] = r * (scalar_t(0.5) * xi[off + 3] + scalar_t(0.25) * (xi[off + 2] + xi[off + 0] + u[off + 0] - u[off + 3]));
        }

        for (int i = 0; i < sc * 4; i++) {
            no += sq(nxi[i]);
        }

        scalar_t scale = (no > w) ? sqrt(w / no) : scalar_t(1);
        for (int i = 0; i < sc * 4; i++) {
            xi[i] = nxi[i] * scale;
        }
    }

    for (int c = 0; c < sc; c++) {
        int off = c * 4;
        u[off + 0] += xi[off + 0] - xi[off + 3];
        u[off + 1] += xi[off + 1] - xi[off + 0];
        u[off + 2] += xi[off + 2] - xi[off + 1];
        u[off + 3] += xi[off + 3] - xi[off + 2];
    }
}

template <typename scalar_t>
__device__ void optimsquare_newton(
    scalar_t* __restrict__ u,
    scalar_t* __restrict__ xi,
    scalar_t w,
    scalar_t epsilon,
    int steps
) {
    u[0] -= xi[0] - xi[3];
    u[1] -= xi[1] - xi[0];
    u[2] -= xi[2] - xi[1];
    u[3] -= xi[3] - xi[2];

    scalar_t z0 = u[3] - u[1];
    scalar_t z1 = u[2] - u[0];
    scalar_t z2 = (u[0] + u[2]) - (u[3] + u[1]);

    scalar_t a = z0 * z0 + z1 * z1;
    scalar_t b = z2 * z2;

    scalar_t tmu = epsilon;
    scalar_t no = a / sq(scalar_t(2) + tmu) + b / sq(scalar_t(4) + tmu);

    if (no > w) {
        for (int i = 0; i < steps; i++) {
            scalar_t noa = a / sq(scalar_t(2) + tmu);
            scalar_t nob = b / sq(scalar_t(4) + tmu);
            scalar_t dno = noa / (scalar_t(2) + tmu) + nob / (scalar_t(4) + tmu);
            tmu += scalar_t(0.5) * (noa + nob - w) / dno;
            tmu = max(epsilon, tmu);
        }
    }

    scalar_t t0 = scalar_t(0.5) * z0 / (scalar_t(2) + tmu);
    scalar_t t1 = scalar_t(0.5) * z1 / (scalar_t(2) + tmu);
    scalar_t t2 = scalar_t(0.5) * z2 / (scalar_t(4) + tmu);

    scalar_t tm = t0 - t1;
    scalar_t tp = t0 + t1;

    xi[0] = -tm - t2;
    xi[1] = tp + t2;
    xi[2] = tm - t2;
    xi[3] = t2 - tp;

    scalar_t s = sq(xi[0]) + sq(xi[1]) + sq(xi[2]) + sq(xi[3]);
    if (s > w) {
        scalar_t factor = sqrt(w / s);
        xi[0] *= factor;
        xi[1] *= factor;
        xi[2] *= factor;
        xi[3] *= factor;
    }

    u[0] += xi[0] - xi[3];
    u[1] += xi[1] - xi[0];
    u[2] += xi[2] - xi[1];
    u[3] += xi[3] - xi[2];
}

template <typename scalar_t>
__global__ void prox_tv_even_kernel(
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> u,
    scalar_t* __restrict__ xi_even,
    scalar_t w,
    scalar_t epsilon,
    int steps,
    bool use_newton,
    int Ke, int Le
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;

    if (k >= Ke || l >= Le) return;

    int sc = u.size(0);
    int px = 2 * k;
    int py = 2 * l;

    int xi_base = (l * Ke + k) * (4 * sc);

    scalar_t local_u[12];
    scalar_t local_xi[12];

    for (int c = 0; c < sc; c++) {
        int off = c * 4;
        local_u[off + 0] = u[c][py][px];
        local_u[off + 1] = u[c][py][px + 1];
        local_u[off + 2] = u[c][py + 1][px + 1];
        local_u[off + 3] = u[c][py + 1][px];

        local_xi[off + 0] = xi_even[xi_base + off + 0];
        local_xi[off + 1] = xi_even[xi_base + off + 1];
        local_xi[off + 2] = xi_even[xi_base + off + 2];
        local_xi[off + 3] = xi_even[xi_base + off + 3];
    }

    if (use_newton && sc == 1) {
        optimsquare_newton<scalar_t>(local_u, local_xi, w, epsilon, steps);
    } else {
        optimsquare_descent<scalar_t>(local_u, local_xi, sc, w, epsilon, steps);
    }

    for (int c = 0; c < sc; c++) {
        int off = c * 4;
        u[c][py][px] = local_u[off + 0];
        u[c][py][px + 1] = local_u[off + 1];
        u[c][py + 1][px + 1] = local_u[off + 2];
        u[c][py + 1][px] = local_u[off + 3];

        xi_even[xi_base + off + 0] = local_xi[off + 0];
        xi_even[xi_base + off + 1] = local_xi[off + 1];
        xi_even[xi_base + off + 2] = local_xi[off + 2];
        xi_even[xi_base + off + 3] = local_xi[off + 3];
    }
}

template <typename scalar_t>
__global__ void prox_tv_odd_kernel(
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> u,
    scalar_t* __restrict__ xi_odd,
    scalar_t* __restrict__ udiff,
    scalar_t w,
    scalar_t epsilon,
    scalar_t theta,
    int steps,
    bool use_newton,
    int Ko, int Lo,
    int sx, int sy
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;

    if (k >= Ko || l >= Lo) return;

    int sc = u.size(0);
    int px = 2 * k + 1;
    int py = 2 * l + 1;

    int xi_base = (l * Ko + k) * (4 * sc);
    int udiff_base = (l * Ko + k) * (4 * sc);

    scalar_t local_u[12];
    scalar_t local_xi[12];

    for (int c = 0; c < sc; c++) {
        int off = c * 4;
        local_u[off + 0] = u[c][py][px] + theta * udiff[udiff_base + off + 0];
        local_u[off + 1] = u[c][py][px + 1] + theta * udiff[udiff_base + off + 1];
        local_u[off + 2] = u[c][py + 1][px + 1] + theta * udiff[udiff_base + off + 2];
        local_u[off + 3] = u[c][py + 1][px] + theta * udiff[udiff_base + off + 3];

        local_xi[off + 0] = xi_odd[xi_base + off + 0];
        local_xi[off + 1] = xi_odd[xi_base + off + 1];
        local_xi[off + 2] = xi_odd[xi_base + off + 2];
        local_xi[off + 3] = xi_odd[xi_base + off + 3];
    }

    if (use_newton && sc == 1) {
        optimsquare_newton<scalar_t>(local_u, local_xi, w, epsilon, steps);
    } else {
        optimsquare_descent<scalar_t>(local_u, local_xi, sc, w, epsilon, steps);
    }

    for (int c = 0; c < sc; c++) {
        int off = c * 4;
        u[c][py][px] = local_u[off + 0] - theta * udiff[udiff_base + off + 0];
        u[c][py][px + 1] = local_u[off + 1] - theta * udiff[udiff_base + off + 1];
        u[c][py + 1][px + 1] = local_u[off + 2] - theta * udiff[udiff_base + off + 2];
        u[c][py + 1][px] = local_u[off + 3] - theta * udiff[udiff_base + off + 3];

        xi_odd[xi_base + off + 0] = local_xi[off + 0];
        xi_odd[xi_base + off + 1] = local_xi[off + 1];
        xi_odd[xi_base + off + 2] = local_xi[off + 2];
        xi_odd[xi_base + off + 3] = local_xi[off + 3];
    }
}

template <typename scalar_t>
__global__ void compute_udiff_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> u_before,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> u_after,
    scalar_t* __restrict__ udiff,
    int Ko, int Lo
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;

    if (k >= Ko || l >= Lo) return;

    int sc = u_before.size(0);
    int px = 2 * k + 1;
    int py = 2 * l + 1;

    int udiff_base = (l * Ko + k) * (4 * sc);

    for (int c = 0; c < sc; c++) {
        int off = c * 4;
        udiff[udiff_base + off + 0] = u_after[c][py][px] - u_before[c][py][px];
        udiff[udiff_base + off + 1] = u_after[c][py][px + 1] - u_before[c][py][px + 1];
        udiff[udiff_base + off + 2] = u_after[c][py + 1][px + 1] - u_before[c][py + 1][px + 1];
        udiff[udiff_base + off + 3] = u_after[c][py + 1][px] - u_before[c][py + 1][px];
    }
}

template <typename scalar_t>
__global__ void compute_gap_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> u,
    const scalar_t* __restrict__ xi,
    scalar_t* __restrict__ gap_arr,
    scalar_t epsilon,
    scalar_t w,
    scalar_t ws,
    scalar_t we2,
    int K, int L,
    bool is_even
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int l = blockIdx.y * blockDim.y + threadIdx.y;

    if (k >= K || l >= L) return;

    int sc = u.size(0);
    int px = is_even ? (2 * k) : (2 * k + 1);
    int py = is_even ? (2 * l) : (2 * l + 1);

    int xi_base = (l * K + k) * (4 * sc);
    scalar_t local_gap = 0;

    scalar_t b = 0, cc = 0, d = 0;

    for (int c = 0; c < sc; c++) {
        int off = c * 4;
        scalar_t u0 = u[c][py][px];
        scalar_t u1 = u[c][py][px + 1];
        scalar_t u2 = u[c][py + 1][px + 1];
        scalar_t u3 = u[c][py + 1][px];

        scalar_t a0 = u0 - u3;
        scalar_t a1 = u1 - u0;
        scalar_t a2 = u2 - u1;
        scalar_t a3 = u3 - u2;

        b += sq(a0) + sq(a1) + sq(a2) + sq(a3);
        cc += xi[xi_base + off + 0] * a0 + xi[xi_base + off + 1] * a1 + 
              xi[xi_base + off + 2] * a2 + xi[xi_base + off + 3] * a3;
        d += sq(xi[xi_base + off + 0]) + sq(xi[xi_base + off + 1]) + 
             sq(xi[xi_base + off + 2]) + sq(xi[xi_base + off + 3]);
    }

    local_gap = epsilon * scalar_t(0.5) * d - cc;
    if (b < we2) {
        local_gap += scalar_t(0.5) * b / epsilon;
    } else {
        local_gap += ws * sqrt(b) - scalar_t(0.5) * epsilon * w;
    }

    gap_arr[l * K + k] = local_gap;
}

void prox_tv_cuda(
    torch::Tensor u,
    double lambda,
    double epsilon,
    int max_iter,
    int inner_steps,
    int gap_iter,
    double gap_tol,
    bool use_newton
) {
    TORCH_CHECK(u.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(u.dim() == 3, "Input must be 3D (C, H, W)");

    int sc = u.size(0);
    int sy = u.size(1);
    int sx = u.size(2);

    TORCH_CHECK(sx % 2 == 0 && sy % 2 == 0, "Image dimensions must be even");

    int Ke = sx / 2;
    int Le = sy / 2;
    int Ko = (sx - 2) / 2;
    int Lo = (sy - 2) / 2;

    auto options = u.options();

    auto xi_even = torch::zeros({Ke * Le * 4 * sc}, options);
    auto xi_odd = torch::zeros({Ko * Lo * 4 * sc}, options);
    auto udiff = torch::zeros({Ko * Lo * 4 * sc}, options);
    auto u_before = torch::empty_like(u);
    auto gap_even = torch::zeros({Ke * Le}, options);
    auto gap_odd = torch::zeros({Ko * Lo}, options);

    dim3 threads(16, 16);
    dim3 blocks_even((Ke + 15) / 16, (Le + 15) / 16);
    dim3 blocks_odd((Ko + 15) / 16, (Lo + 15) / 16);

    double tau = 0.25;
    double mug = tau * epsilon * sqrt(2.0);
    double muf = mug / (1.0 + mug);
    double q = (muf + mug) / (1.0 + muf);

    double t = 1.0;

    AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "prox_tv_cuda", ([&] {
        scalar_t w = scalar_t(2.0 * lambda * lambda);
        scalar_t ws = scalar_t(sqrt(2.0) * lambda);
        scalar_t eps_scaled = scalar_t(epsilon) / ws;
        scalar_t we2 = w * eps_scaled * eps_scaled;

        for (int it = 0; it < max_iter; it++) {
            double tt = t;
            t = 0.5 * ((1.0 - q * t * t) + sqrt((1.0 - q * t * t) * (1.0 - q * t * t) + 4.0 * t * t));
            double theta = (tt - 1.0) / t * (1.0 - t * (muf + mug) + mug) / (1.0 - tau * muf);

            u_before.copy_(u);

            prox_tv_even_kernel<scalar_t><<<blocks_even, threads>>>(
                u.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                xi_even.data_ptr<scalar_t>(),
                w, eps_scaled, inner_steps, use_newton && sc == 1,
                Ke, Le
            );

            if (Ko > 0 && Lo > 0) {
                compute_udiff_kernel<scalar_t><<<blocks_odd, threads>>>(
                    u_before.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                    u.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                    udiff.data_ptr<scalar_t>(),
                    Ko, Lo
                );

                prox_tv_odd_kernel<scalar_t><<<blocks_odd, threads>>>(
                    u.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                    xi_odd.data_ptr<scalar_t>(),
                    udiff.data_ptr<scalar_t>(),
                    w, eps_scaled, scalar_t(theta), inner_steps, use_newton && sc == 1,
                    Ko, Lo, sx, sy
                );
            }

            if ((it + 1) % gap_iter == 0) {
                compute_gap_kernel<scalar_t><<<blocks_even, threads>>>(
                    u.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                    xi_even.data_ptr<scalar_t>(),
                    gap_even.data_ptr<scalar_t>(),
                    eps_scaled, w, ws, we2,
                    Ke, Le, true
                );

                if (Ko > 0 && Lo > 0) {
                    compute_gap_kernel<scalar_t><<<blocks_odd, threads>>>(
                        u.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                        xi_odd.data_ptr<scalar_t>(),
                        gap_odd.data_ptr<scalar_t>(),
                        eps_scaled, w, ws, we2,
                        Ko, Lo, false
                    );
                }

                cudaDeviceSynchronize();
                double gap = gap_even.sum().item<double>() + gap_odd.sum().item<double>();
                double rmse = sqrt(gap / (sx * sy));

                if (rmse < gap_tol) {
                    break;
                }
            }
        }
    }));

    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prox_tv_cuda", &prox_tv_cuda, "TV prox operator (CUDA)",
        py::arg("u"),
        py::arg("lambda"),
        py::arg("epsilon"),
        py::arg("max_iter"),
        py::arg("inner_steps"),
        py::arg("gap_iter"),
        py::arg("gap_tol"),
        py::arg("use_newton")
    );
}
