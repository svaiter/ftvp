#ifndef KERNELS_EPS_2D_COLOR_CUH
#define KERNELS_EPS_2D_COLOR_CUH

__global__ void opt_eps_split(int sx, int sc, double *xi, double *u, double epsilon, double w, double ws, double we2, int L, int K, int steps, int is_even_step);

__global__ void over_relax_eps_2d(int sx, int sc, double *xio, double *xiobar, double *u, double theta, int Lo, int Ko);

__global__ void gap_arr_eps_2d(int sx, int sc, double* gl, double *xie, double *u, double epsilon, double w, double ws, double we2, int Le, int Ke, int is_even_step);

#endif
