#ifndef KERNELS_EPS_2D_CUH
#define KERNELS_EPS_2D_CUH

__global__ void opt_eps_split(int sx, double *xi, double *u, double epsilon, double w, double ws, double we2, int L, int K, int steps, int use_newton, int is_even_step);

__global__ void over_relax_eps_2d(int sx, double *xio, double *xiobar, double *u, double theta, int Lo, int Ko, int is_even_step);

__global__ void gap_arr_eps_2d(int sx, double* gl, double *xie, double *u, double epsilon, double w, double ws, double we2, int Le, int Ke, int is_even_step);

#endif
