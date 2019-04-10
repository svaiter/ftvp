#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <sys/time.h>
#include "ftvp-color.cuh"
#include "memory-color.cuh"
#include "kernels/kernels-2d-color.cuh"

// Handle errors raised by the GPU
// From http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

// Return the exec time in ms.
inline double bench_time(timeval start, timeval end) {
  return (end.tv_sec*1000.0 + (end.tv_usec / 1000.0)) - (start.tv_sec*1000.0 + (start.tv_usec / 1000.0));
}

extern
void prox_tv_2d_noalloc(struct memory_cuda *mem, struct res_cuda *res,
                        int sx, int sy, int sc, double lambda, double epsilon,
                        int Nit, int block_size, int steps, int gapiter, double gap_factor) {
  int it=0; // current iteration
  double t=1; // relaxation parameter
  double tt;
  double theta=0; // relaxation parameter
  double gap=0;
  double tau = 0.25;
  timeval start, end;

  int sxy = sx * sy;
  double w = 2 * lambda * lambda;
  double ws =  M_SQRT2*lambda; // sqrt(w)
  epsilon /= ws; //  |Du|_C threshold
  double we2 = w*epsilon*epsilon;

  double gamma = epsilon * tau;
  double gammap = gamma + gamma/(1+gamma);
  double q = gammap/(1+gamma);

  int Ke = sx/2, Le=sy/2;
  int Ko = (sx-2)/2, Lo = (sy-2)/2;

  dim3 blocks_odd((Ko+block_size-1)/block_size, (Lo+block_size-1)/block_size);
  dim3 blocks_even((Ke+block_size-1)/block_size, (Le+block_size-1)/block_size);
  dim3 threads(block_size, block_size);

  gettimeofday(&start, NULL);
  do {
    // Over-relaxation
    tt = t;
    t = .5*((1-q*t*t)+sqrt((1-q*t*t)*(1-q*t*t)+4*(1+gammap)/(1+gamma)*t*t));
    theta = (tt-1)/t * (1-(t-1)*gammap);
    if (theta > 0) {
      over_relax_eps_2d<<<blocks_odd,threads>>>(sx, sc, mem->dev_xio, mem->dev_xiobar, mem->dev_u, theta, Lo, Ko);

      // Swap xio and xiobar
      mem->dev_xioswp = mem->dev_xio;
      mem->dev_xio = mem->dev_xiobar;
      mem->dev_xiobar = mem->dev_xioswp;
    }

    // Minimization
    opt_eps_split<<<blocks_even,threads>>>(sx, sc, mem->dev_xie, mem->dev_u, epsilon, w, ws, we2, Le, Ke, steps, 1);
    opt_eps_split<<<blocks_odd,threads>>>(sx, sc, mem->dev_xiobar, mem->dev_u, epsilon, w, ws, we2, Lo, Ko, steps, 0);

    // Gap
    if (it % gapiter == 0) {
      gap_arr_eps_2d<<<blocks_odd,threads>>>(sx, sc, mem->dev_glo, mem->dev_xio, mem->dev_u, epsilon, w, ws, we2, Lo, Ko, 0);
      thrust::device_ptr<double> D = thrust::device_pointer_cast(mem->dev_glo);
      gap = thrust::reduce(D, D+Ko*Lo, (double) 0, thrust::plus<double>());
      gap_arr_eps_2d<<<blocks_even,threads>>>(sx, sc, mem->dev_gle, mem->dev_xie, mem->dev_u, epsilon, w, ws, we2, Le, Ke, 1);
      D = thrust::device_pointer_cast(mem->dev_gle);
      gap += thrust::reduce(D, D+Ke*Le, (double) 0, thrust::plus<double>());
    }
    it++;
  }  while (it < Nit && gap > sxy * 3 * gap_factor);
  gettimeofday(&end, NULL);

  double msec = bench_time(start, end);

  res->it = it;
  res->msec = msec;
  res->gap = gap;
  res->rmse = sqrt(gap/(sx*sy*3));
  return;
}

extern
res_cuda* prox_tv(int sx, int sy, int sc,
                  double *u, double lambda, double epsilon,
                  int Nit, int block_size, int steps, int gapiter, double gap_factor,
                  OptimizationMethod opt_meth) {

  struct res_cuda *res = (struct res_cuda *) malloc(sizeof(struct res_cuda));
  struct memory_cuda *mem = (struct memory_cuda *) malloc(sizeof(struct memory_cuda));

  init_memory(mem, res, sx, sy, sc, u);
  prox_tv_2d_noalloc(mem, res, sx, sy, sc, lambda, epsilon, Nit, block_size, steps, gapiter, gap_factor);

  // Memory management
  gpuErrchk( cudaMemcpy(u, mem->dev_u, sx*sy*sc*sizeof(double), cudaMemcpyDeviceToHost) );
  free_memory(mem);
  return res;
}
