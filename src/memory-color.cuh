#ifndef MEMORY_COLOR_CUH
#define MEMORY_COLOR_CUH

#include <stdio.h>

/**
 * This is passed as the first argument to prox_tv_Xd_noalloc containing all
 * auxiliary variables used in the optimization.
 *
 * - dev_xie: dual variable on even square/cube
 * - dev_xio: dual variable on odd square/cube
 * - dev_u: current primal solution
 * - dev_xiobar: over-relaxed dual variable (odd square)
 * - dev_xioswp: temporary variable for swapping over-relaxation
 * - dev_gle: dual gap per square/cube even
 * - dev_glo: dual gap per square/cube odd
 *
 */
struct memory_cuda {
  double *dev_xie;
  double *dev_xio;
  double *dev_u;
  double *dev_xiobar;
  double *dev_xioswp;
  double *dev_gle;
  double *dev_glo;
};

/**
 * This is used to return informations about the optimization, such as
 * final iteration, dual gap and execution time.
 *
 * - it: final iteration.
 * - msec: execution time on GPU in msec.
 * - gap: final dual gap.
 *
 */
struct res_cuda {
  int it;
  double msec;
  double gap;
  double rmse;
};

/**
 * Initialize CUDA driver.
 *
 * Useful in a benchmarking setting, since the overhead of CUDA initialization can be quite
 * time consuming.
 *
 */
extern void init_cuda();

/**
 * Initialize FTVP auxiliary variables on the GPU.
 *
 * This function is typically called before prox_tv_2d_noalloc or prox_tv_3d_noalloc.
 *
 * @param mem Auxiliary variables.
 * @param res Output data structure.
 * @param sx, sy, sz Dimensions of the image.
 * @param u Initial image on CPU.
 *
 */
extern void init_memory(struct memory_cuda *mem, struct res_cuda *res, int sx, int sy, int sc, double* u);

/**
 * Destroy FTVP auxiliary variables on the GPU.
 *
 * This function is typically called after prox_tv_2d_noalloc or prox_tv_3d_noalloc.
 *
 * @param mem Auxiliary variables.
 *
 */
extern void free_memory(struct memory_cuda *mem);

#endif
