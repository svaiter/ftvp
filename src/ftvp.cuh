#ifndef FTVP_CUH
#define FTVP_CUH

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h> // for fabs
#include "memory.cuh"

/**
 * Type of optimization
 **/
typedef enum {OE_SPLIT_NEWTON,
              OE_SPLIT_DESCENT} OptimizationMethod;

/**
 * Update strategy
 **/
const int UPDATE_STD = 0;
const int UPDATE_CONSTANT = 1;
const int UPDATE_VARYING = 2;

/**
 * Compute the proximity operator associated to the epsilon-isotropic
 * Total Variation in 2D.
 *
 * See prox_tv documentation for an detailed explanation. This function is used with
 * GPU memory already allocated.
 * You should called init_memory before it, and free_memory after.
 *
 * @param mem Auxiliary variables.
 * @param res Output data structure.
 * @param sx, sy Dimensions of the image.
 * @param lambda Regularization parameter.
 * @param epsilon smoothing parameter.
 * @param Nit Maximum number of iterations.
 * @param block_size GPU computation block size.
 * @param steps Inner Newton number of steps.
 * @param gapiter Test gap each gapiter iteration.
 * @param gap_factor Objective gap bound (stop at size_image * gap_factor)
 * @param use_newton if > 0, use newton otherwise descent step.
 * @param upd_strat method to select the relaxation
 *
 */
extern void prox_tv_2d_noalloc(struct memory_cuda *mem, struct res_cuda *res,
                               int sx, int sy, double lambda, double epsilon,
                               int Nit, int block_size, int steps, int gapiter, double gap_factor, int use_newton, int upd_strat);

/**
 * Compute the proximity operator associated to the isotropic Total Variation epsilon in 2D/3D.
 *
 * Solve the problem
 * uargmin_x lambda/2 ||x - u||_2^2 + TV(x)
 * where TV is the isotropic Total Variation.
 *
 * @param sx, sy, sz Dimensions of the image.
 * @param u Array to be proceed.
 * @param lambda Regularization parameter.
 * @param epsilon Smoothing parameter.
 * @param Nit Maximum number of iterations.
 * @param block_size GPU computation block size.
 * @param steps Inner Newton number of steps.
 * @param gapiter Test gap each gapiter iteration.
 * @param gap_factor Objective gap bound (stop at size_image * gap_factor)
 * @param opt_meth Method of optimization.
 * @param upd_strat method to select the relaxation
 *
 */
extern res_cuda* prox_tv(int sx, int sy, int sz,
                         double *u, double lambda, double epsilon,
                         int Nit, int block_size, int steps, int gapiter, double gap_factor,
                         OptimizationMethod opt_meth, int upd_strat);

#endif
