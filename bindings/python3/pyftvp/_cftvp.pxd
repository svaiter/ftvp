cdef extern from "ftvp.cuh":
    cpdef enum OptimizationMethod:
        OE_SPLIT_NEWTON
        OE_SPLIT_DESCENT

    cdef struct res_cuda:
        int it
        double msec
        double gap
        double rmse

    res_cuda* prox_tv(int sx, int sy, int sz,
                      double *u, double la, double epsilon,
                      int Nit, int block_size, int steps, int gapiter, double gap_factor, OptimizationMethod opt_meth, int upd_strat)
