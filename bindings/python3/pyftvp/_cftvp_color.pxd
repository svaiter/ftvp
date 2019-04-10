cdef extern from "ftvp-color.cuh":
    cpdef enum OptimizationMethod:
        OE_SPLIT_NEWTON
        OE_SPLIT_DESCENT

    cdef struct res_cuda:
        int it
        double msec
        double gap
        double rmse

    res_cuda* prox_tv(int sx, int sy, int sc,
                      double *u, double la, double eps,
                      int Nit, int block_size, int steps, int gapiter, double gap_factor, OptimizationMethod opt_meth)
