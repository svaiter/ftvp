#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <getopt.h>
#include "../src/ftvp-color.cuh"
#include "io_png.h"

struct globalArgs_t {
  int blocks;
  int iters;
  int steps;
  int gapiter;
  char *input_file;
  double lambda;
  double epsilon;
  int verbosity;
} globalArgs;

static const char *optString = "b:i:s:g:e:vh?";

void display_usage( void )
{
  fprintf(stdout,"usage: fast_tv_prox [-i GLOBAL ITERATIONS] [-b BLOCK SIZE] [-s NEWTON STEPS] [-g GAP ITER] [-e EPSILON] [-v] filename lambda\n");
  exit( EXIT_FAILURE );
}

int main(int argc,char **argv)
{
  unsigned int *im, max;
  double *img;
  size_t nx, ny, nc;
  double *u, lambda, epsilon;
  double w;
  int sx,sy,sc,sxy;
  int iters;
  int steps;
  int gapiter;
  int blocks;
  timeval start, end;

  int opt = 0;
  globalArgs.blocks = 16;
  globalArgs.iters = 10;
  globalArgs.steps = 7;
  globalArgs.gapiter = 1;
  globalArgs.lambda = 0.0;
  globalArgs.epsilon = 0.0;
  globalArgs.input_file = NULL;
  globalArgs.verbosity = 0;

  opt = getopt( argc, argv, optString );
  while( opt != -1 ) {
    switch( opt ) {                
    case 'b':
      globalArgs.blocks = atoi(optarg);
      break;
                
    case 'i':
      globalArgs.iters = atoi(optarg);
      break;

    case 's':
      globalArgs.steps = atoi(optarg);
      break;

    case 'g':
      globalArgs.gapiter = atoi(optarg);
      break;

    case 'e':
      globalArgs.epsilon = atof(optarg);
      break;

    case 'v':
      globalArgs.verbosity++;
      break;

    case 'h':   /* fall-through is intentional */
    case '?':
      display_usage();
      break;
                
    default:
      /* You won't actually get here. */
      break;
    }
    opt = getopt( argc, argv, optString );
  }
    
  globalArgs.input_file = argv[optind];
  globalArgs.lambda = atof(argv[optind+1]);
  iters = globalArgs.iters;
  steps = globalArgs.steps;
  gapiter = globalArgs.gapiter;
  blocks = globalArgs.blocks;
  lambda = globalArgs.lambda;
  epsilon = globalArgs.epsilon;

  // max = readimg2(globalArgs.input_file, &sx,&sy,&im);
  img = io_png_read_flt_opt(globalArgs.input_file, &nx, &ny, &nc, IO_PNG_OPT_RGB);
  sx = (int)nx;
  sy = (int)ny;
  sc = (int)nc;

  sxy=sx*sy;
  if (globalArgs.verbosity > 0) {
    fprintf(stdout,"Image: %s read.\n", globalArgs.input_file);
    fprintf(stdout,"Size: %d x %d x %d = %d x %d [0-%d]\n",sx,sy,sc,sxy,sc,max);
    fprintf(stdout,"Computing prox_TV(im, %f) with:\n", lambda);
    fprintf(stdout," - maximum iterations: %d\n", iters);
    fprintf(stdout," - inner iterations: %d\n", steps);
    fprintf(stdout," - GPU block size: %d x %d\n", blocks, blocks);
  }

  u=(double *) malloc(sxy*sc*sizeof(double));
  for (int i=0; i<sx; i++) {
    for (int j=0; j<sy; j++) {
        u[i*sc + j*sc*sx]     = (double) img[i+j*sx] * 255;
        u[1 + i*sc + j*sc*sx] = (double) img[i+sx*j+sx*sy] * 255;
        u[2 + i*sc + j*sc*sx] = (double) img[i+sx*j+2*sx*sy] * 255;
    }
  }

  init_cuda(); // Init cuda driver.
  
  gettimeofday(&start, NULL);
  res_cuda *res = prox_tv(sx, sy, 3, u, lambda, epsilon, iters, blocks, steps, gapiter, 0.25, OE_SPLIT_NEWTON);
  gettimeofday(&end, NULL);

  if (globalArgs.verbosity > 0) {
    double msec = (end.tv_sec*1000.0 + (end.tv_usec / 1000.0)) -
      (start.tv_sec*1000.0 + (start.tv_usec / 1000.0));
    fprintf(stdout,"Computing done:\n");
    fprintf(stdout," -> Final iteration: %d\n", res->it);
    fprintf(stdout," -> Time elapsed (without memory allocation): %f ms\n", res->msec);
    fprintf(stdout," -> Time elapsed (with memory allocation): %f ms.\n", msec);
    fprintf(stdout," -> Final gap: %f\n", res->gap);
    fprintf(stdout," -> Final normalized gap: %f\n", sqrt((res->gap)/(sx * sy)));
    fprintf(stdout, "i/m/g: %d|%f|%f\n",res->it,res->msec,res->gap);
  }

  for (int i=0; i<sx; i++) {
    for (int j=0; j<sy; j++) {
      img[i+j*sx]         = u[i*sc + j*sc*sx] / 255;
      img[i+sx*j+sx*sy]   = u[1 + i*sc + j*sc*sx] / 255;
      img[i+sx*j+2*sx*sy] = u[2 + i*sc + j*sc*sx] / 255;
    }
  }
  io_png_write_flt("double_rgb.png", img, nx, ny, 3);

  exit(0);
}
