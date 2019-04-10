// parallel color total variation - A. Chambolle & T. Pock,
// Pauline Tan, Samuel Vaiter [partial descent steps]
// typical compilation:
// cc -std=gnu99 -fopenmp -O -DVERB1 -DNITD=6 -o [your executable] TV4colorCPTV.c [your main.c] -lm
// possible flags -DVARYING -DNITD=# -DCONSTANT -DVERB1 -DVERB2
// -DNITD=5 : number of inner loops  -DVARYING varying steps
//  usage TV4color(image *,size_x,size_y,lambda,epsilon,MAX ITER.,RMSE)
// epsilon = smoothing [Huber total variation]
// image = double of size size_x*size_y*3 [RGB]
// cite:
// Antonin Chambolle and Thomas Pock.
// A remark on accelerated block coordinate descent for computing the proximity operators of a sum of convex functions.
// SMAI Journal of Computational Mathematics, 1 :29--54, 2015.
// and
// Antonin Chambolle, Pauline Tan, and Samuel Vaiter.
// Accelerated alternating descent methods for Dykstra-like problems. 
// J. Math. Imag. Vision , 59(3):48--497, november 2017.

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <math.h>
#include <omp.h>

#ifndef NITD
#define NITD 5
#endif

// computes Huber-TV 
// usage TV4color(image *,size_x,size_y,lambda,epsilon,MAX ITER.,RMSE)
void TV4color(double *,int,int,double,double, int, double );

double SQ(double);
void optimsquare(double R[4], double G[4], double B[4],
		 double xiR[4], double xiG[4], double xiB[4],
		 double w,double epsilon);

inline double SQ(double x) { return x*x;}

void optimsquare(double uR[4], double uG[4], double uB[4],
		 double xiR[4], double xiG[4], double xiB[4],
		 double w,double epsilon)
{
  double no,nxiR[4],nxiG[4],nxiB[4];
  double r = 1./(1+epsilon/4.);
  uR[0] -= xiR[0]-xiR[3];
  uG[0] -= xiG[0]-xiG[3];
  uB[0] -= xiB[0]-xiB[3];
  for (int i=1;i<=3;i++) { 
    uR[i] -= xiR[i]-xiR[i-1];
    uG[i] -= xiG[i]-xiG[i-1];
    uB[i] -= xiB[i]-xiB[i-1];
  }
  for (int it=0;it<NITD;it++) {
    //  ... descente ...
    if ((no 
	 = SQ(nxiR[0]=r*(.5*xiR[0]+.25*(xiR[3]+xiR[1]+ uR[1]-uR[0])))
	 + SQ(nxiR[1]=r*(.5*xiR[1]+.25*(xiR[0]+xiR[2]+ uR[2]-uR[1])))
	 + SQ(nxiR[2]=r*(.5*xiR[2]+.25*(xiR[1]+xiR[3]+ uR[3]-uR[2])))
	 + SQ(nxiR[3]=r*(.5*xiR[3]+.25*(xiR[2]+xiR[0]+ uR[0]-uR[3])))
	 + SQ(nxiG[0]=r*(.5*xiG[0]+.25*(xiG[3]+xiG[1]+ uG[1]-uG[0])))
	 + SQ(nxiG[1]=r*(.5*xiG[1]+.25*(xiG[0]+xiG[2]+ uG[2]-uG[1])))
	 + SQ(nxiG[2]=r*(.5*xiG[2]+.25*(xiG[1]+xiG[3]+ uG[3]-uG[2])))
	 + SQ(nxiG[3]=r*(.5*xiG[3]+.25*(xiG[2]+xiG[0]+ uG[0]-uG[3])))
	 + SQ(nxiB[0]=r*(.5*xiB[0]+.25*(xiB[3]+xiB[1]+ uB[1]-uB[0])))
	 + SQ(nxiB[1]=r*(.5*xiB[1]+.25*(xiB[0]+xiB[2]+ uB[2]-uB[1])))
	 + SQ(nxiB[2]=r*(.5*xiB[2]+.25*(xiB[1]+xiB[3]+ uB[3]-uB[2])))
	 + SQ(nxiB[3]=r*(.5*xiB[3]+.25*(xiB[2]+xiB[0]+ uB[0]-uB[3]))))
	> w) {
      no = sqrt(w/no);
    } else {
      no = 1;
    }
    xiR[0]= nxiR[0]*no;
    xiR[1]= nxiR[1]*no;
    xiR[2]= nxiR[2]*no;
    xiR[3]= nxiR[3]*no;
    xiG[0]= nxiG[0]*no;
    xiG[1]= nxiG[1]*no;
    xiG[2]= nxiG[2]*no;
    xiG[3]= nxiG[3]*no;
    xiB[0]= nxiB[0]*no;
    xiB[1]= nxiB[1]*no;
    xiB[2]= nxiB[2]*no;
    xiB[3]= nxiB[3]*no;
  }
  uR[0] += xiR[0]-xiR[3];
  uG[0] += xiG[0]-xiG[3];
  uB[0] += xiB[0]-xiB[3];
  for (int i=1;i<=3;i++) { 
    uR[i] += xiR[i]-xiR[i-1];
    uG[i] += xiG[i]-xiG[i-1];
    uB[i] += xiB[i]-xiB[i-1];
  }
}

#define _USE_MATH_DEFINES

void TV4color(double *u, int sx, int sy, double lambda, double epsilon,
	    int NIT, double MAXERR)
{
  double t;
  double *xie, *xio, *udif;
  double w,ws,we2;
  double gap,rmse;
  int sxy;
  int it=0;

  fprintf(stdout,"maxerr %lf\n",MAXERR);
  // int check=16, itcheck;

  w =   2*lambda*lambda;
  ws =  M_SQRT2*lambda; // sqrt(w);
  epsilon /= ws; // seuil du |Du|_C
  we2 = w*epsilon*epsilon; // ou epsilon^2 avant division...

  double tau = .25; // normalement
#ifdef VARYING // seems the best for exact minimization
  double gamma = epsilon*tau*M_SQRT2;
  double gammap = gamma + gamma/(1+gamma);
  double q = gammap/(1+gamma);
#else // gives exactly the same as above!!
  double mug=tau*epsilon*M_SQRT2, muf = muf/(1+muf);
  double q = (muf+mug)/(1+muf);
#endif

  sxy=sx*sy;

  int Ke = sx/2, Le=sy/2;
  int Ko = (sx-2)/2, Lo = (sy-2)/2;

  xie = (double *) malloc(3*Ke*Le*4*sizeof(double));
  xio = (double *) malloc(3*Ko*Lo*4*sizeof(double));
  udif=(double *) malloc(3*sxy*sizeof(double));
  memset(xie,0,3*Ke*Le*4*sizeof(double));
  memset(xio,0,3*Ko*Lo*4*sizeof(double));
  //  memset(udif,0,sxy*sizeof(double));

  double *uR=u, *uG=u+sxy, *uB=u+2*sxy;
  double *udifR=udif, *udifG=udif+sxy, *udifB=udif+2*sxy;
  double *xieR=xie, *xieG=xie+Ke*Le*4, *xieB=xie+Ke*Le*8;
  double *xioR=xio, *xioG=xio+Ko*Lo*4, *xioB=xio+Ko*Lo*8;

  // omp_set_num_threads(6); default is slightly faster
  it=0;
  t=1; // iterative sequence
  do {
    it++;
#if defined(CONSTANT) // seems ok for few descent
    double theta = 1/(1+tau*sqrt(epsilon));
    theta *= theta;
#elif defined(VARYING) // seems good for exact minimization
    double tt=t;
    t = .5*((1-q*t*t)+sqrt(SQ(1-q*t*t)+4*(1+gammap)/(1+gamma)*t*t));
    double theta = (tt-1)/t * (1-(t-1)*gammap);
#else // seems good compromise
    double tt = t;
    t = .5*((1.-q*t*t)+sqrt(SQ(1-q*t*t)+4*t*t));
    //    double theta = (tt-1)/t * (1-(t-1)*tau*epsilon);
    double theta = (tt-1)/t * (1-t*(muf+mug)+mug)/(1-tau*muf);
#endif
    // itcheck = (it%check)==1;
    {
      //if (itcheck)
      gap=0;
#pragma omp parallel for reduction(+:gap)
      for (int l=0;l<Le;l++) {
	int i= l*2*sx;
	int kk=l*Ke;
	for (int k=0;k<Ke;k++,kk++,i+=2) {
	// optimiser un carre
	  double aruR[4] = {uR[i], uR[i+1], uR[i+1+sx], uR[i+sx]};
	  double aruG[4] = {uG[i], uG[i+1], uG[i+1+sx], uG[i+sx]};
	  double aruB[4] = {uB[i], uB[i+1], uB[i+1+sx], uB[i+sx]};
	  optimsquare(aruR,aruG,aruB,xieR+kk*4,xieG+kk*4,xieB+kk*4,w,epsilon);

	  // pour la surrelaxation du xi_even
	  udifR[i] = (aruR[0]-uR[i]);
	  udifR[i+1] = (aruR[1]-uR[i+1]);
	  udifR[i+1+sx] = (aruR[2]-uR[i+1+sx]);
	  udifR[i+sx] = (aruR[3]-uR[i+sx]);
	  uR[i]=aruR[0]; uR[i+1]=aruR[1]; uR[i+1+sx]=aruR[2]; uR[i+sx]=aruR[3];
	  udifG[i] = (aruG[0]-uG[i]);
	  udifG[i+1] = (aruG[1]-uG[i+1]);
	  udifG[i+1+sx] = (aruG[2]-uG[i+1+sx]);
	  udifG[i+sx] = (aruG[3]-uG[i+sx]);
	  uG[i]=aruG[0]; uG[i+1]=aruG[1]; uG[i+1+sx]=aruG[2]; uG[i+sx]=aruG[3];
	  udifB[i] = (aruB[0]-uB[i]);
	  udifB[i+1] = (aruB[1]-uB[i+1]);
	  udifB[i+1+sx] = (aruB[2]-uB[i+1+sx]);
	  udifB[i+sx] = (aruB[3]-uB[i+sx]);
	  uB[i]=aruB[0]; uB[i+1]=aruB[1]; uB[i+1+sx]=aruB[2]; uB[i+sx]=aruB[3];

	  // if (itcheck)
	  { // calcul du gap  TV_e(Du) - <xi,Du> + e*xi^2/2
	    double aR[4],*xiR=xieR+4*kk,b,c,d;
	    double aG[4],*xiG=xieG+4*kk;
	    double aB[4],*xiB=xieB+4*kk;
	    aR[3] = aruR[0]-aruR[3];
	    aG[3] = aruG[0]-aruG[3];
	    aB[3] = aruB[0]-aruB[3];
	    b = aR[3]*aR[3]+aG[3]*aG[3]+aB[3]*aB[3];
	    c = xiR[3]*aR[3]+xiG[3]*aG[3]+xiB[3]*aB[3];
	    d = xiR[3]*xiR[3]+xiG[3]*xiG[3]+xiB[3]*xiB[3];
	    for (int m=0;m<3;m++) {
	      aR[m] = aruR[m+1]-aruR[m];
	      aG[m] = aruG[m+1]-aruG[m];
	      aB[m] = aruB[m+1]-aruB[m];
	      b += aR[m]*aR[m]+aG[m]*aG[m]+aB[m]*aB[m];
	      c += xiR[m]*aR[m]+xiG[m]*aG[m]+xiB[m]*aB[m];
	      d += xiR[m]*xiR[m]+xiG[m]*xiG[m]+xiB[m]*xiB[m];
	    }
	    gap += epsilon*.5*d-c;
	    if (b < we2) gap += .5*b/epsilon; // happens only if epsilon>0
	    else gap += ws*sqrt(b)-.5*epsilon*w; // TV_eps
	  }
	}
      }

#pragma omp parallel for reduction(+:gap)
      for (int l=0;l<Lo;l++) {
	int i=(1+l*2)*sx+1;
	int kk=l*Ko;
	for (int k=0;k<Ko;k++,kk++,i+=2) {  
	  double aruR[4] = {uR[i], uR[i+1], uR[i+1+sx], uR[i+sx]};
	  double aruG[4] = {uG[i], uG[i+1], uG[i+1+sx], uG[i+sx]};
	  double aruB[4] = {uB[i], uB[i+1], uB[i+1+sx], uB[i+sx]};

	  // if (itcheck) 
	  {  // calcul gap avant optim
	    // TV_e(Du) - <xi,Du> + e*xi^2/2
	    double aR[4],*xiR=xioR+4*kk,b,c,d;
	    double aG[4],*xiG=xioG+4*kk;
	    double aB[4],*xiB=xioB+4*kk;
	    aR[3] = aruR[0]-aruR[3];
	    aG[3] = aruG[0]-aruG[3];
	    aB[3] = aruB[0]-aruB[3];
	    b = aR[3]*aR[3]+aG[3]*aG[3]+aB[3]*aB[3];
	    c = xiR[3]*aR[3]+xiG[3]*aG[3]+xiB[3]*aB[3];
	    d = xiR[3]*xiR[3]+xiG[3]*xiG[3]+xiB[3]*xiB[3];
	    for (int m=0;m<3;m++) {
	      aR[m] = aruR[m+1]-aruR[m];
	      aG[m] = aruG[m+1]-aruG[m];
	      aB[m] = aruB[m+1]-aruB[m];
	      b += aR[m]*aR[m]+aG[m]*aG[m]+aB[m]*aB[m];
	      c += xiR[m]*aR[m]+xiG[m]*aG[m]+xiB[m]*aB[m];
	      d += xiR[m]*xiR[m]+xiG[m]*xiG[m]+xiB[m]*xiB[m];
	    }
	    gap += epsilon*.5*d-c;
	    if (b < we2) gap += .5*b/epsilon; // happens only if epsilon>0
	    else gap += ws*sqrt(b)-.5*epsilon*w; // TV_eps
 	  } // puis optimisation 

	  aruR[0] += theta*udifR[i];
	  aruR[1] += theta*udifR[i+1];
	  aruR[2] += theta*udifR[i+1+sx];
	  aruR[3] += theta*udifR[i+sx];
	  aruG[0] += theta*udifG[i];
	  aruG[1] += theta*udifG[i+1];
	  aruG[2] += theta*udifG[i+1+sx];
	  aruG[3] += theta*udifG[i+sx];
	  aruB[0] += theta*udifB[i];
	  aruB[1] += theta*udifB[i+1];
	  aruB[2] += theta*udifB[i+1+sx];
	  aruB[3] += theta*udifB[i+sx];

	  optimsquare(aruR,aruG,aruB,xioR+kk*4,xioG+kk*4,xioB+kk*4,w,epsilon);

	  uR[i]=aruR[0]-theta*udifR[i];
	  uR[i+1]=aruR[1]-theta*udifR[i+1];
	  uR[i+1+sx]=aruR[2]-theta*udifR[i+1+sx];
	  uR[i+sx]=aruR[3]-theta*udifR[i+sx];
	  uG[i]=aruG[0]-theta*udifG[i];
	  uG[i+1]=aruG[1]-theta*udifG[i+1];
	  uG[i+1+sx]=aruG[2]-theta*udifG[i+1+sx];
	  uG[i+sx]=aruG[3]-theta*udifG[i+sx];
	  uB[i]=aruB[0]-theta*udifB[i];
	  uB[i+1]=aruB[1]-theta*udifB[i+1];
	  uB[i+1+sx]=aruB[2]-theta*udifB[i+1+sx];
	  uB[i+sx]=aruB[3]-theta*udifB[i+sx];
	}
      }
    }

    // if (gap<0) { fprintf(stderr,"gap negatif!\n"); exit(0); }
    rmse = sqrt(gap/sxy); // gap moyen = RMSE

#ifdef VERB2
    fprintf(stdout,"%d  %f    \r",it,rmse); fflush(stdout);
#endif
  } while (it < NIT && rmse > MAXERR);
  free(xieR); free(xioR);
  free(udifR);
  /* free(xieG); free(xioG); */
  /* free(udifG); */
  /* free(xieB); free(xioB); */
  /* free(udifB); */
#ifdef VERB1
    fprintf(stdout,"%d  %f    \n",it,rmse);
#endif
}
