__device__ inline double sq(double x) { return x*x;}

__device__ double optimsquare_eps_2d_descent(double u[4], double xi[4], double epsilon, double w, int steps) {
  double no, nxi[4], r;
  r = 1./(1+epsilon/4.);

  u[0] -= xi[0]-xi[3];
  for (int i=1;i<=3;i++) {
    u[i] -= xi[i]-xi[i-1];
  }

  for (int it=0; it<steps; it++) {
    if ((no = sq(nxi[0]=r*(.5*xi[0]+.25*(xi[3]+xi[1]+ u[1]-u[0])))
         + sq(nxi[1]=r*(.5*xi[1]+.25*(xi[0]+xi[2]+ u[2]-u[1])))
         + sq(nxi[2]=r*(.5*xi[2]+.25*(xi[1]+xi[3]+ u[3]-u[2])))
         + sq(nxi[3]=r*(.5*xi[3]+.25*(xi[2]+xi[0]+ u[0]-u[3])))) > w) {
      no = sqrt(w/no);
    } else {
      no = 1;
    }
    xi[0]= nxi[0]*no;
    xi[1]= nxi[1]*no;
    xi[2]= nxi[2]*no;
    xi[3]= nxi[3]*no;
  }

  u[0] += xi[0]-xi[3];
  for (int i=1;i<=3;i++) {
    u[i] += xi[i]-xi[i-1];
  }

  return 0.0;
}


__device__ double optimsquare_eps_2d(double u[4], double xi[4], double epsilon, double w, int steps)
{
  double z0, z1, z2;
  double t0, t1, t2;
  double a, b;
  double dno, no, noa, nob; // Derivative of the objective
  double tm, tp;
  double tmu;

  u[0] -= xi[0]-xi[3];
  for (int i=1;i<=3;i++) {
    u[i] -= xi[i]-xi[i-1];
  }

  z0 = u[3]-u[1];
  z1 = u[2]-u[0];
  z2 = (u[0]+u[2])-(u[3]+u[1]);

  a = z0*z0 + z1*z1;
  b = z2*z2;

  // If the current no is bigger than w, we do a Newton descent
  // for solving f_{a,b} = w. Otherwise, we keep mu = 0.
  tmu = epsilon;
  no = a/((2+tmu)*(2+tmu))+b/((4+tmu)*(4+tmu));
  if (no > w) {
    for (int i=0; i < steps; i++) {
      noa = a/((2+tmu)*(2+tmu));
      nob = b/((4+tmu)*(4+tmu));
      dno = noa/(2+tmu)+nob/(4+tmu);
      tmu += .5*(noa+nob-w)/dno;
      tmu = max(epsilon, tmu);
    }
  }

  // Recover the values of xi and u after finding mu.
  t0 = .5*z0/(2.+tmu);
  t1 = .5*z1/(2.+tmu);
  t2 = .5*z2/(4.+tmu);

  tm = t0-t1;
  tp = t0+t1;

  xi[1] = tp+t2;
  xi[2] = tm-t2;
  xi[3] = t2-tp;
  xi[0] = -tm-t2;

  double s = sq(xi[0]) + sq(xi[1]) + sq(xi[2]) + sq(xi[3]);
  if (s > w) {
    double factor = sqrt(w/s);
    for (int i=0; i<4; i++) {
      xi[i] *= factor;
    }
  }

  u[0] += xi[0]-xi[3];
  for (int i=1;i<=3;i++) {
    u[i] += xi[i]-xi[i-1];
  }

  return tmu;
}

__global__ void opt_eps_split(int sx, double *xi, double *u, double epsilon, double w, double ws, double we2, int L, int K, int steps, int use_newton, int is_even_step) {
  int i,kk;
  int l = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if ((l < L) && (k < K)) {

    if (is_even_step) {
      i = l * 2 * sx + 2 * k;
      kk = l * K + k;
    } else {
      i = (1 + l * 2) * sx + 1 + 2 * k;
      kk = l * K + k;
    }

    double aru[4];
    aru[0]=u[i];
    aru[1]=u[i+1];
    aru[2]=u[i+1+sx];
    aru[3]=u[i+sx];

    double axi[4];
    axi[0] = xi[kk*4];
    axi[1] = xi[kk*4 + 1];
    axi[2] = xi[kk*4 + 2];
    axi[3] = xi[kk*4 + 3];

    if (use_newton) {
      optimsquare_eps_2d(aru, axi, epsilon, w, steps);
    } else {
      optimsquare_eps_2d_descent(aru, axi, epsilon, w, steps);
    }

    u[i]=aru[0];
    u[i+1]=aru[1];
    u[i+1+sx]=aru[2];
    u[i+sx]=aru[3];

    xi[kk*4] = axi[0];
    xi[kk*4 + 1] = axi[1];
    xi[kk*4 + 2] = axi[2];
    xi[kk*4 + 3] = axi[3];
  }
}

__global__ void over_relax_eps_2d(int sx, double *xio, double *xiobar, double *u, double theta, int Lo, int Ko, int is_even_step) {
  int l = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if ((l < Lo) && (k < Ko)) {
    int i, kk;
    if (is_even_step) {
      i = l * 2 * sx + 2 * k;
      kk = l * Ko + k;
    } else {
      i = (1 + l * 2) * sx + 1 + 2 * k;
      kk = l * Ko + k;
    }
    double dx[4];
    int m;
    int kx = kk * 4;

    for (m=0; m < 4; m++)
      dx[m] = theta * (xiobar[kx+m] - xio[kx+m]);
    u[i] += dx[0]-dx[3];
    u[i+1] += dx[1]-dx[0];
    u[i+1+sx] += dx[2]-dx[1];
    u[i+sx] += dx[3]-dx[2];
    for (m=0; m<4; m++) 
      xio[kx+m] = xiobar[kx+m] + dx[m];
  }
}

__global__ void gap_arr_eps_2d(int sx, double* gl, double *xie, double *u, double epsilon, double w, double ws, double we2, int Le, int Ke, int is_even_step) {
  int l = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if ((l < Le) && (k < Ke)) {
    int i;
    if (is_even_step) {
      i = 2 * l * sx + 2 * k;
    } else {
      i = (2 * l + 1) * sx + 1 + 2 * k;
    }
    int kx = 4*(l*Ke + k);
    double aru[4], a[4], b, c, d, gap = 0;
    double *xi = xie + kx;
    aru[0]=u[i];
    aru[1]=u[i+1];
    aru[2]=u[i+1+sx];
    aru[3]=u[i+sx];
    // TV_e(Du) - <xi,Du> + e*xi^2/2
    a[3] = aru[0]-aru[3];
    b = a[3]*a[3];
    c = xi[3]*a[3];
    d = xi[3]*xi[3];
    for (int m=0;m<3;m++) {
      a[m] = aru[m+1]-aru[m];
      b += a[m]*a[m];
      c += xi[m]*a[m];
      d += xi[m]*xi[m];
    }

    gap += epsilon*.5*d-c;
    if (b < we2) {
      gap += .5*b/epsilon; // here epsilon>0
    } else {
      gap += ws*sqrt(b)-.5*epsilon*w; // TV_eps
    }
    gl[l*Ke + k] = gap;
  }
}
