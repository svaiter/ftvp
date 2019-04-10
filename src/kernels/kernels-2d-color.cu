__device__ inline double sq(double x) { return x*x;}

__device__ double optimsquare_eps_2d_descent(double u[12], double xi[12], int sc, double epsilon, double w, int steps) {
  double no, nxi[12], r;
  r = 1./(1+epsilon/4.);

  for (int c = 0; c < 3; c++) {
    u[c*4 + 0] -= xi[c*4 + 0]-xi[c*4 + 3];
    for (int i=1;i<=3;i++) {
      u[c*4 + i] -= xi[c*4 + i]-xi[c*4 + i-1];
    }
  }

  for (int it=0; it<steps; it++) {
    no = 0;
    for (int c = 0; c < 3; c++) {
      nxi[c*4 + 0]=r*(.5*xi[c*4 + 0]+.25*(xi[c*4 + 3]+xi[c*4 + 1]+ u[c*4 + 1]-u[c*4 + 0]));
      nxi[c*4 + 1]=r*(.5*xi[c*4 + 1]+.25*(xi[c*4 + 0]+xi[c*4 + 2]+ u[c*4 + 2]-u[c*4 + 1]));
      nxi[c*4 + 2]=r*(.5*xi[c*4 + 2]+.25*(xi[c*4 + 1]+xi[c*4 + 3]+ u[c*4 + 3]-u[c*4 + 2]));
      nxi[c*4 + 3]=r*(.5*xi[c*4 + 3]+.25*(xi[c*4 + 2]+xi[c*4 + 0]+ u[c*4 + 0]-u[c*4 + 3]));
      // no += sq(nxi[c*4 +0]) + sq(nxi[c*4 +1]) + sq(nxi[c*4 +2]) + sq(nxi[c*4 +3]);
    }
    for (int i = 0; i < 4; i++) {
      no += sq(nxi[i]) + sq(nxi[4 + i]) + sq(nxi[8 + i]);
    }
    if (no > w) {
      no = sqrt(w/no);
    } else {
      no = 1;
    }
    for (int c = 0; c < 3; c++) {
      xi[c*4 + 0]= nxi[c*4 + 0]*no;
      xi[c*4 + 1]= nxi[c*4 + 1]*no;
      xi[c*4 + 2]= nxi[c*4 + 2]*no;
      xi[c*4 + 3]= nxi[c*4 + 3]*no;
    }
  }

  for (int c = 0; c < 3; c++) {
    u[c*4 + 0] += xi[c*4 + 0]-xi[c*4 + 3];
    for (int i=1;i<=3;i++) {
      u[c*4 + i] += xi[c*4 + i]-xi[c*4 + i-1];
    }
  }

  return 0.0;
}


__global__ void opt_eps_split(int sx, int sc, double *xi, double *u, double epsilon, double w, double ws, double we2, int L, int K, int steps, int is_even_step) {
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

    double aru[12];
    for (int c = 0; c < 3; c++) {
      aru[c*4 + 0]=u[c + sc*i];
      aru[c*4 + 1]=u[c + sc*(i+1)];
      aru[c*4 + 2]=u[c + sc*(i+1+sx)];
      aru[c*4 + 3]=u[c + sc*(i+sx)];
    }

    double axi[12];
    for (int c = 0; c < 3; c++) {
      axi[c*4 + 0] = xi[c + sc*kk*4];
      axi[c*4 + 1] = xi[c + sc*(kk*4 + 1)];
      axi[c*4 + 2] = xi[c + sc*(kk*4 + 2)];
      axi[c*4 + 3] = xi[c + sc*(kk*4 + 3)];
    }

    optimsquare_eps_2d_descent(aru, axi, sc, epsilon, w, steps);

    for (int c = 0; c < 3; c++) {
      u[c + sc*i]=aru[c*4 + 0];
      u[c + sc*(i+1)]=aru[c*4 + 1];
      u[c + sc*(i+1+sx)]=aru[c*4 + 2];
      u[c + sc*(i+sx)]=aru[c*4 + 3];
    }

    for (int c = 0; c < 3; c++) {
      xi[c + sc*kk*4] = axi[c*4 + 0];
      xi[c + sc*(kk*4 + 1)] = axi[c*4 + 1];
      xi[c + sc*(kk*4 + 2)] = axi[c*4 + 2];
      xi[c + sc*(kk*4 + 3)] = axi[c*4 + 3];
    }
  }
}

__global__ void over_relax_eps_2d(int sx, int sc, double *xio, double *xiobar, double *u, double theta, int Lo, int Ko) {
  int l = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if ((l < Lo) && (k < Ko)) {
    int j = 1 + 2*l;
    int i = j * sx + 1 + 2*k;
    int kk = Ko * l + k;
    double dx[12];
    int m;
    int kx = kk * 4;

    for (int c = 0; c < 3; c++) {
      for (m=0; m < 4; m++)
        dx[c*4 + m] = theta * (xiobar[c + sc*(kx+m)] - xio[c + sc*(kx+m)]);
      u[c + sc*i] += dx[c*4 + 0]-dx[c*4 + 3];
      u[c + sc*(i+1)] += dx[c*4 + 1]-dx[c*4 + 0];
      u[c + sc*(i+1+sx)] += dx[c*4 + 2]-dx[c*4 + 1];
      u[c + sc*(i+sx)] += dx[c*4 + 3]-dx[c*4 + 2];
      for (m=0; m<4; m++)
        xio[c + sc*(kx+m)] = xiobar[c + sc*(kx+m)] + dx[c*4 + m];
    }
  }
}

__global__ void gap_arr_eps_2d(int sx, int sc, double* gl, double *xie, double *u, double epsilon, double w, double ws, double we2, int Le, int Ke, int is_even_step) {
  int l = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if ((l < Le) && (k < Ke)) {
    int i;
    if (is_even_step) {
      i = 2 * l * sx + 2 * k;
    } else {
      i = (2 * l + 1) * sx + 1 + 2 * k;
    }
    int kk = l*Ke + k;
    double aru[12], a[12], xi[12], b = 0, cc = 0, d = 0, gap = 0;

    for (int c = 0; c < 3; c++) {
      aru[c*4 + 0]=u[c + sc*i];
      aru[c*4 + 1]=u[c + sc*(i+1)];
      aru[c*4 + 2]=u[c + sc*(i+1+sx)];
      aru[c*4 + 3]=u[c + sc*(i+sx)];
      xi[c*4 + 0] = xie[c + sc*kk*4];
      xi[c*4 + 1] = xie[c + sc*(kk*4 + 1)];
      xi[c*4 + 2] = xie[c + sc*(kk*4 + 2)];
      xi[c*4 + 3] = xie[c + sc*(kk*4 + 3)];
    }

    for (int c = 0; c < 3; c++) {
      // TV_e(Du) - <xi,Du> + e*xi^2/2
      a[c*4 + 3] = aru[c*4 + 0]-aru[c*4 + 3];
      b += a[c*4 + 3]*a[c*4 + 3];
      cc += xi[c*4 + 3]*a[c*4 + 3];
      d += xi[c*4 + 3]*xi[c*4 + 3];
      for (int m=0;m<3;m++) {
        a[c*4 + m] = aru[c*4 + m+1]-aru[c*4 + m];
        b += a[c*4 + m]*a[c*4 + m];
        cc += xi[c*4 + m]*a[c*4 + m];
        d += xi[c*4 + m]*xi[c*4 + m];
      }
    }

    gap += epsilon*.5*d-cc;
    if (b < we2) {
      gap += .5*b/epsilon; // here epsilon>0
    } else {
      gap += ws*sqrt(b)-.5*epsilon*w; // TV_eps
    }
    gl[l*Ke + k] = gap;
  }
}
