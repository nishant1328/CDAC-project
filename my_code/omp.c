#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<complex.h>
#include<lapacke.h>
#include<cblas.h>
#include<omp.h>

#define PI 3.1415926535897932

double potential(double x){
    return 0.50*x*x;
}

double complex **calloc_2d_array_complex(int rows, int cols) {
    double complex **array = (double complex**)calloc(rows, sizeof(double complex*));
    
    if (array == NULL) {
        printf("Memory allocation failed for rows!\n");
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        array[i] = (double complex*)calloc(cols, sizeof(double complex));
        if (array[i] == NULL) {
            printf("Memory allocation failed for row %d!\n", i);
            exit(1);
        }
    }

    return array;
}

void free_2d_array_complex(double complex** array, int rows) {
    for (int i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
}

void copy_matrix_complex_to_complex(int ncsf, double complex **source, double complex **dest) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ncsf; i++) {
        for (int j = 0; j < ncsf; j++) {
            dest[i][j] = source[i][j];
        }
    }
}

void copy_matrix_real_to_complex(int ncsf, double **source, double complex **dest) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ncsf; i++) {
        for (int j = 0; j < ncsf; j++) {
            __real__ dest[i][j] = source[i][j];
            __imag__ dest[i][j] = 0.0;
        }
    }
}

int timeprop(int ncsf, double xmin, double xmax, double lmda, int flqchnl, int noptc, 
             int istate, double totime, int ntim, double omga, double epsln){

  double mass = 10000.0; 
  double dtim = totime/ntim;
  double dx = (xmax-xmin)/(ncsf-1);
  int tchnl = 2*flqchnl+1;
  int nfloq = ncsf*tchnl;
  int inzr = flqchnl*ncsf;
  double tau = 2.0*PI/omga;

	printf("ncsf = %d \n", ncsf);
  printf("dx = %lf \n", dx);
  printf("length = %lf \n", (xmax-xmin));
  printf("dt = %lf \n", dtim);
  printf("totime = %lf \n", totime);
  printf("tau = %lf \n", tau);
  printf("omega = %lf \n", omga);
  printf("epsln = %lf \n", epsln);
  printf("tchnl = %d \n", tchnl);
  printf("tau = %lf \n", tau);
  printf("ntim = %d \n", ntim);


 double complex *coff;  // Declare coff
    coff = (double complex*)calloc(ncsf * tchnl, sizeof(double complex));  // Allocate memory for coff
    double complex *cofi = (double complex*)calloc(nfloq, sizeof(double complex)); // Declare cofi
    double complex *cofh;  // Declare cofh
    cofh = (double complex*)calloc(ncsf * tchnl, sizeof(double complex));  // Allocate memory for cofh

 double complex **cumt1 = calloc_2d_array_complex(ncsf, ncsf);
    double complex **cumt2 = calloc_2d_array_complex(ncsf, ncsf);




  double **hmt = (double**)calloc(ncsf,sizeof(double*));
  for (int i = 0; i < ncsf; i++) {
    hmt[i] = (double*)calloc(ncsf,sizeof(double));
  }

  #pragma omp parallel for 
  for (int k = 0; k < ncsf; k++) {
    hmt[k][k] = (PI*PI)/(6.0*dx*dx);
    for (int j = 0; j < k; j++) {
      hmt[k][j] = (pow(-1.0,(k-j)))/((k-j)*(k-j)*dx*dx);
      hmt[j][k] = hmt[k][j];
    }
  }

  double *zdipole, *cpm;  
  zdipole = (double *)calloc(ncsf,sizeof(double));
  cpm = (double *)calloc(ncsf,sizeof(double));

  #pragma omp parallel for
  for (int i = 0; i < ncsf; i++) {
    double x = xmin + i*dx;
    zdipole[i] = x;
    hmt[i][i] = hmt[i][i] + potential(x);
    if (abs(x) > (xmax-30.0)) {
      cpm[i] = (abs(x)-(xmax-30.0))*(abs(x)-(xmax-30.0));
    }
  }

  double *h; 
  h = (double *)calloc(ncsf*ncsf,sizeof(double));

  #pragma omp parallel for collapse(2)
  for (int i = 0; i < ncsf; i++) {
    for (int j = 0; j < ncsf; j++) {
      h[i*ncsf+j] = hmt[i][j];
    }
  }

  double *eig = (double *)calloc(ncsf,sizeof(double));  
  int info;
  info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', ncsf, h, ncsf, eig);
  if (info != 0) {
      printf("Error in LAPACKE_dsyev: %d\n",info);
      return -1;
  }

  #pragma omp parallel for collapse(2)
  for (int i = 0; i < ncsf; i++) {
    for (int j = 0; j < ncsf; j++) {
      hmt[i][j] = h[i*ncsf+j];
    }
  }

//  double complex *cofi = (double complex*)calloc(nfloq, sizeof(double complex));

  #pragma omp parallel for
  for (int i = 0; i < ncsf; i++) {
    cofi[inzr + i] = hmt[i][istate];
  }

  //double complex **cumt1 = calloc_2d_array_complex(ncsf, ncsf);
 // double complex **cumt2 = calloc_2d_array_complex(ncsf, ncsf);
  double complex com1 = -I * dtim;  
  
  #pragma omp parallel for 
  for (int i = 0; i < ncsf; i++) {
    double complex exevl = cexp(com1 * eig[i]);
    for (int j = 0; j < ncsf; j++) {
      cumt1[i][j] = exevl * hmt[j][i];
    }
  }

  free(eig);

  copy_matrix_real_to_complex(ncsf, hmt, cumt2); 

  double complex **exhm = calloc_2d_array_complex(ncsf, ncsf);

  double complex *h1, *h2, *h3;
  h1 = (double complex *)calloc(ncsf*ncsf,sizeof(double complex));
  h2 = (double complex *)calloc(ncsf*ncsf,sizeof(double complex));
  h3 = (double complex *)calloc(ncsf*ncsf,sizeof(double complex));

  #pragma omp parallel for collapse(2)
  for (int i = 0; i < ncsf; i++) {
    for (int j = 0; j < ncsf; j++) {
      h1[j*ncsf+i] = cumt1[i][j];        
      h2[j*ncsf+i] = cumt2[i][j];        
    }	  
  }

  double complex alpha = 1.0 + 0.0*I;
  double complex beta = 0.0 + 0.0*I;

  cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              ncsf, ncsf, ncsf, &alpha, h2, ncsf, 
              h1, ncsf, &beta, h3, ncsf);

  #pragma omp parallel for collapse(2)
  for (int i = 0; i < ncsf; i++) {
    for (int j = 0; j < ncsf; j++) {
      exhm[i][j] = h3[j*ncsf+i];
    }
  }

  #pragma omp parallel for
  for (int itim = 0; itim < ntim; itim++) {
    double time = itim * dtim;
    double epls = epsln * sin((PI * time) / totime) * sin((PI * time) / totime);
    int n1 = 0;
    for (int i = 0; i < tchnl; i++) {
      int findx = -(tchnl + 2)/2 + i;
      for (int j = 0; j < ncsf; j++) {
        coff[n1] = cofi[inzr + j] * cexp(findx * I * omga * time);
        n1++;
      }
    }
  }

  free(coff);
  free(cofi);
  free(cofh);
  free(h1);
  free(h2);
  free(h3);
  free(zdipole);
  free_2d_array_complex(cumt1, tchnl);

  return 0;
}

int main(){
  int ntime, ncstep, istate, flqchnl;
  int ncsf, noptc;
  double xmin, xmax, epsln, omga;
  double tau, totime, lmda;

  ncsf = 1000; flqchnl = 5; xmin = 0 ; xmax = 10.0;
  epsln = 0.5; omga = 0.5; lmda = 0.0;
  noptc = 5; istate = 1;
  
  tau = 2.0*PI/omga;
  totime = noptc*tau;
  ntime = (int)totime;
  
  //printf("ntime = %d, totime = %lf, tau = %lf\n",ntime, totime, tau);
  
  double start=omp_get_wtime();
  
  timeprop(ncsf, xmin, xmax, lmda, flqchnl, noptc, istate, totime, ntime, omga, epsln);
  

  double end=omp_get_wtime();


  printf("Time taken in openmp is %lf\n",end-start);

  return 0;
}

