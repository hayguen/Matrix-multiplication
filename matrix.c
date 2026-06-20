#define _POSIX_C_SOURCE 199309L
// Required for CLOCK_MONOTONIC to measure precise timer without
// moving backward
#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef double FLOAT;


void multiply(int N, int M, int L, const FLOAT *restrict A, const FLOAT *restrict B,
              FLOAT *restrict C) {

#pragma omp parallel for // from 1.2 sec to 0.607 sec
  for (int row = 0; row < N; row++) {
    for (int k = 0; k < M; k++) {
      for (int col = 0; col < L; col++) {
        C[row * L + col] += A[row * M + k] * B[k * L + col];
      }
    }
  }
}

int main(int argc, char *argv[]) {

  struct timespec start, end;

  int N = 0, M = 0, L = 0;
  FLOAT *A, *B, *C;

  if ( 1 < argc )  N = atoi(argv[1]);
  if ( 2 < argc )  M = atoi(argv[2]);
  if ( 3 < argc )  L = atoi(argv[3]);
  if (N <= 0)      N = 1024;
  if (M <= 0)      M = N;
  if (L <= 0)      L = M;

  A = (FLOAT *)malloc( N * M * sizeof(FLOAT) ); // 64-bit or 8 byte float
  B = (FLOAT *)malloc( M * L * sizeof(FLOAT) );
  C = (FLOAT *)calloc( N * L, sizeof(FLOAT)); // auto initialize to 0

  for (int row = 0; row < N; row++)
    for (int col = 0; col < M; col++)
      A[row*M + col] = row + col;

  for (int row = 0; row < M; row++)
    for (int col = 0; col < L; col++)
      B[row*L + col] = row - col;

  printf("\n\tNaive Matrix multiplication with max compiler optimization,Loop "
         "re-order and distributed across multiple core\n");

  printf("\nStarting benchmark for (%dx%d) x (%dx%d) matrix...\n", N, M, M, L);
  clock_gettime(CLOCK_MONOTONIC, &start);
  multiply(N, M, L, A, B, C);
  clock_gettime(CLOCK_MONOTONIC, &end);

  double time_taken =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("\nFunction took %f seconds to execute.\n", time_taken);

  // Calcuating GFLOPS for i3-2nd gen machine with theoratical DP GFLOPS as 52.8
  double operations = 2.0 * (double)N * (double)M * (double)L;

  double achived_gflpos = operations / (time_taken * 1e9);
  double theoretical_gflops = 52.8;

  double efficiency = (achived_gflpos / theoretical_gflops) * 100.0;

  if ( N <= 4 && M <= 5 )
  {
    for (int row = 0; row < N; row++) {
      printf("\n");
      for (int col = 0; col < L; col++)
        printf("%.1f\t", C[row*L + col]);
    }
    printf("\n");
  }

  printf(
      "\n Achieved GFLOPS: %f\n Theoretical DP GFLOPS: %f\n Efficiency %f%% \n ",
      achived_gflpos, theoretical_gflops, efficiency);
  printf("Verification: C[0] = %f\n\n", C[0]);
  free(A);
  free(B);
  free(C);
}
