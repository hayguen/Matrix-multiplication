/*
      Final matrix with tiling
      compile ->  gcc -O3 -march=native -fopenmp matrix-tiling.c -o matrix-block
      run ->sudo perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses ./matrix-block

*/

#define _POSIX_C_SOURCE 199309L
// Required for CLOCK_MONOTONIC to measure precise timer without
// moving backward
#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define Total_SIZE 1032 * 1024
#define BLOCK_SIZE 64

void multiply(int n,int stride, const double *restrict A, const double *restrict B,
              double *restrict C) {

#pragma omp parallel for collapse(2) // from 1.2 sec to 0.607 sec
  for (int i = 0; i < n; i += BLOCK_SIZE) {
    for (int j = 0; j < n; j += BLOCK_SIZE) {
      for (int k = 0; k < n; k += BLOCK_SIZE) {

        for (int ii = i; ii < BLOCK_SIZE + i; ii++) {
          for (int kk = k; kk < BLOCK_SIZE + k; kk++) {
            for (int jj = j; jj < BLOCK_SIZE + j; jj++) {
              /* __builtin_prefetch(
              &B[(kk+4)*n + j],
              0,
              1
          );

          __builtin_prefetch(
              &A[ii*n + kk+4],
              0,
              1
          );*/

              C[ii * stride + jj] += A[ii * stride + kk] * B[kk * stride + jj];
            }
          }
        }
      }
    }
  }
}

int main() {

  struct timespec start, end;

  int n = 1024;
 int stride = 1032; // Our padding offset
  size_t size = stride * n * sizeof(double);
// 32-byte alignment is perfect for AVX 256-bit registers
  double *A = (double *)aligned_alloc(64, size); 
  double *B = (double *)aligned_alloc(64, size);
  double *C = (double *)aligned_alloc(64, size);

  for (int i = 0; i < Total_SIZE; i++) {
    C[i] = 0; // auto initialize to 0
  }
  for (int i = 0; i < Total_SIZE; i++) {
    A[i] = 2.0;
    B[i] = 3.0;
  }
  printf("\n\tMatrix multiplication with "
         "Tiling,alligned_alloc,padding-8rows and rest previous "
         "optimisation \n");

  printf("\nStarting benchmark for %d x %d matrix...\n", n, n);
  clock_gettime(CLOCK_MONOTONIC, &start);
  multiply(n,stride, A, B, C);
  clock_gettime(CLOCK_MONOTONIC, &end);

  double time_taken =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("\nFunction took %f seconds to execute.\n", time_taken);

  // Calcuating GFLOPS for i3-2nd gen machine with theoratical DP GFLOPS as 52.8
  double operations = 2.0 * (double)n * (double)n * (double)n;

  double achived_gflpos = operations / (time_taken * 1e9);
  double theoretical_gflops = 52.8;

  double efficiency = (achived_gflpos / theoretical_gflops) * 100.0;

  printf(
      "\n Achived GFLOPS: %f\n Theoretical DP GFLOPS: %f\n Efficiency %f%% \n ",
      achived_gflpos, theoretical_gflops, efficiency);
  printf("Verification: C[0] = %f\n\n", C[0]);
  free(A);
  free(B);
  free(C);
}
