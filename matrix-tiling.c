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

#define BLOCK_SIZE 64
typedef double FLOAT;

void multiply(int N, int M, int MSTRIDE, int L, int LSTRIDE,
              const FLOAT *restrict A, const FLOAT *restrict B,
              FLOAT *restrict C) {
#pragma omp parallel for collapse(2) // from 1.2 sec to 0.607 sec
  for (int i = 0; i < N; i += BLOCK_SIZE) {
    for (int j = 0; j < L; j += BLOCK_SIZE) {
      const int ii_limit = (BLOCK_SIZE + i) < N ? (BLOCK_SIZE + i) : N;
      const int jj_limit = (BLOCK_SIZE + j) < L ? (BLOCK_SIZE + j) : L;
      for (int k = 0; k < M; k += BLOCK_SIZE) {

        const int kk_limit = (BLOCK_SIZE + k) < M ? (BLOCK_SIZE + k) : M;
        for (int ii = i; ii < ii_limit; ii++) {
          for (int kk = k; kk < kk_limit; kk++) {
            for (int jj = j; jj < jj_limit; jj++) {
              /*
               * __builtin_prefetch( &B[(kk+4)*n + j], 0, 1 );
               * __builtin_prefetch( &A[ii*n + kk+4], 0, 1 );
               */
              C[ii * LSTRIDE + jj] += A[ii * MSTRIDE + kk] * B[kk * LSTRIDE + jj];
            }
          }
        }
      }
    }
  }
}


static unsigned row_stride_len(unsigned N, size_t s)
{
  const unsigned cache_assoc_ways[] = { 4, 8, 10, 12, 16 };  // known cache assoc ways
  const size_t CACHE_LINE_SZ = 64;
  const size_t min_sz = N * s;
  size_t n_cache_lines = ( min_sz + CACHE_LINE_SZ - 1 ) / CACHE_LINE_SZ;
  for ( unsigned k = 0; k < (sizeof(cache_assoc_ways) / sizeof(cache_assoc_ways[0])); ++k )
    if ( !(n_cache_lines % cache_assoc_ways[k]) )
      ++n_cache_lines;
  const size_t row_stride_sz = n_cache_lines * CACHE_LINE_SZ;
  unsigned n_stride_len = (row_stride_sz + s - 1) / s;
  return n_stride_len;
}


int main(int argc, char *argv[]) {

  struct timespec start, end;
  int N = 0, M = 0, L = 0;
  int MSTRIDE = 0, LSTRIDE = 0;
  int F = 1;
  FLOAT *A, *B, *C;

  if ( 1 < argc )  N = atoi(argv[1]);
  if ( 2 < argc )  M = atoi(argv[2]);
  if ( 3 < argc )  L = atoi(argv[3]);
  if ( 4 < argc )  F = atoi(argv[4]);
  if (N <= 0)      N = 1024;
  if (M <= 0)      M = N;
  if (L <= 0)      L = M;

  MSTRIDE = F ? row_stride_len(M, sizeof(FLOAT)) : M;
  LSTRIDE = F ? row_stride_len(L, sizeof(FLOAT)) : L;

  A = aligned_alloc(BLOCK_SIZE, N * MSTRIDE * sizeof(FLOAT)); // get total size but in perfect chunck of block
  B = aligned_alloc(BLOCK_SIZE, M * LSTRIDE * sizeof(FLOAT));
  C = aligned_alloc(BLOCK_SIZE, N * LSTRIDE * sizeof(FLOAT));

  for (int i = 0; i < N * LSTRIDE; i++)
    C[i] = 0; // auto initialize to 0

  for (int row = 0; row < N; row++)
      for (int col = 0; col < M; col++)
          A[row*MSTRIDE + col] = row + col;

  for (int row = 0; row < M; row++)
      for (int col = 0; col < L; col++)
          B[row*LSTRIDE + col] = row - col;

  printf("\n\tMatrix multiplication with "
         "Tiling,alligned_alloc,padding-8rows and rest previous "
         "optimisation \n");

  printf("\nStarting benchmark for (%dx%d stride %d) x (%dx%d stride %d) matrix...\n", N, M, MSTRIDE, M, L, LSTRIDE);
  clock_gettime(CLOCK_MONOTONIC, &start);
  multiply(N, M, MSTRIDE, L, LSTRIDE, A, B, C);
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
        printf("%.1f\t", C[row*LSTRIDE + col]);
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
