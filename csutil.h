#include <stdio.h>

#ifndef BLOCKSIZE
#define BLOCKSIZE 256
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__device__ int d_m;
__device__ float d_r;

inline void gpuAssert(cudaError_t, const char *, int, bool);
__global__ void reduce(int *, int *);
__device__ bool is_equal(float, float, float);
__global__ void findvec(float *, float *, int *);
__global__ void czek(int, int *, int *, int *, int *, int *, int *);

