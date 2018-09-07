#include <stdio.h>
#include "csutil.h"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess){
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// Kernel functions
__global__ 
void reduce(int *g_idata, int *g_out) {
  __shared__ int sdata[BLOCKSIZE];
  // each thread loads one element from global to shared mem
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  sdata[threadIdx.x] = g_idata[i];

  __syncthreads();
  // do reduction in shared mem
  for (int s=1; s < blockDim.x; s *=2)
  {
    int index = 2 * s * threadIdx.x;
    if (index < blockDim.x)
    {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }

  // write result for this block to global mem
  if (threadIdx.x == 0){
      atomicAdd(g_out, sdata[0]);
  }
}

__device__
bool is_equal(float a, float b, float eps)
{
  return fabs(a - b) < eps ? true : false;
}

__global__
void findvec(float *base_vec, float *in, int *out)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  int ok = true;
  for (int i = 0; i < d_m; i++){
    if (!is_equal(in[index + i], base_vec[i], d_r)){
      ok = false;
      break;
    }
  }

  if (ok) out[index] = 1;
}

/**************************
  not used kernel functions
***************************/
__global__
void czek(int n, int *idx, int *str, int *bI, int *bD, int *tI, int *gD)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride){
      idx[i] = index;
      str[i] = stride;
      bI[i] = blockIdx.x;
      bD[i] = blockDim.x;
      tI[i] = threadIdx.x;
      gD[i] = gridDim.x;
  }
}


