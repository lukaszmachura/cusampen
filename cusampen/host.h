#ifndef CUSAMPEN_HOST_H
#define CUSAMPEN_HOST_H

#include <nvrtc.h>
#include <cuda.h>
#include <stdio.h> //for printf
#include <string>
#include <immintrin.h>

#include <cusampen/cuda/cuda.h>

namespace cusampen{

const std::string get_kernel_source(int m, float eps, int length, int apen){
  // count self match
  if (apen){
    apen = 0;
  } else { 
    apen = -1;
  }
  std::string source = "                                                    \n\
  #define M " + std::to_string(m) + "                                       \n\
  #define EPS " + std::to_string(eps) + "f                                  \n\
  #define LENGTH " + std::to_string(length) + "                             \n\
  #define APEN   " + std::to_string(apen) + "                               \n\
                                                                            \n\
  __device__ __forceinline__                                                \n\
  int is_equal(float a, float b, float eps)                                 \n\
  {                                                                         \n\
    return fabsf(a - b) <= eps ? 1 : 0;                                     \n\
  }                                                                         \n\
                                                                            \n\
  extern \"C\" __global__ void sampen(float *x,                             \n\
                                      int *mcounts,                         \n\
                                      int *mpocounts){                      \n\
    int index = blockIdx.x * blockDim.x + threadIdx.x;                      \n\
                                                                            \n\
    // we can compare for both M and M + 1 sizes                            \n\
    // to use less load/store operations                                    \n\
    if (index <= LENGTH - M){                                               \n\
      float base[M + 1];                                                    \n\
      float values[M + 1];                                                  \n\
      int mcounter = APEN;                                                  \n\
      int mpocounter = APEN;                                                \n\
      int state;                                                            \n\
                                                                            \n\
      // fetch M and M + 1 base vectors from memory at index                \n\
      #pragma unroll (M + 1)                                                \n\
      for (int i = 0; i < M + 1; ++i){                                      \n\
        base[i] = x[i + index];                                             \n\
        values[i] = x[i];                                                   \n\
      }                                                                     \n\
                                                                            \n\
      for (int i = 0; i < LENGTH - M; ++i){                                 \n\
        state = 1;                                                          \n\
        #pragma unroll (M)                                                  \n\
        for (int j = 0; j < M; ++j){                                        \n\
          state *= is_equal(base[j], values[j], EPS);                       \n\
          values[j] = values[j + 1];                                        \n\
        }                                                                   \n\
        mcounter += state;                                                  \n\
        state *= is_equal(base[M], values[M], EPS);                         \n\
        values[M] = x[M + i + 1];                                           \n\
        mpocounter += state;                                                \n\
      }                                                                     \n\
                                                                            \n\
      state = 1;                                                            \n\
      #pragma unroll (M)                                                    \n\
      for (int j = 0; j < M; ++j){                                          \n\
        state *= is_equal(base[j], values[j], EPS);                         \n\
      }                                                                     \n\
      mcounter += state;                                                    \n\
      mcounts[index] = mcounter;                                            \n\
      mpocounts[index] = mpocounter;                                        \n\
    }                                                                       \n\
  }                                                                         \n\
  ";
  //printf("%s\n", source.c_str());
  return source;
}


const std::string compile(const std::string &source){
  cuda::program prog(source);
  const char *options[] = {"-use_fast_math ", "-restrict", "-I.", "-arch=compute_50", "-std=c++14"};
  //const char *options[] = {"-use_fast_math ", "-restrict", "-I.", "-arch=compute_35", "-std=c++11"};
  //const char *options[] = {"-use_fast_math ", "-restrict", "-I.", "-arch=compute_75", "-std=c++14"};
  return prog.compile(options, 5);
}
  
int sum(int *x, int n){
  int s = 0;
  #pragma omp parallel for simd reduction(+:s)
  for (int i = 0; i < n; ++i){
    s += x[i];
  }
  return s;
}

} 
#endif
