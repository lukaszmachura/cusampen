#ifndef CPC_CUDA_ERRCHK_H
#define CPC_CUDA_ERRCHK_H

#include <nvrtc.h>
#include <cuda.h>
#include <stdio.h>

namespace cpc { namespace cuda {

#define nvrtcErrchk(val) {cpc::cuda::nvrtcCheck((val), #val, __FILE__, __LINE__);}
inline void nvrtcCheck(nvrtcResult result, char const *const func,
                 const char *const file, int const line) {
  if (result != NVRTC_SUCCESS) {
    fprintf(stderr, "NVRTC error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), nvrtcGetErrorString(result), func);
    exit(1);
  }
}


#define cuErrchk(val) {cpc::cuda::cuCheck((val), #val, __FILE__, __LINE__);}
inline void cuCheck(CUresult result, char const *const func,
                 const char *const file, int const line) {
  if (result != CUDA_SUCCESS) {
    const char *msg;
    cuGetErrorString(result, &msg);
    fprintf(stderr, "CU error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), msg, func);
    exit(1);
  }
}

} //namespace cuda
} //namespace cpc


#endif //CPC_CUDA_ERRCHK_H
