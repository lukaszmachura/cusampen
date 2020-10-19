#ifndef CUSAMPEN_CUDA_CUDA_H
#define CUSAMPEN_CUDA_CUDA_H

#include <cuda.h>

#include <cusampen/cuda/context.h>
#include <cusampen/cuda/device.h>
#include <cusampen/cuda/dim.h>
#include <cusampen/cuda/errchk.h>
#include <cusampen/cuda/function.h>
#include <cusampen/cuda/memory.h>
#include <cusampen/cuda/module.h>
#include <cusampen/cuda/nvrtc.h>
#include <cusampen/cuda/stream.h>

namespace cusampen { namespace cuda {

inline void init(){
  cuErrchk(cuInit(0));
}

inline int get_max_blocks_per_multiprocessor(int version){
  if ((version < 5000) || (version == 7050)){
    return 16;
  } else {
    return 32;
  }
}

} //namespace cuda
} //namespace cusampen

#endif //CUSAMPEN_CUDA_CUDA_H
