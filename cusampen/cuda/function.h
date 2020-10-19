#ifndef CUSAMPEN_CUDA_FUNCTION_H
#define CUSAMPEN_CUDA_FUNCTION_H

#include <cuda.h>

#include <cusampen/cuda/dim.h>
#include <cusampen/cuda/errchk.h>
#include <cusampen/cuda/stream.h>

namespace cusampen { namespace cuda {

class function{
  private:
    CUfunction fun;
  public:
    inline function(): fun(nullptr){}
    inline function(CUfunction &fun): fun(fun){}

    inline void 
    launch(dim3 &blocks, dim3 &threads, unsigned int smem, stream &st, void **params) const {
      cuErrchk(cuLaunchKernel(fun,           //kernel
                              blocks.x,      //width of grid in blocks
                              blocks.y,      //height of grid in blocks
                              blocks.z,      //depth of grid in blocks
                              threads.x,     //width of each thread block
                              threads.y,     //height of each thread block
                              threads.z,     //depth of each thread block
                              smem,          //dynamic shared0memory size per thread block in bytes
                              st,            //stream identifier
                              params,        //array if pointers to kernel parameters
                              nullptr));     //extra options
    }

    inline operator CUfunction() const { return fun; }
    inline int get_attribute(CUfunction_attribute attrib) const{
      int attr;
      cuErrchk(cuFuncGetAttribute(&attr, attrib, fun));
      return attr;
    }
};

} //namespace cuda
} //namespace cusampen

#endif //CUSAMPEN_CUDA_FUNCTION_H
