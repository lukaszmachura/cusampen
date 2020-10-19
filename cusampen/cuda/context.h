#ifndef CUSAMPEN_CUDA_CONTEXT_H
#define CUSAMPEN_CUDA_CONTEXT_H

#include <cuda.h>

#include <cusampen/cuda/device.h>
#include <cusampen/cuda/errchk.h>

namespace cusampen { namespace cuda {

class device;

class context {
  private:
    CUcontext ctx;
  public:
    inline context(int flags, const device &dev){
      cuErrchk(cuCtxCreate(&ctx, flags, dev));
    }

    inline void set_current() const {
      cuErrchk(cuCtxSetCurrent(ctx));
    }

    inline operator CUcontext() const { return ctx; }

    //inline void synchronize() { cuErrchk(cuCtxSynchronize()); }

    inline void destroy(){
      cuErrchk(cuCtxDestroy(ctx));
    }

    inline ~context(){
      destroy();
    }
};

class primary_context {
  private:
    CUcontext ctx;
    CUdevice dev;

  public:
    inline primary_context(const device &dev): dev(dev){
      cuErrchk(cuDevicePrimaryCtxRetain(&ctx, dev));
    }

    inline void set_current() { cuErrchk(cuCtxSetCurrent(ctx)); }

    inline void synchronize() { cuErrchk(cuCtxSynchronize()); }
    
    inline void destroy(){
      cuErrchk(cuDevicePrimaryCtxRelease(dev));
    }

    inline ~primary_context(){
      destroy();
    }
};


} //namespace cuda
} //namespace cusampen

#endif //CPC_CUDA_CONTEXT_H
