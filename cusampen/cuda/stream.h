#ifndef CUSAMPEN_CUDA_STREAM_H
#define CUSAMPEN_CUDA_STREAM_H

#include <cuda.h>

#include <cusampen/cuda/errchk.h>

namespace cusampen { namespace cuda {

class stream{
  private:
    CUstream _stream;
  public:
    inline stream(unsigned int flags = CU_STREAM_NON_BLOCKING){
      cuErrchk(cuStreamCreate(&_stream, flags));
    }

    inline void synchronize() const{
      cuErrchk(cuStreamSynchronize(_stream));
    }

    inline void destroy(){
      cuErrchk(cuStreamDestroy(_stream));
    }

    inline ~stream(){
      destroy();
    }

    inline operator CUstream() const { return _stream; }
};

} //namespace cuda
} //namespace cusampen

#endif //CUSAMPEN_CUDA_STREAM_H
