#ifndef CUSAMPEN_CUDA_DEVICE_H
#define CUSAMPEN_CUDA_DEVICE_H

#include <cuda.h>

#include <cusampen/cuda/errchk.h>

namespace cusampen { namespace cuda {

class device{
  private:
    CUdevice dev;
  public:
    inline device(int ordinal){
      cuErrchk(cuDeviceGet(&dev, ordinal));
    }

    inline operator CUdevice() const { return dev; }

    int get_attribute(CUdevice_attribute attrib) const{
      int attr;
      cuErrchk(cuDeviceGetAttribute(&attr, attrib, dev));
      return attr;
    }

    int get_version(){
      int major = get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
      int minor = get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
      return major * 1000 + minor * 10;
    }
};


} //namespace cuda
} //namespace cusampen

#endif //CUSAMPEN_CUDA_DEVICE_H
