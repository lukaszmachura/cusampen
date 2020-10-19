#ifndef CUSAMPEN_CUDA_MODEL_H
#define CUSAMPEN_CUDA_MODEL_H

#include <cuda.h>

#include <cusampen/cuda/errchk.h>
#include <cusampen/cuda/stream.h>

namespace cusampen { namespace cuda {
template<typename T>
class device_ptr;

template<typename T>
class host_ptr;

template<typename T>
class host_ptr{
  private:
    T *ptr;

  public:
    inline host_ptr(): ptr(nullptr) {}

    inline host_ptr(::size_t size, int flags){
      cuErrchk(cuMemHostAlloc((void **) &ptr, size, flags));
    }

    inline host_ptr(::size_t size) {
      cuErrchk(cuMemAllocHost((void **) &ptr, size));
    }

    inline ~host_ptr(){
      cuErrchk(cuMemFreeHost((void *)ptr));
    }


    inline operator T*() const { return ptr; }
    inline operator void*() const { return static_cast<void *>(ptr); }
    inline operator host_ptr*() const { return &ptr; }

    inline T& operator[](::size_t idx) { return ptr[idx]; }
    inline const T& operator[](::size_t idx) const{ return ptr[idx]; }

    inline void to_device(device_ptr<T> &dst, ::size_t size) const{
      cuErrchk(cuMemcpyHtoD(dst, (void *) ptr, size));
    }

    inline void to_device(device_ptr<T> &dst, ::size_t size, stream &st) const {
      cuErrchk(cuMemcpyHtoDAsync(dst,(void *) ptr, size, st));
    }
};

template<typename T>
class device_ptr {
  private:
    CUdeviceptr ptr;
    bool view;

  public:
    inline device_ptr(): ptr(0), view(true) {}

    inline device_ptr(::size_t size): view(false){
      cuErrchk(cuMemAlloc(&ptr, size));
    }

    inline device_ptr(device_ptr &pointer): view(true), ptr(pointer.ptr){}

    inline device_ptr(CUdeviceptr ptr): view(true), ptr(ptr) {}

    inline ~device_ptr(){
      if (!view){
        cuErrchk(cuMemFree(ptr));
      }
    }

    inline void * operator &() { return &ptr; }
    inline operator CUdeviceptr() const { return ptr;}

    inline device_ptr<T> operator[](::size_t idx) {
      return device_ptr<T>(((CUdeviceptr) &((T *) ptr)[idx]));
    }

    inline const device_ptr<T> operator[](::size_t idx) const {
     return device_ptr<T>(((CUdeviceptr) &((T *) ptr)[idx]));
    } 

    inline void to_host(host_ptr<T> &dst, ::size_t size) const{
      cuErrchk(cuMemcpyDtoH((void *) dst, ptr, size));
    }

    inline void to_host(host_ptr<T> &dst, ::size_t size, stream &st) const{
      cuErrchk(cuMemcpyDtoHAsync((void *) dst, ptr, size, st));
    }
};

template<typename T>
inline
CUdeviceptr get_device_ptr(CUdeviceptr &ptr, int idx){
  return ((CUdeviceptr) &((T *) ptr)[idx]);
} 


} //namespace cuda
} //namespace cusampen

#endif //CUSAMPEN_CUDA_MODEL_H
