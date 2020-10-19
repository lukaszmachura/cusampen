#ifndef CUSAMPEN_CUDA_MODULE_H
#define CUSAMPEN_CUDA_MODULE_H

#include <cuda.h>
#include <string>

#include <cusampen/cuda/errchk.h>
#include <cusampen/cuda/function.h>

namespace cusampen { namespace cuda {

class module{
  private:
    CUmodule _module;

  public:
    inline module(): _module(nullptr) {}

    inline module(const std::string &ptx){
      cuErrchk(cuModuleLoadDataEx(&_module, ptx.c_str(), 0, 0, 0));
    }

    inline module(module &that): _module(that._module){
      that._module = nullptr;
    }

    inline module(module &&that): _module(that._module){
      that._module = nullptr;
    }

    inline function get_function(std::string name) const{
      CUfunction fun;
      cuErrchk(cuModuleGetFunction(&fun, _module, name.c_str()));
      return function(fun);
    }

    inline operator CUmodule() const { return _module; }

    inline module& operator=(module &that) {
      if (this != &that) {
        _module = that._module;
        that._module = nullptr;
      }
      return *this;
    }

    inline module& operator=(module &&that) {
      unload();
      this->_module = that._module;
      that._module = nullptr;
      return *this;
    }

    inline void unload(){
      if (_module != nullptr){
        cuErrchk(cuModuleUnload(_module));
        _module = nullptr;
      }
    }

    inline ~module(){
      unload();
    }
};

} //namespace cuda
} //namespace cusampen

#endif //CUSAMPEN_CUDA_MODULE_H
