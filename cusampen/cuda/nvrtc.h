#ifndef CUSAMPEN_CUDA_NVRTC_H
#define CUSAMPEN_CUDA_NVRTC_H

#include <cuda.h>
#include <nvrtc.h>
#include <string>

#include <cusampen/cuda/errchk.h>

namespace cusampen { namespace cuda {

class program {
  private:
    nvrtcProgram _program;
  public:
    inline program(const std::string &src, const char *name=nullptr, 
                   int num_headers=0, const char **headers = nullptr, 
                   const char **include_names=nullptr){
     nvrtcErrchk(nvrtcCreateProgram(&_program,
                                    src.c_str(), //buffer
                                    name,   //name
                                    num_headers,      //numHeaders
                                    headers,   //headers
                                    include_names)); //include names
    }

    inline std::string compile(const char **options, int num_options){
      nvrtcResult result = nvrtcCompileProgram(_program, num_options, options);
      
      //printf("log: %s\n", log);
      if (result != NVRTC_SUCCESS){
        size_t log_size;
        nvrtcErrchk(nvrtcGetProgramLogSize(_program, &log_size));
        char *log = new char[log_size];
        nvrtcErrchk(nvrtcGetProgramLog(_program, log));
        fprintf(stderr, "Error while compiling program:\n%s\n", log); 
        delete[] log;
        exit(EXIT_FAILURE);
      }

      size_t ptx_size;
      nvrtcErrchk(nvrtcGetPTXSize(_program, &ptx_size));
      char *ptx = new char[ptx_size];
      nvrtcErrchk(nvrtcGetPTX(_program, ptx));
      return std::string(ptx);
    }

    inline operator nvrtcProgram() const { return _program; };
    
    inline void destroy(){
      nvrtcErrchk(nvrtcDestroyProgram(&_program));
    }

    inline ~program(){
      destroy();
    }
};

} //namespace cuda
} //namespace cusampen

#endif //CUSAMPEN_CUDA_NVRTC_H
