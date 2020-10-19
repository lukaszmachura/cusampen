#ifndef CUSAMPEN_CUDA_DIM_H
#define CUSAMPEN_CUDA_DIM_H

namespace cusampen { namespace cuda {

class dim3 {
  public:
    int x;
    int y;
    int z;

    inline constexpr dim3(int x=1, int y=1, int z=1): x(x), y(y), z(z){}
};

} //namespace cuda
} //namespace cusampen

#endif //CUSAMPEN_CUDA_DIM_H
