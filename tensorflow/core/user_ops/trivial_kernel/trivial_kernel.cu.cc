#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "trivial_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

template<typename T>
__global__ void kernel(T input) {
    std::cout << "Hello from the GPU!!!!" << " You inputted " << input << std::endl;
}

template <typename T>
void compute_functor<GPUDevice, T>::operator()(const Eigen::GpuDevice& d, T input) {
    kernel<T><<<1, 1>>>(input);
}

// explicitly instantiate the template (generate code)
template struct compute_functor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA 