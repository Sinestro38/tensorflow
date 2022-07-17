#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "add_one_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "stdio.h"

template <typename T>
__global__ void add_one(const int size, const T* in, T* out) {
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = start_idx; i < size; i += blockDim.x * gridDim.x ) {
        out[i] = in[i] + 1;
        printf("Adding one in the GPU... \n");
    }
}

template <typename T>
void compute_gpu_kernel(const int size, const T* in, T* out) {
    // launch cuda kernel to add one foreach
    int block_count = 1024;
    int thread_per_block = 20;
    add_one<T><<<block_count, thread_per_block>>>(size, in, out);
}

template void compute_gpu_kernel<int>(const int size, const int* in, int* out);
template void compute_gpu_kernel<float>(const int size, const float* in, float* out);
#endif // GOOGLE_CUDA