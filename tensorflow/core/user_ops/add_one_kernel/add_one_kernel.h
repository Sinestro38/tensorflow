#ifndef ADD_ONE_KERNEL_H_
#define ADD_ONE_KERNEL_H_

#ifdef GOOGLE_CUDA
template <typename T>
void compute_gpu_kernel(const int size, const T* in, T* out);
#endif

template<typename T>
void compute_cpu_kernel(const int size, const T* in, T* out);

#endif ADD_ONE_KERNEL_H_