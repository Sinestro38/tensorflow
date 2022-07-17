#ifndef ADD_ONE_KERNEL_H_
#define ADD_ONE_KERNEL_H_

#ifdef GOOGLE_CUDA
void compute_gpu_kernel(const int size, const int* in, int* out);
#endif
void compute_cpu_kernel(const int size, const int* in, int* out);

#endif ADD_ONE_KERNEL_H_