#ifndef TRIVIAL_KERNEL_H_
#define TRIVIAL_KERNEL_H_

#include <unsupported/Eigen/CXX11/Tensor>


template<typename Device, typename T>
struct compute_functor {
    void operator() (Device& d);
};

#if GOOGLE_CUDA
template<typename T>
struct compute_functor<Eigen::GpuDevice, T> {
    void operator() (const Eigen::GpuDevice& d, T input);
};
#endif

#endif TRIVIAL_KERNEL_H_