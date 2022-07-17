#include "trivial_kernel.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("TrivialKernel");


class TrivialKernelOp : public OpKernel {
  public:
    explicit TrivialKernelOp(OpKernelConstruction* context) : OpKernel(context) { };
    
    void Compute(OpKernelContext* context) {
        compute_functor<GPUDevice, int> kernel;
        kernel(context->eigen_device<GPUDevice>(), 555);
    }
};


REGISTER_KERNEL_BUILDER(Name("TrivialKernel").Device(DEVICE_CPU), TrivialKernelOp)