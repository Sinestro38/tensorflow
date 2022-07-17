

#include "add_one_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("PavanAddOne")
  .Attr("T: numbertype")
  .Input("input: T")
  .Output("output: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
      c->set_output(0, c->input(0));
      return Status::OK();
  });

// cpu kernel computation function definition
template <typename T>
void compute_cpu_kernel(const int size, const T* in, T* out) {
  for (int i = 0; i < size; i++) {
    out[i] = in[i] + 1;
  }
}

template <typename T>
class MyCpuAddOneOp : public OpKernel {
  public:
    explicit MyCpuAddOneOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) {
        // fetch inputs
        const Tensor& input_tensor = context->input(0);
        const T* input_arr = input_tensor.flat<T>().data();

        // allocate outputs
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input_tensor.shape(), &output_tensor));
        T* output_arr = output_tensor->flat<T>().data();

        // do the computation
        compute_cpu_kernel<T>(static_cast<int>(input_tensor.NumElements()), input_arr, output_arr);
    }    

};

template <typename T>
class MyGpuAddOneOp : public OpKernel {
  public:
    explicit MyGpuAddOneOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) {
        // fetch inputs
        const Tensor& input_tensor = context->input(0);
        const T* input_arr = input_tensor.flat<T>().data();

        // allocate outputs
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input_tensor.shape(), &output_tensor));
        T* output_arr = output_tensor->flat<T>().data();

        // do the computation
        compute_gpu_kernel<T>(static_cast<int>(input_tensor.NumElements()), input_arr, output_arr);
    }    
};


REGISTER_KERNEL_BUILDER(Name("PavanAddOne").Device(DEVICE_CPU).TypeConstraint<int>("T"), MyCpuAddOneOp<int>);
REGISTER_KERNEL_BUILDER(Name("PavanAddOne").Device(DEVICE_CPU).TypeConstraint<float>("T"), MyCpuAddOneOp<float>);
REGISTER_KERNEL_BUILDER(Name("PavanAddOne").Device(DEVICE_GPU).TypeConstraint<int>("T"), MyGpuAddOneOp<int>);
REGISTER_KERNEL_BUILDER(Name("PavanAddOne").Device(DEVICE_GPU).TypeConstraint<float>("T"), MyGpuAddOneOp<float>);