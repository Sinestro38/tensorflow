#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("AddTwo")
    .Attr("T: {float32, int32, double}")
    .Input("to_add: T")
    .Output("added: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

template <typename T>
class AddTwoOp : public OpKernel {
  public:  
    explicit AddTwoOp(OpKernelConstruction* context) : OpKernel(context) {  }

    void Compute(OpKernelContext* context) override {
        // fetch input tensor
        const Tensor& input_tensor = context->input(0);
        auto input = input_tensor.flat<T>();

        // allocate output tensor
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, 
                       context->allocate_output(0, input_tensor.shape(), &output_tensor));
        auto output_data = output_tensor->flat<T>();

        // add two to each element
        for (int i{0}; i < input.size(); i++) {
            output_data(i) = input(i) + 2;
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("AddTwo").Device(DEVICE_CPU).TypeConstraint<float>("T"), AddTwoOp<float>);
REGISTER_KERNEL_BUILDER(Name("AddTwo").Device(DEVICE_CPU).TypeConstraint<int32>("T"), AddTwoOp<int32>);
REGISTER_KERNEL_BUILDER(Name("AddTwo").Device(DEVICE_CPU).TypeConstraint<double>("T"), AddTwoOp<double>);