#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// OP NOTE WORKING
REGISTER_OP("MyMaxPool")
    .Input("matrix: int32")
    .Output("max: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

class MyMaxPoolOp : public OpKernel
{
  public:
    explicit MyMaxPoolOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& input_tensor = context->input(0);
        const auto input = input_tensor.flat<int32>();

        Tensor* output_tensor = nullptr;
        TensorShape output_shape({}); // output just a scalar
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
        auto output = output_tensor->flat<int32>();

        // do the computation
        int max = 0;
        for (int i{0}; i < input.size(); i++)
        {
            if (input(i) > max) {
                max = input(i);
            }
        }
        output(0) = max;
    }
};



REGISTER_KERNEL_BUILDER(Name("MyMaxPool").Device(DEVICE_CPU), MyMaxPoolOp)