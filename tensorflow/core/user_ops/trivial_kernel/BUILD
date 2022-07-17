load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    # trivial_kernel.cc  trivial_kernel.cu.cc  trivial_kernel.h
    name = "trivial_kernel.so",
    srcs = ["trivial_kernel.h", "trivial_kernel.cc"],
    gpu_srcs = ["trivial_kernel.cu.cc", "trivial_kernel.h"],
)
