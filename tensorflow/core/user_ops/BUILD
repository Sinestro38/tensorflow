load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)
tf_custom_op_library(
    name = "add_two.so",
    srcs = ["add_two.cc"],
)
tf_custom_op_library(
    # kernel_example.cc  kernel_example.cu.cc  kernel_example.h
    name = "kernel_example.so",
    srcs = ["kernel_example.h", "kernel_example.cc"],
    gpu_srcs = ["kernel_example.cu.cc", "kernel_example.h"],
)

# User ops.

load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)

package(
    default_visibility = [
        "//tensorflow/core:__pkg__",
    ],
    licenses = ["notice"],
)

# One target for all user ops.
cc_library(
    name = "user_ops_op_lib",
    srcs = glob(["*.cc"]),
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//tensorflow/core:__pkg__"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)
