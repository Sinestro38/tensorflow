# TODO
# (input) -> tensor of square nxn size
# (output) -> scalar which is max element within nxn matrix
import numpy as np
import tensorflow as tf

my_pool_module = tf.load_op_library('../../../../bazel-bin/tensorflow/core/user_ops/max_pool/max_pool.so')
max_pool = my_pool_module.my_max_pool
print(max_pool([[1, 2], [4, 3]]).numpy())