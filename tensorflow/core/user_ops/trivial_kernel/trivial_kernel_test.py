import tensorflow as tf

op_path = r"../../../../bazel-bin/tensorflow/core/user_ops/trivial_kernel/trivial_kernel.so"
class ZeroOutTest(tf.test.TestCase):
  def testZeroOut(self):
    kernel_module = tf.load_op_library(op_path)
    kernel = kernel_module.trivial_kernel
    with self.test_session():
      kernel()
    #   print("RESULT: ", result.numpy())
    #   self.assertAllEqual(result.numpy(), [[7], [6], [5], [4], [3]])

if __name__ == "__main__":
  tf.test.main()