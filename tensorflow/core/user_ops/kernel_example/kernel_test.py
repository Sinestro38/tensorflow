import tensorflow as tf

op_path = r"../../../../bazel-bin/tensorflow/core/user_ops/kernel_example/kernel_example.so"
class ZeroOutTest(tf.test.TestCase):
  def testZeroOut(self):
    kernel_module = tf.load_op_library(op_path)
    example = kernel_module.example
    with self.test_session():
      result = example([[5], [4], [3], [2], [1]])
      print("RESULT: ", result.numpy())
    #   self.assertAllEqual(result.numpy(), [[7], [6], [5], [4], [3]])

if __name__ == "__main__":
  tf.test.main()