import tensorflow as tf

class ZeroOutTest(tf.test.TestCase):
  def testZeroOut(self):
    add_two_module = tf.load_op_library('../../../bazel-bin/tensorflow/core/user_ops/add_two.so')
    add_two = add_two_module.add_two
    with self.test_session():
      result = add_two([[5], [4], [3], [2], [1]])
      self.assertAllEqual(result.numpy(), [[7], [6], [5], [4], [3]])

if __name__ == "__main__":
  tf.test.main()