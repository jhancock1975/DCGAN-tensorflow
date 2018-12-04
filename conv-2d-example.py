# example from https://github.com/tensorflow/tensorflow/issues/4306
# github user karimhasebou answer from August 11, 2018
# conv2d_transpose documentation at: 
# https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose
import tensorflow as tf

x = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
x = tf.reshape(x, [1, 4, 4, 1])
x = tf.to_float(x)

def create_filter(sub_list):
   sub_list_len = len(sub_list)
   sub_list = [[sub_list] for i in range(sub_list_len)]
   sub_list = tf.reshape(sub_list, [sub_list_len, sub_list_len,1,1])
   return tf.to_float(sub_list)

f = create_filter([1, 1, 1, 1, 1])
result = tf.nn.conv2d_transpose(value=x,filter=f,output_shape=[1,16,16,1],
  strides=[1,4,4,1])

sess = tf.Session()
print("x")
print(sess.run(x))
print("f")
print(sess.run(f))
print("result")
print(sess.run(result).reshape(16,16))
sess.close()
