import tensorflow as tf
import numpy as np

input = tf.constant(np.random.rand(3, 4))
k = 2
#返回的是输入中每一行(batch_size)前k个数及其所在行的位置
#返回两个array,第一个为每行的前k个数，第二个其所在的位置
output = tf.nn.top_k(input, k)
with tf.Session() as sess:
    print(sess.run(input))
    print(sess.run(output))

import tensorflow as tf
import numpy as np

input = tf.constant(np.random.rand(3, 4), tf.float32)
k = 2  # targets对应的索引是否在每行的最大的前k(2)个数据中
#判断所给的索引是否在topk中
output = tf.nn.in_top_k(input, [3, 3, 3], k)
with tf.Session() as sess:
    print(sess.run(input))
    print(sess.run(output))
