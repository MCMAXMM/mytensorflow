#tensorflow怎么手动padding
import tensorflow as tf
t = tf.constant([[[1, 2, 3], [2, 3, 4]],
                 [[1, 2, 3], [2, 3, 4]],
                 [[1, 2, 3], [2, 3, 4]]])
t2 = tf.constant([[1, 2, 3]])
a = tf.pad(t, [[1, 1], [2, 2], [1, 1]])
c = tf.pad(t2, [[1, 1], [2, 2]])
with tf.Session() as sess:
    a, c = sess.run([a, c])
    print(a)
    print(a.shape)
    print(c)
