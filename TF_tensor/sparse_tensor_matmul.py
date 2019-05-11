import tensorflow as tf

'''
[[1,0],
 [0,1]]
'''
st = tf.SparseTensor(values=[1, 2], indices=[[0, 0], [1, 1]], dense_shape=[4, 4])
dt = tf.ones(shape=[4,4],dtype=tf.int32)
result = tf.sparse_tensor_dense_matmul(st,dt)
sess = tf.Session()
with sess.as_default():
    print(result.eval())
