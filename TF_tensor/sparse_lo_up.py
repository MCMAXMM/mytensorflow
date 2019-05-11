import tensorflow as tf
import numpy as np
import scipy.sparse as sp
#tf.enable_eager_execution()
b=np.random.randint(0,2,(12,12))
a = np.arange(16).reshape(4, 4)
b = np.arange(8, 16).reshape(2, 4)
c = np.arange(12, 20).reshape(2, 4)

print(a)
print(b)
print(c)

a = tf.Variable(a, dtype=tf.float32)
b = tf.Variable(b, dtype=tf.float32)
c = tf.Variable(c, dtype=tf.float32)

idx = tf.SparseTensor(indices=[[0,0], [0,2], [1,0], [1, 1]], values=[1,1,2,0], dense_shape=(2,3))
result = tf.nn.embedding_lookup_sparse(a, idx, None, combiner=None)
#解释一下就是就是取idx中每一行对应的值所对应的向量求平均（在没给weight的条件下
# 如果给weight的话就是加权和）
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
r = sess.run(result)
print(r)
