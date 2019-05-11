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
result = tf.nn.embedding_lookup_sparse(a, idx, idx, combiner=None)
#将idx当成类weight,weight也是稀疏的
#解释一下就是就是取idx中每一行对应的值所对应的向量求平均（在没给weight的条件下
# 如果给weight的话就是加权和）
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
r = sess.run(result)
print(r)
####输出为#####

#embedding:
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]
WARNING:tensorflow:The default value of combiner will change from "mean" to "sqrtn" after 2016/11/01.
2019-05-11 23:30:52.694134: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
[[ 4.  5.  6.  7.]#第一行的加权和 ，分别为embedding中的1,1，权重为1,1
 [ 8.  9. 10. 11.]]#第二行的加权和，分别为embedding中2,0，权重为2,0 最后除以向量数目
