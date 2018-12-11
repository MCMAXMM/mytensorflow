import tensorflow as tf
#tf.tile(input,multiples,name=None)
#创建一个新的tensor，这个新的tensor中有多（multiples）个旧的tensor(input)
import tensorflow as tf
tf.enable_eager_execution()
#input: A Tensor. 1-D or higher.
#multiples: A Tensor. Must be one of the following types: int32, int64. 1-D. Length must be the same as the number of dimensions in input
#name: A name for the operation (optional).
input=tf.constant([[1,2,3,4],[1,2,3,4]],dtype=tf.float32)
a=tf.tile(input,[3,2])#[3,2]指的是在第一个维度上扩充三倍，第二个维度上扩充两倍
#multiples必须带[]
print(a)
输出为：
tf.Tensor(
[[1. 2. 3. 4. 1. 2. 3. 4.]
 [1. 2. 3. 4. 1. 2. 3. 4.]
 [1. 2. 3. 4. 1. 2. 3. 4.]
 [1. 2. 3. 4. 1. 2. 3. 4.]
 [1. 2. 3. 4. 1. 2. 3. 4.]
 [1. 2. 3. 4. 1. 2. 3. 4.]], shape=(6, 8), dtype=float32)
