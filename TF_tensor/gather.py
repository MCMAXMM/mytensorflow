import tensorflow as tf
tf.enable_eager_execution()
a=tf.range(1,10,dtype=tf.int32)
b=tf.gather(a,[2,3,1,2,3,4])
print(b)
#输出为：tf.Tensor([3 4 2 3 4 5], shape=(6,), dtype=int32)
