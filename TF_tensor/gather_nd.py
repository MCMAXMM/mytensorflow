import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
a=np.arange(0,100,1).reshape((4,5,5))
a=tf.convert_to_tensor(a,dtype=tf.float32)
b=tf.convert_to_tensor([[1,2,3],[1,0,0],[2,1,1],[1,0,0]])
c=tf.gather_nd(a,b)
print(c)
#其实就是根据索引返回值
print(a[1][2][3])
print(a[1][0][0])
