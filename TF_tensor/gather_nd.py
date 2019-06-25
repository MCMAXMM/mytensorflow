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


#如果少一个维度的话输出就是向量
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
a=np.arange(0,100,1).reshape((4,5,5))
a=tf.convert_to_tensor(a,dtype=tf.float32)
b=tf.convert_to_tensor([[1,2],[1,0],[2,1],[1,0]])
c=tf.gather_nd(a,b)
print(c)
print(a[1][2])

#输出单个维度的索引，返回一个5*5的矩阵
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
a=np.arange(0,100,1).reshape((4,5,5))
a=tf.convert_to_tensor(a,dtype=tf.float32)
b=tf.convert_to_tensor([[1],[1],[2],[1]])
c=tf.gather_nd(a,b)
print(c)
print(a[1])
