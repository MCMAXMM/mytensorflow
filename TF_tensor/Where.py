import tensorflow as tf
tf.enable_eager_execution()
import numpy  as np
#1.如果只有一个输入(condiction)的话就返还这个条件为True的坐标
b=tf.where([[True,False],[True,False]])
c=tf.constant([[1,2],[3,4]])

#2.有三个输入，根据第一个输入condition，将第二个输入中的元素替换掉
a=np.array([[1,0,0],[0,1,1]])
a1=np.array([[3,2,3],[4,5,6]])
d=tf.equal(a,1)
print(d)
#tf.Tensor(
#[[ True False False]
# [False  True  True]], shape=(2, 3), dtype=bool)
e=tf.where(d,a1,a)
print(e)
#tf.Tensor(
#[[3 0 0]
 #[0 5 6]], shape=(2, 3), dtype=int32)
