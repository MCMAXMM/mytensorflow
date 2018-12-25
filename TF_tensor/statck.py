import tensorflow as tf
tf.enable_eager_execution()
x = tf.constant([[1, 4],[2,8]])
y = tf.constant([[2, 5],[4,10]])
z = tf.constant([[3, 6],[6,12]])
a=tf.stack([x, y, z],axis=0) #[3,2,2] 注意3的位置与axis一致
#a中axis=0，最简单直接将想x,y,z按照顺序堆叠起来，
b=tf.stack([x, y, z], axis=1) #[2,3,2]
#b中先将x[0],y[0],z[0]堆叠到一块，再将x[1],y[1],z[1]堆叠起来
c=tf.stack([x,y,z],axis=2)#[2,2,3]
#c中先将x[0][0],y[0][0],z[0][0],堆成一列，再将想x[0][1],y[0][1],z[0][1]堆成一列
print(a)
print(b)
print(c)
#tf.get_variable,只需初始化名字和形状，他会帮你随机初始化，当然你可以自己制定初始化器

#请注意，当初始化器是 tf.Tensor 时，您不应指定变量的形状，因为将使用初始化器张量的形状
a=tf.get_variable("machao",(1,2,3))
b=tf.get_variable("adsf",(2,3,4),initializer=tf.uniform_unit_scaling_initializer)
print(a)
print(a.shape)
print(b)

import tensorflow as tf
tf.enable_eager_execution()
#tf.stack其实就是堆叠，也就是将两个tensor堆积起来，axis控制在那个轴堆叠，
d=tf.read_file(r"D:\download\Downloads\facades\train\1.jpg")
d=tf.image.decode_jpeg(d)
d1=d[:,0:256,:]
d2=d[:,256:,:]
a=tf.stack((d1,d2),axis=2)
print(tf.shape(a))
