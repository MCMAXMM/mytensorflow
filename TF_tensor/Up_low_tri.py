import tensorflow as tf
import numpy  as np
tf.enable_eager_execution()
data=tf.constant(np.ones((2,3,3)),dtype=tf.float32)
b=tf.constant([[[1.,2, 3],[4., 5., 6],[7., 8., 9.]],[[10.,11, 12],[13., 14., 15],[16., 17., 18.]]],dtype=tf.float32)
c=tf.equal(b,1)
f=tf.boolean_mask(data,c)
d=tf.ones((2,3,3),dtype=tf.float32)
e=tf.linalg.band_part(d,3,0)
mask=tf.equal(e,1)
value=tf.boolean_mask(b,mask)
value=tf.reshape(value,(2,6))
print(value)
