import tensorflow as tf
a=tf.constant(1.0)
b=tf.constant(2.0)
c=a+b
with tf.Session() as sess:
#如果你使用了eval()，你就不用使用sess去run it
  print(c.eval()
