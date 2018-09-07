import tensorflow as tf
a=tf.constant(1.0)
b=tf.constant(2.0)
c=a+b
p = tf.placeholder(tf.float32)
t = p + 1.0

with tf.Session() as sess:
#如果你使用了eval()，你就不用使用sess去run it
  print(c.eval()
  print(t.eval())  # This will fail, since the placeholder did not get a value.
  print(t.eval(feed_dict={p:2.0}))  # This will succeed because we're feeding a value
                           # to the placeholder.
