import tensorflow as tf
#1.常数形式
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
#
#
#
#2.变量形式
my_variable = tf.get_variable("my_variable", [1, 2, 3])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())#需要初始化下
    print(my_variable.eval())
