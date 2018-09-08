import tensorflow as tf
import numpy as np
a=tf.Graph()
#如果你想创建自己的图，那么在创建Session()时要指定这个图(graph)是哪个
#session要和它的图对应，不指定的话，session对应的就是默认图
with a.as_default():
    b=tf.get_variable("input",initializer=np.ones((2,3)))
with tf.Session(graph=a) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))
