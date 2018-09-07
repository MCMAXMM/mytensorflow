import tensorflow as tf
#创建变量的方式有两种，但是切记变量一定需要初始化
#1.第一种
#a=tf.Variable(initial_value,name)
#2.第二种
my_variable = tf.get_variable("my_variable",initializer=np.ones((2,3)))
#上面的initializer可以是[1,2,3,4]这样的列表，也可以是tf.random.initializer(),也可以是np.ones()
with tf.Session() as sess:
    sess.run(tf.group(tf.local_variables_initializer(),tf.global_variables_initializer()))
    print(sess.run(my_variable))
