import tensorflow as tf
#创建变量的方式有两种，但是切记变量一定需要初始化
#1.第一种
#a=tf.Variable(initial_value,name)
#2.第二种
my_variable = tf.get_variable("my_variable",initializer=np.ones((2,3)))
#如果initailizer后面的是一个constant,get_variable中shape就不用指定了，否则会报错
#上面的initializer可以是[1,2,3,4]这样的列表，也可以是tf.random.initializer(),也可以是np.ones()
with tf.Session() as sess:
    sess.run(tf.group(tf.local_variables_initializer(),tf.global_variables_initializer()))
    print(sess.run(my_variable))
#
#如果initializer后面用的是initializer()来初始化的，就需要指定shape
a=tf.get_variable("input",[2,3],initializer=tf.random_normal_initializer())
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(a))
#部分初始化变量
with tf.Session() sess:
    sess.run(a.initializer)#初始后面没有括号,这个操作只负责初始化
    print(sess.run(a))
