#详细都在这个网站上
import tensorflow as tf
# path='./checkpoint_dir/machao.ckpt'
# w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
# w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
# saver = tf.train.Saver()
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(w1))
# print(sess.run(w2))
# saver.save(sess, path)
# sess.close()
import tensorflow as tf
with tf.Session() as sess:
    #使用下面的那种方式恢复图模型，就不用把图模型再写一遍了
    saver = tf.train.import_meta_graph('.\checkpoint_dir\machao.ckpt.meta')
    # print(sess.run(w1))
    saver.restore(sess,'.\checkpoint_dir\machao.ckpt')
    a=sess.run("w1:0")#因为是使用的saver恢复的图，所以必须使用“w1:0"才能跑出结果
    print(sess.run("w2:0"))
    # print(sess.run(w2))
    print(a)
    


#1-----------实践--保存模型
import tensorflow as tf
 
 
w1 = tf.placeholder("float", name="w1")
w2 = tf.placeholder("float", name="w2")
b1= tf.Variable(2.0,name="bias") 
 
#定义一个op，用于后面恢复
w3 = tf.add(w1,w2)
w4 = tf.multiply(w3,b1,name="op_to_restore")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
 
#创建一个Saver对象，用于保存所有变量
saver = tf.train.Saver()
 
#通过传入数据，执行op
print(sess.run(w4,feed_dict ={w1:4,w2:8}))
#打印 24.0 ==>(w1+w2)*b1
 
#现在保存模型
saver.save(sess, './checkpoint_dir/MyModel',global_step=1000)

#2----------load模型
import tensorflow as tf
 
sess=tf.Session()
#先加载图和参数变量
saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')
saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
 
 
# 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
feed_dict ={w1:13.0,w2:17.0}
 
#接下来，访问你想要执行的op
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
 
print(sess.run(op_to_restore,feed_dict))
#打印结果为60.0==>(13+17)*2

