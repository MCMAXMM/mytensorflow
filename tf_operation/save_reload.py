import tensorflow as tf 
#常用的保存方式
bottom = layers.fully_connected(inputs=bottom, num_outputs=7, activation_fn=None, scope='logits_classifier'
prediction = tf.nn.softmax(logits, name='prob')
saver_path = './model/checkpoint/model.ckpt'
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    sess.run(init)
    saved_path = saver.save(sess,saver_path) #这个保存了三个东西， .meta是图的结构， 还有两个是模型中变量的值
    
       
     
     
     
     #要想图结构和模型（恢复图结构，没错，从空白的代码段中恢复一个graph，就不需要重新定义图了）  
    meta_path = './model/checkpoint/model.ckpt.meta'
    model_path = './model/checkpoint/model.ckpt'
    saver = tf.train.import_meta_graph(meta_path) # 导入图
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        saver.restore(sess, model_path) # 导入变量值
        graph = tf.get_default_graph()
        prob_op = graph.get_operation_by_name('prob') # 这个只是获取了operation， 至于有什么用还不知道
	prediction = graph.get_tensor_by_name('prob:0') # 获取之前prob那个操作的输出，即prediction
	print( ress.run(prediciton, feed_dict={...})) # 要想获取这个值，需要输入之前的placeholder （这里我编辑文章的时候是在with里面的，不知道为什么查看的时候就在外面了...）
  print(sess.run(graph.get_tensor_by_name('logits_classifier/weights:0'))) # 这个就不需要feed了，因为这是之前train operation优化的变量，即模型的权重




#关于获取保存的模型中的tensor或者输出，还有一种办法就是用tf.add_to_collection()，
#假如上面每次定义一次运算后，可以在后面添加tf.add_to_collection()：
    ......
    bottom = layers.fully_connected(inputs=bottom, num_outputs=7, activation_fn=None, scope='logits_classifier')
    ### add collection
    tf.add_to_collection('logits',bottom)
    ......
    prediction = tf.nn.softmax(logits, name='prob')
    ### add collection
    tf.add_to_collection('prob',prediction)
    ......
#恢复模型后，通过tf.get_collection()来获取tensor：
    ......
    x = tf.get_collection('inputs')[0]
    prob = tf.get_collection('prob')[0]
    print(x)
    print(prob)
    .....
