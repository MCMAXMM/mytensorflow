import tensorflow as tf
from tensorflow import keras
tf.set_random_seed(1234)
train_data,test_data=keras.datasets.mnist.load_data()
test_datas=test_data[0][0:3000]
test_datas=test_datas.reshape((-1,784))
# test_datas=tf.convert_to_tensor(test_datas,dtype=tf.float32,name="test_data")
test_label=test_data[1][0:3000].reshape(3000,1)
# test_label=tf.convert_to_tensor(test_label,dtype=tf.int32,name="test_label")
batch=1000
# data=tf.placeholder(dtype=tf.float32,shape=[None,784],name="input")
# label=tf.placeholder(dtype=tf.int32,shape=[None,1],name="label")
# with tf.variable_scope("fullconect"):
#     tmp=tf.layers.dense(data,384,name="layer1")
#     tmp=tf.nn.relu(tmp,name="relu1")
#     tmp=tf.layers.dense(tmp,128,name="layer2")
#     tmp=tf.nn.relu(tmp,name="relu2")
#     tmp=tf.layers.dense(tmp,32,name="layer3")
#     tmp=tf.nn.relu(tmp,name="relu3")
#     tmp=tf.layers.dense(tmp,10,name="layer4")
# with tf.name_scope("train"):
#     optimizer=tf.train.GradientDescentOptimizer(0.001)
#     loss=tf.losses.sparse_softmax_cross_entropy(labels=label,logits=tmp)
#     tf.summary.scalar(name="loss",tensor=loss)
#     tf.add_to_collection(name="loss",value=loss)
#
# merge=tf.summary.merge_all()
# with tf.name_scope('save'):
#     saver=tf.train.Saver()
# with tf.Session() as sess:
#     writer = tf.summary.FileWriter(logdir="./masummary",graph=sess.graph)
#     sess.run(tf.global_variables_initializer())
#     for epoch in range(100):
#        safd,losses,result =sess.run([train_op,loss,merge],feed_dict={data:test_datas,label:test_label})
#        writer.add_summary(result,global_step=epoch)
#     saver.save(sess=sess,save_path="./machao/checkpoint/model.ckpt")
sess=tf.Session()
saver=tf.train.import_meta_graph(r"./machao/checkpoint/model.ckpt.meta")
saver.restore(sess,save_path=r"./machao/checkpoint/model.ckpt")
graph=tf.get_default_graph()
data=graph.get_tensor_by_name("input:0")
label=graph.get_tensor_by_name("label:0")
relu3=graph.get_tensor_by_name("fullconect/layer2/BiasAdd:0")
kernel=graph.get_tensor_by_name("fullconect/layer1/kernel:0")
tf.summary.histogram("weight",kernel)
tf.stop_gradient(relu3)

with tf.name_scope("reset"):
    output=tf.layers.dense(relu3,64,name="reset1")
    output=tf.layers.dense(output,10,name="reset2")
optimizer=tf.train.GradientDescentOptimizer(0.001)
loss=tf.losses.sparse_softmax_cross_entropy(labels=label,logits=output)
tf.summary.scalar("loss",loss)
train_op=optimizer.minimize(loss)
var=tf.trainable_variables(scope="reset")
writer=tf.summary.FileWriter("./mylog",sess.graph)
merge=tf.summary.merge_all()
sess.run(tf.initialize_variables(var))
for epoch in range(100):
    train_op=optimizer.minimize(loss)
    a,e,b,c=sess.run([merge,kernel,train_op,loss],feed_dict={data:test_datas,label:test_label})
    writer.add_summary(a,global_step=epoch)
    print(c)
