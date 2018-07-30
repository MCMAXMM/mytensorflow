import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.set_random_seed(1234)
np.random.seed(123)
plt.rcParams["font.sans-serif"]="SimHei"
plt.rcParams["axes.unicode_minus"]=False
x=np.linspace(-2,2,200).reshape(200,1)
noise=np.random.normal(0,0.5,x.shape)
y=np.power(x,2)+noise
plt.subplot(121)
plt.hist(y,bins=50)
plt.subplot(122)
plt.scatter(x,y,c=x)
class regression_board:
    def __init__(self):
        self.x=tf.placeholder(tf.float32,[None,1])
        self.y=tf.placeholder(tf.float32,[None,1])
        with tf.variable_scope("hidden"):
            self.hidden1=tf.layers.dense(self.x,10,tf.nn.relu)
            self.hidden2=tf.layers.dense(self.hidden1,1)
            tf.summary.histogram("hidden1", self.hidden1)
            tf.summary.histogram("hidden2", self.hidden2)
        with tf.variable_scope("loss"):
            self.loss=tf.losses.mean_squared_error(self.y,self.hidden2)
            tf.summary.scalar("loss", self.loss)
        with tf.name_scope("train") as scope:
             self.merge_op = tf.summary.merge_all()
             self.train_op=tf.train.GradientDescentOptimizer(learning_rate=0.03).minimize(self.loss)

    def train(self,sess,data,label):
        result,loss,predict,_=sess.run([self.merge_op,self.loss,self.hidden2,self.train_op],feed_dict={self.x:data,self.y:label})
        print(loss)
        return result,loss,predict
    def save(self,sess,path):
        Saver=tf.train.Saver()
        Saver.save(sess,path)
    def restore(self,sess,path):
        Saver=tf.train.Saver()
        Saver.restore(sess,path)
mynet=regression_board()
save_path="./myparams"
log_path="./mylog"

init=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init)
    writer=tf.summary.FileWriter(log_path,graph=sess.graph)
    for step in range(6):
        result,loss,predict=mynet.train(sess,x,y)
        mynet.save(sess,save_path)
        writer.add_summary(result,step)
        print(step)
with tf.Session() as sess:
    mynet.restore(sess,save_path)
    result,loss,predict=mynet.train(sess,x,y)
    plt.plot(x,predict)
plt.show()
