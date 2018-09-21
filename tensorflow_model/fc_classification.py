import tensorflow as tf
from matplotlib import cm
import numpy as np
import matplotlib.pyplot  as plt

plt.rcParams["font.sans-serif"]="SimHei"
plt.rcParams["axes.unicode_minus"]=False
tf.set_random_seed(12)
np.random.seed(12)
data=np.ones((100,2))
noise=np.random.normal(0,1,data.shape)
data1=2*data+noise
label1=np.zeros((100,1))
label2=np.ones((100,1))
data2=-2*data+noise
data=np.vstack((data1,data2))
label=np.vstack((label1,label2))
print(data.shape)

class mynetwork(object):
    def __init__(self,data,label):#data,label可以在类中的所有方法调用
        self.x=tf.placeholder(tf.float32,[None,2],name="input")
        self.y=tf.placeholder(tf.int32,[None,1],name="output")
        self.hidden1=tf.layers.dense(self.x,10,activation=tf.nn.relu)
        hidden2=tf.layers.dense(self.hidden1,2)#不加self的话无法再剩下的方法中使用这个变量
        self.loss=tf.losses.sparse_softmax_cross_entropy(self.y,hidden2)
        self.optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(self.loss)
        self.predict=tf.argmax(hidden2,axis=1)
        self.accuracy=tf.metrics.accuracy(labels=self.y,predictions=self.predict)[1]

    def train(self,sess):
            accuracy,loss,_=sess.run([self.accuracy,self.loss,self.optimizer],feed_dict={self.x:data,self.y:label})
            print(loss)
            return accuracy
    def get_predict(self):
        predict = sess.run([self.predict], feed_dict={self.x: data, self.y: label})
        return predict
    def plot(self,predict,accuracy):
        plt.cla()
        plt.scatter(data[:,0],data[:,1],c=predict)
        plt.xlabel("x轴")
        plt.ylabel("y轴")
        plt.ylim(-4,4)
        plt.xlim(-4,4)
        plt.xticks(np.linspace(-4,4,8))
        plt.yticks(np.linspace(-4,4,8))
        plt.text(3,3,s="accuracy=%5.4f"% accuracy)
        plt.pause(0.5)


net=mynetwork(data,label)
with tf.Session() as sess:
    init=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init)
    for step in range(100):
        # loss, _ = sess.run([net.loss, net.optimizer], feed_dict={net.x: data, net.y: label})
        # print(loss)
        accuracy=net.train(sess)
        print(accuracy)
        net.plot(np.array(net.get_predict()).reshape(200,),accuracy)
plt.show()
