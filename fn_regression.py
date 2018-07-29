"""
使用面向对象的方式创建一个回归网络，并保存模型
使用面向对象的方式创建网络时，方便训练好的参数的导入
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
tf.set_random_seed(12)
np.random.seed(12)
#定义我的超参数
H_SIZE=10
LR=0.03
TRAIN_NUMBER=100
PATH="./params/"
#生成训练数据
x=np.linspace(-3,3,200).reshape(200,1)
noise=np.random.normal(0,2,x.shape)
y=np.power(x,3)+noise
#可视化生成好的数据
plt.scatter(x,y,cmap=cm.get_cmap("rainbow"))
# plt.show()
#创建我的网络
class Regression:
    def __init__(self,hidden_size,lr):
        self.x=tf.placeholder(tf.float32,shape=[None,1])
        self.y=tf.placeholder(tf.float32,shape=[None,1])
        hidden1=tf.layers.dense(self.x,hidden_size,activation=tf.nn.relu)
        self.predict=tf.layers.dense(hidden1,1)
        self.loss=tf.losses.mean_squared_error(self.y,self.predict)
        self.train_op=tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(self.loss)
    #定义训练的方法
    def train(self,sess,data,label):
        predict,loss,_=sess.run([self.predict,self.loss,self.train_op],feed_dict={self.x:data,self.y:label})
        return predict,loss
    #定义绘画的方法
    def plot(self,data,label,predict):
        plt.cla()
        plt.scatter(data,label,c="green")
        plt.plot(data,predict,c="red")
        plt.pause(0.5)
    #保存我的参数
    def save(self,sess,path):
        saver=tf.train.Saver()
        saver.save(sess,path,write_meta_graph=False)
    #把保存好的参数调到我的模型中
    def restore(self,sess,path):
        saver=tf.train.Saver()
        saver.restore(sess,path)

net1=Regression(H_SIZE,LR)
net2=Regression(H_SIZE,LR)
#第一次训练过程
# with tf.Session() as sess:
#     init=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
#     sess.run(init)
#     for step in range(TRAIN_NUMBER):
#         predict,loss=net1.train(sess,x,y)
#         print(loss)
#         net1.plot(x,y,predict)
#         net1.save(sess,PATH)
# plt.show()
#导模型
with tf.Session() as sess:
    net1.restore(sess,PATH)
    predict,loss=net1.train(sess,x,y)
    print(loss)
    net1.plot(x,y,predict)
    plt.show()
