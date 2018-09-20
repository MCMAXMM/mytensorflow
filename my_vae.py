from __future__ import absolute_import, division, print_function
import tensorflow as tf
tfe=tf.contrib.eager
tf.enable_eager_execution()
import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
from IPython import display


#导入数据
(train_images,_),(test_images,_)=tf.keras.datasets.mnist.load_data()
train_images=train_images.reshape(-1,28,28,1).astype("float32")
test_images=test_images.reshape(-1,28,28,1).astype("float32")
train_images/=255.
test_images/=255.
#矩阵二值化
train_images[train_images>=.5]=1.
train_images[train_images<.5]=0#将矩阵二值化
test_images[test_images>=.5]=1
test_images[test_images<.5]=0
TRAIN_BUF=60000
BATCH_SIZE=100
TEST_BUF=10000
#生成数据集
train_dataset=tf.data.Dataset.from_tensor_slices(train_images[1:200]).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset=tf.data.Dataset.from_tensor_slices(test_images[1:200]).shuffle(TEST_BUF).batch(BATCH_SIZE)
#define model
class CVAE(tf.keras.Model):
    def __init__(self,latent_dim):
        super(CVAE,self).__init__()
        self.latent_dim=latent_dim
        #推断网络  X--->Z ——encoder
        self.inference_net=tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(28,28,1)),
        tf.keras.layers.Conv2D(32,3,(2,2),activation=tf.nn.relu),
        tf.keras.layers.Conv2D(64,3,(2,2),activation=tf.nn.relu),tf.keras.layers.Flatten(),
                                                tf.keras.layers.Dense(latent_dim+latent_dim),])
        #生成网络 Z--->X ——decoder
        self.generative_net=tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                                                 tf.keras.layers.Dense(units=7*7*32,activation=tf.nn.relu),
                                                 tf.keras.layers.Reshape(target_shape=(7,7,32)),
                                                 tf.keras.layers.Conv2DTranspose(64,3,(2,2),padding="same",activation=tf.nn.relu),
                                                 tf.keras.layers.Conv2DTranspose(32,3,(2,2),padding="same",activation=tf.nn.relu),
                                                 tf.keras.layers.Conv2DTranspose(1,3,(1,1),padding="same"),])
    def sample(self,eps=None):#抽样Gaussian	N(μ;RR⊤)可以重新参数化 从一个简单分布中抽	ϵ∼N(0;1)得到的这个值μ+Rϵ服从所要求的分布
    #详细的介绍在这个网址http://blog.shakirm.com/2015/10/machine-learning-trick-of-the-day-4-reparameterisation-tricks/
        if eps is None:
            eps=tf.random_normal(shape=(100,self.latent_dim))#p(z)
        return self.decode(eps,apply_sigmoid=True)#p(x|z)
    def encode(self,x):
        mean,logvar=tf.split(self.inference_net(x),num_or_size_splits=2,axis=1)#q(z|x)
        return mean,logvar#inference返回的是latent变量的均值，log方差（对角矩阵）
    def reparameterize(self,mean,logvar):
        eps=tf.random_normal(shape=mean.shape)
        return eps*tf.exp(logvar*.5)+mean
    def decode(self,z,apply_sigmoid=False):
        logits=self.generative_net(z)
        if apply_sigmoid:
            probs=tf.sigmoid(logits)
            return probs
        return logits
def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_loss(model, x):
  mean, logvar = model.encode(x)#q(z|x)
  z = model.reparameterize(mean, logvar)#重新参数化有关z的东西
  x_logit = model.decode(z)#p(x|z)

  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_gradients(model, x):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  return tape.gradient(loss, model.trainable_variables), loss

optimizer = tf.train.AdamOptimizer(1e-4)
def apply_gradients(optimizer, gradients, variables, global_step=None):
    optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
epochs = 100
latent_dim = 50
num_examples_to_generate = 16
random_vector_for_generation = tf.random_normal(
      shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)
def generate_and_save_images(model, epoch, test_input):
    predictions = model.sample(test_input)
    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
generate_and_save_images(model, 0, random_vector_for_generation)
for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        gradients, loss = compute_gradients(model, train_x)
        apply_gradients(optimizer, gradients, model.trainable_variables)
    end_time = time.time()

    if epoch % 1 == 0:
        print(epoch)
        loss = tfe.metrics.Mean()
        for test_x in test_dataset.make_one_shot_iterator():
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        display.clear_output(wait=False)
        print('Epoch: {}, Test set ELBO: {}, '
              'time elapse for current epoch {}'.format(epoch,
                                                        elbo,
                                                        end_time - start_time))
        generate_and_save_images(
            model, epoch, random_vector_for_generation)



