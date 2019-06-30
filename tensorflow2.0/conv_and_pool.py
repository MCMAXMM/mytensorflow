import tensorflow as tf
from tensorflow import keras
tf .enable_eager_execution()
a1=tf.random_normal(shape=(1,4,3))
a2=tf.random_normal(shape=(1,4,2,3))
a3=tf.random_normal(shape=(1,4,2,2,3))
#只在中间的维度进行池化
out1=keras.layers.AveragePooling1D()(a1)
out2=keras.layers.AveragePooling2D()(a2)
out3=keras.layers.AveragePooling3D()(a3)
conv1d=keras.layers.Conv1D(12,2,padding="same")(a1)
print(conv1d.shape)
print(a1.shape)
print(out1.shape)
print(a2.shape)
print(out2.shape)
print(a3.shape)
print(out3.shape)
