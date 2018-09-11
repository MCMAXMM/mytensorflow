#这里面主要介绍了卷积，反卷积，上采样，same，valid 之后tensor尺寸的变化
import tensorflow as tf

from tensorflow import keras
tf.enable_eager_execution()
# data=tf.random_normal([32,27,27,3],dtype=tf.float32)
# conv1=keras.layers.Conv2D(10,(5,5),strides=(2,2),padding="same")(data)
# print("conv1",conv1.shape)
# maxpool=keras.layers.MaxPool2D((6,7),(2,2),padding="valid")(conv1)
# print(maxpool.shape)
# #不管卷积还是池化，只要是valid,计算尺寸时就是ceil((width-filter_width+1)/stride_width)
# #不管卷积还是池化，只要是same,计算尺寸时就是ceil(width/stride_width)
# #反卷积就是逆向操作就行
data1=tf.random_normal([32,27,27,3],dtype=tf.float32)
data1=keras.layers.Conv2DTranspose(32,(5,5),(2,2),padding="same")(data1)#反卷积
print(data1.shape)

a=tf.constant([[[[1,2,3,4],[5,6,7,8]]]])
print(a.shape)
data=tf.keras.layers.UpSampling2D((2,2))(a)#上采样，将数据的行和列分别复制扩大n倍,这里是都扩大2倍
print(data.shape)
print(data)
