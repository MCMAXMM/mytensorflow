import tensorflow as tf
from tensorflow import  keras
#图模型
a = tf.sequence_mask([1, 2, 3], 5)
b = tf.sequence_mask([[1, 2], [3, 4]])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(b))
    
    
#keras模型    
#keras怎么处理不等长的序列
X=keras.preprocessing.sequence.pad_sequences(X, maxlen=100, padding='pre')
model = keras.Sequential()
model.add(keras.layers.Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(keras.layers.LSTM(32))
