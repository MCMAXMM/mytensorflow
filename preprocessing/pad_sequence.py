import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
tf.enable_eager_execution()
#显示的使用列表
a=list([[1,2,3,4],[1,2,3],[1,2],[1,2,3,4,5,6,7]])
b=np.array(a)
print(b)
dataset=keras.preprocessing.sequence.pad_sequences(np.array(a),maxlen=10,padding="post")
print(dataset)
