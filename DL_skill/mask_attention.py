import tensorflow as tf
tf.enable_eager_execution()
def vanilla_attention(queries, keys, keys_length):
  '''
    queries:     [B, H]
    keys:        [B, T, H]
    keys_length: [B]
  '''
  #key_length里面存储的是每个时间序列的实际长度
  #queries = tf.tile(queries, [1, 2])
  queries = tf.expand_dims(queries, 1) # [B, 1, H]
  # Multiplication
  outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1])) # [B, 1, T]
  # Mask
  key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
  key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)#padding设置非常大的负数
  outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]
  #tf.where(cond,x,y),若cond对应位置为true，返回x对应位置，否则返回y的对应位置
  # Scale
  outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
  print(outputs)
  #最后softmax时原来非常大的负数变成了0
  outputs = tf.nn.softmax(outputs)  # [B, 1, T]
  print(outputs)
  # Weighted sum
  outputs = tf.matmul(outputs, keys)  # [B, 1, H]
  return outputs
queries=tf.random_normal((5,10),seed=2019)
keys=tf.random_normal((5,6,10),seed=2019)
keys_length=[3,4,2,3,4]
outputs=vanilla_attention(queries, keys, keys_length)

