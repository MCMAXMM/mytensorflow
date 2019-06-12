import tensorflow as tf
import scipy.sparse as sp
tf.enable_eager_execution()
tags=tf.SparseTensor(indices=[[0, 0], [1, 2],[1,3]], values=[1,2,1], dense_shape=[3, 4])
embedding_params=tf.Variable(tf.truncated_normal([3,3]))
print(tags)
embedded_tags=tf.nn.embedding_lookup_sparse(embedding_params,sp_ids=tags,sp_weights=None)
print(embedding_params)
print(embedded_tags)
