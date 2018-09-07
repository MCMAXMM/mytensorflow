import tensorflow as tf
#一种是官方中的collection是
#By default every tf.Variable gets placed in the following two collections: 
    #tf.GraphKeys.GLOBAL_VARIABLES --- variables that can be shared across multiple devices,        
    #tf.GraphKeys.TRAINABLE_VARIABLES --- variables for which TensorFlow will calculate gradients.
#官方的collections
my_local = tf.get_variable("my_local", shape=(),collections=[tf.GraphKeys.LOCAL_VARIABLES])
my_non_trainable = tf.get_variable("my_non_trainable",
                                   shape=(),
                                   trainable=False)
#自己创建的到时候按照顺序去取数据
#放数据
tf.add_to_collection("my_collection_name", my_local)
#取数据
tf.get_collection("my_collection_name")
