import tensorflow as tf
tf.enable_eager_execution()
#a shape:[1,2,2,1]
a=tf.constant([[[[1],[2]],[[3],[4]]]],dtype=tf.float32)
#b shape:[1,1,1,4]
#space_to_depth 主要是将wh转移到channel上,block_size必须能够整除w,h 
b=tf.space_to_depth(a,2,data_format="NHWC")
#c shape:[1,2,2,1]
c=tf.depth_to_space(b,2)#上面操作的逆操作
print(c.shape)
print(b.shape)
