import tensorflow as tf
input1 = tf.random_normal([1,10,10,32])(训练)
input2 = tf.random_normal([1,20,20,32])(测试)
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])
#第一种共享变量的形式，创建的varialbe_scope有相同的名字
with tf.variable_scope("model"):
  output1 = my_image_filter(input1)
with tf.variable_scope("model", reuse=True):
  output2 = my_image_filter(input2)
#第二种共享变量的形式，使用scope.reuse_variables()激活
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
  scope.reuse_variables()
  output2 = my_image_filter(input2)
#因为使用字符串名字不太安全，可以使用下面的方式
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
with tf.variable_scope(scope, reuse=True):
  output2 = my_image_filter(input2)
######################
#########################
#########################
#我的练习代码
import tensorflow as tf
with tf.variable_scope("a_variable_scope") as scope:
    initializer=tf.constant_initializer(5.0)
    var3=tf.get_variable(name="var3",shape=[1],dtype=tf.float32,initializer=initializer)
    print(tf.get_variable_scope())#<tensorflow.python.ops.variable_scope.VariableScope object at 0x000000001784B898>
    var4 = tf.get_variable(name="var4", initializer=tf.constant([4.1],dtype=tf.float32),dtype=tf.float32)
    scope.reuse_variables()

    print(tf.get_variable_scope())#<tensorflow.python.ops.variable_scope.VariableScope object at 0x000000001784B898>
    var3_reuse=tf.get_variable(name="var3")
    var4_reuse=tf.get_variable(name="var4",initializer=tf.constant([4.0]),dtype=tf.float32)
    #重用后的变量，变量名字（里面那个name）必须相同，数据类型必须相同，数值不必相同
    b=tf.assign(var4_reuse,[5.0])#给重用后的变量4重新赋值
print(var3.name)#a_variable_scope/var3:0
print(var3_reuse.name)#a_variable_scope/var3:0
print(var4.name)#a_variable_scope/var4:0
print(var4_reuse.name)#a_variable_scope/var4:0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))#[5.]
    print(sess.run(var4_reuse))#[5.]
    print(sess.run(var4))#[5.]
