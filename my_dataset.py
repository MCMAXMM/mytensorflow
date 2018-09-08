import tensorflow as tf
#######有关dataset的操作
###一个关于dataset的操作还有一个是关于iterator的操作
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset2.output_shapes)  # ==> "((), (100,))"

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"

########################################
dataset = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"
###################################################
dataset1 = dataset1.map(lambda x: ...)

dataset2 = dataset2.flat_map(lambda x, y: ...)

# Note: Argument destructuring is not available in Python 3.
dataset3 = dataset3.filter(lambda x, (y, z): ...)


#前两种迭代器是通过dataset中iterator实现的，后两种是通过tf.data.Iterator.来创建迭代器
############################################3
#第一种迭代器，这是最简单的迭代器，不用初始化，如果不调用repeat()只能使用一次
#one-shot,

# dataset=tf.data.Dataset.range(100)
# dataset=dataset.make_one_shot_iterator()
# mydata=dataset.get_next()
# with tf.Session() as sess:
#     for i in range(20):
#         print(sess.run(mydata))
#第二种迭代器，需要初始化
#initiable

max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
sess=tf.Session()
# Initialize an iterator over a dataset with 10 elements.
sess.run(iterator.initializer, feed_dict={max_value: 10})
for i in range(10):
  value = sess.run(next_element)
  assert i == value

# Initialize the same iterator over a dataset with 100 elements.
sess.run(iterator.initializer, feed_dict={max_value: 100})
for i in range(100):
  value = sess.run(next_element)
  assert i == value
sess.close()
#第三种迭代器，比较适用于不同的数据集但是有相同的数据类型和数据结构，例如，测试集和验证集
#reinitializable
#这种迭代器是通过它的结构来定义的  如 tf.data.Iterator.from_structure

# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

# A reinitializable iterator is defined by its structure. We could use the
# `output_types` and `output_shapes` properties of either `training_dataset`
# or `validation_dataset` here, because they are compatible.
iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# Run 20 epochs in which the training dataset is traversed, followed by the
# validation dataset.
for _ in range(20):
  # Initialize an iterator over the training dataset.
  sess.run(training_init_op)
  for _ in range(100):
    sess.run(next_element)

  # Initialize an iterator over the validation dataset.
  sess.run(validation_init_op)
  for _ in range(50):
    sess.run(next_element)
#第四种迭代器，可以和tf.placeholder一块使用

import tensorflow as tf
#第一步创建数据集
training_dataset=tf.data.Dataset.range(100).map(lambda x: x+tf.random_uniform([],-10,10,tf.int64)).repeat()
validation_dataset=tf.data.Dataset.range(50)
#第二步创建handle
handle=tf.placeholder(tf.string,shape=[])
#使用Iterator创建一个iterator，更像一个管道中介用来流数据的
iterator=tf.data.Iterator.from_string_handle(handle,training_dataset.output_types,training_dataset.output_shapes)
next_element=iterator.get_next()
#创建数据集的iterator
training_iterator=training_dataset.make_one_shot_iterator()
validation_iterator=validation_dataset.make_initializable_iterator()
with tf.Session() as sess:
    #获取handle
    training_handle=sess.run(training_iterator.string_handle())
    print(training_handle)
    validation_handle = sess.run(validation_iterator.string_handle())
    while True:
        for _ in range(200):
            #通过handle来获取数据
            print(sess.run(next_element,feed_dict={handle:training_handle}))
        sess.run(validation_iterator.initializer)
        for _ in range(50):
            print(sess.run(next_element, feed_dict={handle: validation_handle}))
