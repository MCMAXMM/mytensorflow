import tensorflow as tf
#http://web.stanford.edu/class/cs20si/lectures/notes_02.pdf
#lazy loading
#solution
#https://danijar.com/structuring-your-tensorflow-models/
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        print(sess.run(tf.add(x, y))) # create the op add only when you need to compute it
    print(tf.get_default_graph().as_graph_def())
