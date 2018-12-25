import tensorflow as tf
tf.enable_eager_execution()
d=tf.read_file(r"D:\download\Downloads\facades\train\1.jpg")
file=tf.data.Dataset.list_files("D:/download/Downloads/facades/train/*.jpg").make_one_shot_iterator()
print(file.get_next())
print(file.get_next())
#tf.Tensor(b'D:\\download\\Downloads\\facades\\train\\389.jpg', shape=(), dtype=string)
#tf.Tensor(b'D:\\download\\Downloads\\facades\\train\\315.jpg', shape=(), dtype=string)
