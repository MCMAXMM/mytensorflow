import tensorflow as tf
tf.enable_eager_execution()
d=tf.read_file(r"D:\download\Downloads\facades\train\1.jpg")
file=tf.data.Dataset.list_files("D:/download/Downloads/facades/train/*.jpg").make_one_shot_iterator()
print(file.get_next())
print(file.get_next())
#tf.Tensor(b'D:\\download\\Downloads\\facades\\train\\389.jpg', shape=(), dtype=string)
#tf.Tensor(b'D:\\download\\Downloads\\facades\\train\\315.jpg', shape=(), dtype=string)

train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.map(lambda x: load_image(x, True))
#train_dataset是一个可以迭代的东西，map将train_dataset中的每一个元素应用于一个函数
train_dataset = train_dataset.batch(1)
