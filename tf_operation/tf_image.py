import tensorflow as tf
from tensorflow import keras
#使用keras下载文件需要制定文件的名称，保存的路径以及下载的路径
path_to_zip=tf.keras.utils.get_file('facades.tar.gz',
               cache_subdir=os.path.abspath('.'),
             origin='https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz',
               extract=True)
# PATH=os.path.join(os.path.dirname(path_to_zip),"facades/")
#os.path.abspath(".")返回当前路径的绝对路径
image=tf.read_file("D:\pythonrun\open_cv\hello.jpg")#使用tf读取图片，此时读出来的数据是二进制数据
image=tf.image.decode_jpeg(image)#相当于将二进制数据解码成jpeg格式，将二进制数据转换成图片矩阵tensor格式,显示的数据
#有其他的如.decode_bmp() .decode_gif(),.decode_png()等等
#是0—255之间的tensor数据
print(image)
#使用dataset获取各个图片的路径
train_dataset = tf.data.Dataset.list_files(r"D:\pythonrun\2018613pytorch\faces\*.jpg")
train_dataset=train_dataset.make_one_shot_iterator()
print(train_dataset.get_next())
#output：tf.Tensor(b'D:\\pythonrun\\2018613pytorch\\faces\\1084239450_e76e00b7e7.jpg', shape=(), dtype=string)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
#下面使用自己定义的load_image()函数来对数据集中每个图片的路径进行处理返回训练的图片
train_dataset = train_dataset.map(lambda x: load_image(x, True))
train_dataset = train_dataset.batch(1)

#使用tensorflow加载本地图片
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.enable_eager_execution()
b=tf.data.Dataset.list_files(r"D:\pythonrun\2018613pytorch\faces\*.jpg")
#获取本地图片的路径String
print(b)
def load_image(file_path):
    data=tf.read_file(file_path)
    image=tf.image.decode_jpeg(data)
    return image#解码成jpeg

b=b.map(lambda x:load_image(x))
c=b.make_one_shot_iterator().get_next()
a=np.array(c).astype(np.int16)
plt.imshow(a)#显示图片
plt.show()
