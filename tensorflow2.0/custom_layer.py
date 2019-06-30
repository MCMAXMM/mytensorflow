class CustomDense(layers.Layer):

  def __init__(self, units=32):
    super(CustomDense, self).__init__()
    self.units = units
#通常来说变量必须在__init__中定义，否则模型中可训练的变量中不会出现你定义的变量，
#当然你也可以build中创建变量，而不用在__init__中创建变量，减轻构造方法的压力（分工明确）
#build中有一个参数input_shape是程序自动得到的，
#程序运行时会先调用build方法，在调用call方法
  def build(self, input_shape):
    self.shape=input_shape
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)

  def call(self, inputs):
    print("hello")
    print(self.shape)
    return tf.matmul(inputs, self.w) + self.b

inputs = keras.Input((4,))
outputs = CustomDense(10)(inputs)

model = keras.Model(inputs, outputs)
print(model.variables)
