import tensorflow as tf
l2 = tf.keras.regularizers.l2
#主要介绍keras的BatchNormalization和droupout的使用
class ConvBlock(tf.keras.Model):
  """Convolutional Block consisting of (batchnorm->relu->conv).
  Arguments:
    num_filters: number of filters passed to a convolutional layer.
    data_format: "channels_first" or "channels_last"
    bottleneck: if True, then a 1x1 Conv is performed followed by 3x3 Conv.
    weight_decay: weight decay
    dropout_rate: dropout rate.
  """

  def __init__(self, num_filters, data_format, bottleneck, weight_decay=1e-4,
               dropout_rate=0):
    super(ConvBlock, self).__init__()
    self.bottleneck = bottleneck

    axis = -1 if data_format == "channels_last" else 1
    inter_filter = num_filters * 4
    # don't forget to set use_bias=False when using batchnorm
    self.conv2 = tf.keras.layers.Conv2D(num_filters,
                                        (3, 3),
                                        padding="same",
                                        use_bias=False,
                                        data_format=data_format,
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=l2(weight_decay))
    self.batchnorm1 = tf.keras.layers.BatchNormalization(axis=axis)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

    if self.bottleneck:
      self.conv1 = tf.keras.layers.Conv2D(inter_filter,
                                          (1, 1),
                                          padding="same",
                                          use_bias=False,
                                          data_format=data_format,
                                          kernel_initializer="he_normal",
                                          kernel_regularizer=l2(weight_decay))
      self.batchnorm2 = tf.keras.layers.BatchNormalization(axis=axis)

  def call(self, x, training=True):
    output = self.batchnorm1(x, training=training)

    if self.bottleneck:
      output = self.conv1(tf.nn.relu(output))
      output = self.batchnorm2(output, training=training)

    output = self.conv2(tf.nn.relu(output))
    output = self.dropout(output, training=training)

    return output
