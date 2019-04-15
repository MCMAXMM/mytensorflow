strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
  model = tf.keras.Sequential()
  model.add(layers.Dense(16, activation='relu', input_shape=(10,)))
  model.add(layers.Dense(1, activation='sigmoid'))

  optimizer = tf.keras.optimizers.SGD(0.2)

  model.compile(loss='binary_crossentropy', optimizer=optimizer)

model.summary()
