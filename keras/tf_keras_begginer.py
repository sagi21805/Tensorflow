import tensorflow as tf
print("TensorFlow version:", tf.__version__)
import numpy as np
import cv2

import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, verbose = 1, batch_size=32)

model.evaluate(x_test,  y_test, verbose=2, batch_size=1)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
img = cv2.cvtColor(cv2.imread(r"/home/sagi21805/Desktop/Vscode/machine-learning/Tensorflow/test.jpg"), cv2.COLOR_BGR2GRAY) / 255.0
print(img)
print(np.argmax(probability_model(np.reshape(img, (1, 28, 28)))))
print(probability_model(np.reshape(img, (1, 28, 28))))

plt.imshow(img)
plt.show()

