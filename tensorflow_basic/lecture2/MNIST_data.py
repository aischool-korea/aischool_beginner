import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=2000)

print(np.shape(mnist.validation.images))
print(np.shape(mnist.validation.labels))
print(np.shape(mnist.train.images))
print(np.shape(mnist.train.labels))
print(np.shape(mnist.test.images))
print(np.shape(mnist.test.labels))
print(mnist.train.labels[7])
plt.imshow(
        mnist.train.images[7].reshape(28, 28),
        cmap="Greys",
        interpolation="nearest",
    )
plt.show()