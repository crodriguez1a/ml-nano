from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(10, activation='softmax'))

"""
The network begins with a sequence of three convolutional layers, followed by max pooling layers.
These first six layers are designed to take the input array of image pixels and convert it
to an array where all of the spatial information has been squeezed out, and only information
encoding the content of the image remains. The array is then flattened to a vector
in the seventh layer of the CNN. It is followed by two dense layers designed to
further elucidate the content of the image. The final layer has one entry for
each object class in the dataset, and has a softmax activation function, so that it returns probabilities.

Things to Remember
Always add a ReLU activation function to the Conv2D layers in your CNN. With the exception of the final layer in the network, Dense layers should also have a ReLU activation function.
When constructing a network for classification, the final layer in the network should be a Dense layer with a softmax activation function. The number of nodes in the final layer should equal the total number of classes in the dataset.
Have fun! If you start to feel discouraged, we recommend that you check out Andrej Karpathy's tumblr with user-submitted loss functions, corresponding to models that gave their owners some trouble. Recall that the loss is supposed to decrease during training. These plots show very different behavior :).
"""

"""
On filters:
ref: https://stackoverflow.com/questions/36243536/what-is-the-number-of-filter-in-cnn
if you have 28x28 input images and a convolutional layer with 20 7x7 filters and stride 1,
you will get 20 22x22 feature maps at the output of this layer. Note that this is presented
to the next layer as a volume with width = height = 22 and depth = num_channels = 20.
You could use the same representation to train your CNN on RGB images such as the ones
from the CIFAR10 dataset, which would be 32x32x3 volumes (convolution is applied only
to the 2 spatial dimensions).
"""
