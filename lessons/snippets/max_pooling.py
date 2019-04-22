from keras.layers import MaxPooling2D

# create a convolutional layer using the pooling api
MaxPooling2D(pool_size, strides, padding)

"""
pool_size - Number specifying the height and width of the pooling window.
There are some additional, optional arguments that you might like to tune:

strides - The vertical and horizontal stride. If you don't specify anything, strides will default to pool_size.
padding - One of 'valid' or 'same'. If you don't specify anything, padding is set to 'valid'.

NOTE: It is possible to represent both pool_size and strides as either a number or a tuple.
"""

MaxPooling2D(pool_size=2, strides=2)
# MaxPooling2D(pool_size=2, strides=1)
