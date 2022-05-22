
import keras
import keras_contrib
import tensorflow as tf


# Building blocks
def adding_conv(x, a, filters, kernel_size, padding, strides, data_format, groups):
    channel_axis = -1 if data_format=='channels_last' else 1
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, 
            activation=None, data_format=data_format)(x)
    c = keras.layers.add([c, a])
    c = keras_contrib.layers.GroupNormalization(groups=groups, axis=channel_axis)(c)
    c = keras.layers.advanced_activations.PReLU()(c)
    return c

def conv(x, filters, kernel_size, padding, strides, data_format, groups):
    channel_axis = -1 if data_format=='channels_last' else 1