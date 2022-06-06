
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
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, 
            activation=None, data_format=data_format)(x)
    c = keras_contrib.layers.GroupNormalization(groups=groups, axis=channel_axis)(c)
    c = keras.layers.advanced_activations.PReLU()(c)
    return c

def down_conv(x, filters, kernel_size, padding, data_format, groups):
    channel_axis = -1 if data_format=='channels_last' else 1
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=2, 
                            activation=None, data_format=data_format)(x)
    c = keras_contrib.layers.GroupNormalization(groups=groups, axis=channel_axis)(c)
    c = keras.layers.advanced_activations.PReLU()(c)
    return c

def up_conv_concat_conv(x, skip, filters, kernel_size, padding, strides, data_format, groups):
    channel_axis = -1 if data_format=='channels_last' else 1
    c = keras.layers.Conv3DTranspose(filters, kernel_size=(2,2,2), strides=(2,2,2), 
                                    data_format=data_format)(x) # up dim(x) by x2
    c = keras_contrib.layers.GroupNormalization(groups=groups, axis=channel_axis)(c)
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, 
                            activation=None, data_format=data_format)(c)
    concat = keras.layers.Concatenate(axis=channel_axis)([c, skip]) # concat after Up; dim(skip) == 2*dim(x)
    c = keras_contrib.layers.GroupNormalization(groups=groups, axis=channel_axis)(concat)
    c = keras.layers.advanced_activations.PReLU()(c)
    return c


# Encoders