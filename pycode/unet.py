#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Import usual libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.backend import *
from tensorflow import keras

from distutils.log import error

#Function to add 2 convolutional layers with the parameters passed to it
def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True, data_format = "channels_first", concat_axis = 1):
    #First layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), data_format=data_format, padding = 'same',
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer = regularizers.l2(0.0001), bias_regularizer = regularizers.l2(0))(input_tensor)
    if batchnorm:
        x = BatchNormalization(epsilon = 0.00001, axis = concat_axis)(x)
    x = Activation('relu')(x)
    # Second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), data_format=data_format, padding = 'same',
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer = regularizers.l2(0.0001), bias_regularizer = regularizers.l2(0))(x)
    if batchnorm:
        x = BatchNormalization(epsilon = 0.00001, axis = concat_axis)(x)
    x = Activation('relu')(x)
    return x

def data_augmentation(input_tensor, randomflipMode="horizontal", randomzoomHeightfactor=(-0.5), randomzoomWidthfactor=(-0.5), randomzoomInterpolationMode="bilinear"):
    x = RandomFlip(mode=randomflipMode)(input_tensor)
    #x = RandomZoom(height_factor = randomzoomHeightfactor, width_factor = randomzoomWidthfactor, interpolation = randomzoomInterpolationMode)(x)
    return x


def UNET(inputDim1, inputDim2, inputDim3, nClasses, imageOrdering, batchnorm=True, enc_conv_kernel_size=3, enc_depth=2, n_filters=2*4, dropout=0.5, implicitNorm=False, inference=False, dataAugmentation=False):

    if imageOrdering == "channels_first":
        concatAxis = 1
        num_channels = inputDim1
        input_height = inputDim2
        input_width = inputDim3
        img_input = Input(shape=(num_channels, input_height, input_width))

    elif imageOrdering == "channels_last":
        concatAxis = -1
        input_height = inputDim1
        input_width = inputDim2
        num_channels = inputDim3
        img_input = Input(shape=(input_height, input_width, num_channels))
    else:
        error("Wrong image ordering")

    #Esto lo tengo que hacer porque al final del modelo tengo que especificar cuál es la entrada y como reuso el nombre de la variable img_input tengo que guardar previamente su estado
    image_input = img_input

    # Verificación de que las capas de maxPooling no van a dar error
    divisionFactor = 2**enc_depth
    assert input_height%divisionFactor == 0
    assert input_width%divisionFactor == 0


    if implicitNorm:
        img_input = BatchNormalization(epsilon = 0, axis = concatAxis, gamma_initializer = initializers.Constant(2), moving_mean_initializer = "zeros", moving_variance_initializer = initializers.Constant(np.square([4.52, 4.52, 4.52, 4.52, 4.52, 4.52, 4.52, 4.52, 4.52, 4.52, 4.52, 4.52, 4.02, 4.52, 4.52, 4.52, 4.52, 4.52, 4.52, 4.52, 4.52, 4.52, 4.52, 4.52, 4.52])), beta_initializer = initializers.Constant(-1))(img_input)

    if dataAugmentation:
        img_input = data_augmentation(img_input)

    ## Encoder
    ## Block 1:
    c1 = conv2d_block(img_input, n_filters, kernel_size = enc_conv_kernel_size, batchnorm = batchnorm, data_format = imageOrdering, concat_axis = concatAxis)
    p1 = MaxPooling2D((2, 2), data_format=imageOrdering)(c1)

    ## Block 2:
    n_filters = n_filters * 2
    c2 = conv2d_block(p1, n_filters, kernel_size = enc_conv_kernel_size, batchnorm = batchnorm, data_format = imageOrdering, concat_axis = concatAxis)
    t2 = Dropout(dropout)(c2)
    p2 = MaxPooling2D((2, 2), data_format=imageOrdering)(t2)

    ## Block 3:
    if enc_depth > 2:
        n_filters = n_filters * 2
        c3 = conv2d_block(p2, n_filters, kernel_size = enc_conv_kernel_size, batchnorm = batchnorm, data_format = imageOrdering, concat_axis = concatAxis)
        t3 = Dropout(dropout)(c3)
        p3 = MaxPooling2D((2, 2), data_format=imageOrdering)(t3)

    ## Block 4:
    if enc_depth > 3:
        n_filters = n_filters * 2
        c4 = conv2d_block(p3, n_filters, kernel_size = enc_conv_kernel_size, batchnorm = batchnorm, data_format = imageOrdering, concat_axis = concatAxis)
        t4 = Dropout(dropout)(c4)
        p4 = MaxPooling2D((2, 2), data_format=imageOrdering)(t4)

    ## Block 5:
    if enc_depth > 4:
        n_filters = n_filters * 2
        c5 = conv2d_block(p4, n_filters, kernel_size = enc_conv_kernel_size, batchnorm = batchnorm, data_format = imageOrdering, concat_axis = concatAxis)
        t5 = Dropout(dropout)(c5)
        p5 = MaxPooling2D((2, 2), data_format=imageOrdering)(t5)


    #Base de la U-Net
    if enc_depth == 5:
        entrada = p5
    elif enc_depth == 4:
        entrada = p4
    elif enc_depth == 3:
        entrada = p3
    elif enc_depth == 2:
        entrada = p2

    ## Block 5:
    n_filters = n_filters * 2
    c6 = conv2d_block(entrada, n_filters, kernel_size = 3, batchnorm = batchnorm, data_format = imageOrdering, concat_axis = concatAxis)
    c6 = Dropout(dropout)(c6)


    #Decoder
    ## Block 7:
    if enc_depth > 4:
        n_filters = n_filters / 2
        up7 = Conv2DTranspose(n_filters, kernel_size=(2,2), strides=(2,2),padding="same", data_format=imageOrdering, activation="relu",
                          kernel_initializer='glorot_uniform', bias_initializer='zeros',
                          kernel_regularizer = regularizers.l2(0.0001), bias_regularizer = regularizers.l2(0))(c6)
        #u7 = LeakyReLU(alpha=0.01)(up7)
        m7  = Concatenate(axis=concatAxis)([up7, t5])
        c7 = conv2d_block(m7, n_filters, kernel_size = 3, batchnorm = batchnorm, data_format = imageOrdering, concat_axis = concatAxis)

    ## Block 8:
    if enc_depth > 3:
        n_filters = n_filters / 2
        if enc_depth > 4:
            up8 = Conv2DTranspose(n_filters, kernel_size=(2,2), strides=(2,2),padding="same", data_format=imageOrdering, activation="relu",
                          kernel_initializer='glorot_uniform', bias_initializer='zeros',
                          kernel_regularizer = regularizers.l2(0.0001), bias_regularizer = regularizers.l2(0))(c7)
        else:
            up8 = Conv2DTranspose(n_filters, kernel_size=(2,2), strides=(2,2),padding="same", data_format=imageOrdering, activation="relu",
                          kernel_initializer='glorot_uniform', bias_initializer='zeros',
                          kernel_regularizer = regularizers.l2(0.0001), bias_regularizer = regularizers.l2(0))(c6)
        m8 = Concatenate(axis=concatAxis)([up8, t4])
        c8 = conv2d_block(m8, n_filters, kernel_size = 3, batchnorm = batchnorm, data_format = imageOrdering, concat_axis = concatAxis)

    ## Block 9:
    if enc_depth > 2:
        n_filters = n_filters / 2
        if enc_depth > 3:
            up9 = Conv2DTranspose(n_filters, kernel_size=(2,2), strides=(2,2),padding="same", data_format=imageOrdering, activation="relu",
                          kernel_initializer='glorot_uniform', bias_initializer='zeros',
                          kernel_regularizer = regularizers.l2(0.0001), bias_regularizer = regularizers.l2(0))(c8)
        else:
            up9 = Conv2DTranspose(n_filters, kernel_size=(2,2), strides=(2,2),padding="same", data_format=imageOrdering, activation="relu",
                          kernel_initializer='glorot_uniform', bias_initializer='zeros',
                          kernel_regularizer = regularizers.l2(0.0001), bias_regularizer = regularizers.l2(0))(c6)
        #u9 = LeakyReLU(alpha=0.01)(up9)
        m9 = Concatenate(axis=concatAxis)([up9, t3])
        c9 = conv2d_block(m9, n_filters, kernel_size = 3, batchnorm = batchnorm, data_format = imageOrdering, concat_axis = concatAxis)

    ## Block 10:
    ## Dependiendo de la profundidad del encoder, la entrada a la base de la U será una u otra
    if enc_depth > 2:
        entradaUp = c9
    else:
        entradaUp = c6

    n_filters = n_filters / 2
    up10 = Conv2DTranspose(n_filters, kernel_size=(2,2), strides=(2,2),padding="same", data_format=imageOrdering, activation="relu",
                          kernel_initializer='glorot_uniform', bias_initializer='zeros',
                          kernel_regularizer = regularizers.l2(0.0001), bias_regularizer = regularizers.l2(0))(entradaUp)
    #u9 = LeakyReLU(alpha=0.01)(up9)
    m10 = Concatenate(axis=concatAxis)([up10, t2])
    c10 = conv2d_block(m10, n_filters, kernel_size = 3, batchnorm = batchnorm, data_format = imageOrdering, concat_axis = concatAxis)

    ## Block 11:
    n_filters = n_filters / 2
    up11 = Conv2DTranspose(n_filters, kernel_size=(2,2), strides=(2,2),padding="same", data_format=imageOrdering, activation="relu",
                          kernel_initializer='glorot_uniform', bias_initializer='zeros',
                          kernel_regularizer = regularizers.l2(0.0001), bias_regularizer = regularizers.l2(0))(c10)
    #u9 = LeakyReLU(alpha=0.01)(up9)
    m11 = Concatenate(axis=concatAxis)([up11, c1])
    c11 = conv2d_block(m11, n_filters, kernel_size = 3, batchnorm = batchnorm, data_format = imageOrdering, concat_axis = concatAxis)


    ## Last layers:
    c12 = Conv2D(filters=nClasses, kernel_size=(1,1), data_format=imageOrdering, activation=None, padding="valid",
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer = regularizers.l2(0.0001), bias_regularizer = regularizers.l2(0))(c11)

    c13 = Softmax(axis=concatAxis)(c12)

    if inference:
        model = CustomModel(inputs=image_input, outputs=c12)
    else:
        model = CustomModel(inputs=image_input, outputs=c13)

    return model


class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # En caso de tener que hacer clipping de gradientes se puede hacer aquí siguiendo lo que se explica en https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem#:~:text=What%20is%20gradient%20clipping%3F,gradients%20to%20update%20the%20weights.

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
