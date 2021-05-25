import sys
import h5py

from h5py import File as HDF5File
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Input, Dense, Reshape, Flatten, Lambda, ##merge, but not used
                          Dropout, BatchNormalization, Activation, Embedding)
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.layers import (UpSampling3D, Conv3D, ZeroPadding3D,
                                        AveragePooling3D)

from tensorflow.keras.models import Model, Sequential


def ecal_sum(image):
    sum = K.sum(image, axis=(1, 2, 3))
    return sum
   

def discriminator(keras_dformat='channels_last'):

#    if keras_dformat =='channels_last':
#        dshape=(25, 25, 25,1)
#        daxis=(1,2,3)
#    else:
#        dshape=(1, 25, 25, 25)
#        daxis=(2,3,4)
#
#    image = Input(shape=dshape)
#
#    x = Conv3D(32, (5, 5,5), data_format=keras_dformat, padding='same')(image)
#    x = LeakyReLU()(x)
#    x = Dropout(0.2)(x)
#
#    x = ZeroPadding3D((2, 2,2))(x)
#    x = Conv3D(8, 5, 5, 5, data_format=keras_dformat,  padding='valid')(x)
#    x = LeakyReLU()(x)
#    x = BatchNormalization()(x)
#    x = Dropout(0.2)(x)
#
#    x = ZeroPadding3D((2, 2, 2))(x)
#    x = Conv3D(8, 5, 5,5, data_format=keras_dformat, padding='valid')(x)
#    x = LeakyReLU()(x)
#    x = BatchNormalization()(x)
#    x = Dropout(0.2)(x)
#
#    x = ZeroPadding3D((1, 1, 1))(x)
#    x = Conv3D(8, 5, 5, 5, data_format=keras_dformat, padding='valid')(x)
#    x = LeakyReLU()(x)
#    x = BatchNormalization()(x)
#    x = Dropout(0.2)(x)
#
#    x = AveragePooling3D((2, 2, 2))(x)
#    h = Flatten()(x)
#
#    dnn = Model(image, h)
#
#    dnn_out = dnn(image)
#
#
#    fake = Dense(1, activation='sigmoid', name='generation')(dnn_out)
#    aux = Dense(1, activation='linear', name='auxiliary')(dnn_out)
#    ecal = Lambda(lambda x: K.sum(x, axis=daxis))(image)
#    Model(input=image, output=[fake, aux, ecal]).summary()
#    return Model(input=image, output=[fake, aux, ecal])
    
    
    
    if keras_dformat =='channels_last':
        dshape=(25, 25, 25,1)
        daxis=(1,2,3)
        axis = -1 
    else:
        dshape=(1, 25, 25, 25)
        daxis=(2,3,4)
        axis = 1 
    #keras_dformat='channels_first'   #i need this when I train gen with ch last and keras with ch first
    #dshape=(25, 25, 25, 1)    
    image = Input(shape=dshape, dtype="float32")     #Input Image
    x = image
    
    x = Conv3D(32, (5,5,5), data_format=keras_dformat, use_bias=False, padding='same')(x)
    x = LeakyReLU() (x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((2, 2, 2))(x)
    x = Conv3D(8, (5,5,5), data_format=keras_dformat, use_bias=False, padding='valid')(x)
    x = LeakyReLU() (x)
    x = BatchNormalization(axis=axis)(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((2, 2, 2))(x)
    x = Conv3D(8, (5,5,5), data_format=keras_dformat, use_bias=False, padding='valid')(x)
    x = LeakyReLU() (x)
    x = BatchNormalization(axis=axis)(x)
    x = Dropout(0.2)(x)

    x = ZeroPadding3D((1, 1, 1))(x)
    x = Conv3D(8, (5,5,5), data_format=keras_dformat, use_bias=False, padding='valid')(x)
    x = LeakyReLU() (x)
    x = BatchNormalization(axis=axis)(x)
    x = Dropout(0.2)(x)

    x = AveragePooling3D((2, 2, 2))(x)
    x = Flatten()(x)

    #Takes Network outputs as input
    fake = Dense(1, activation='sigmoid', name='generation')(x)   #Klassifikator true/fake
    aux = Dense(1, activation='linear', name='auxiliary')(x)       #Soll sich an E_in (Ep) annähern
    #Takes image as input
    ecal = Lambda(lambda x: tf.math.reduce_sum(x, axis=daxis))(image)    #Energie, die im Netzwerk steckt, Summe über gesamtes NEtzwerk
    return Model(inputs=image, outputs=[fake, aux, ecal])


def generator(latent_size=200, return_intermediate=False, keras_dformat='channels_last'):

#     if keras_dformat =='channels_last':
#        dim = (7,7,8,8)
#     else:
#        dim = (8, 7, 7,8)
#
#     loc = Sequential([
#         Dense(64 * 7* 7, input_dim=latent_size),
#         Reshape(dim),
#         Conv3D(64, (6, 6, 8), data_format=keras_dformat, padding='same', kernel_initializer='he_uniform'),
#         LeakyReLU(),
#         BatchNormalization(),
#         UpSampling3D(size=(2, 2, 2)),
#
#         ZeroPadding3D((2, 2, 0)),
#         Conv3D(6, (6, 5, 8), data_format=keras_dformat, kernel_initializer='he_uniform'),
#         LeakyReLU(),
#         BatchNormalization(),
#         UpSampling3D(size=(2, 2, 3)),
#
#         ZeroPadding3D((1,0,3)),
#         Conv3D(6, (3, 3, 8), data_format=keras_dformat, kernel_initializer='he_uniform'),
#         LeakyReLU(),
#         Conv3D(1, (2, 2, 2), data_format=keras_dformat, use_bias=False, kernel_initializer='glorot_normal'),
#         Activation('relu')
#
#     ])
#
#     latent = Input(shape=(latent_size, ))
#
#     fake_image = loc(latent)
#
#     Model(input=[latent], output=fake_image).summary()
#     return Model(input=[latent], output=fake_image)
     
     
    if keras_dformat =='channels_last':
        dim = (7,7,8,8)
        axis = -1
    else:
        dim = (8, 7, 7,8)
        axis = 1
    
    latent = Input(shape=(latent_size))
    x = Dense(8*8*7*7)(latent)   #shape (none, 625) #none is batch size
    x = Reshape(dim) (x)
    
    x = Conv3D(64, (6,6,8), data_format=keras_dformat, padding='same', kernel_initializer='he_uniform')(x)
    x = LeakyReLU() (x)
    x = BatchNormalization(axis=axis) (x)

    x = UpSampling3D(size=(2, 2, 2), data_format=keras_dformat)(x)
    x = ZeroPadding3D((2, 2, 0))(x)
    x = Conv3D(6, (6,5,8), data_format=keras_dformat, padding='valid', kernel_initializer='he_uniform')(x)
    x = LeakyReLU() (x)
    x = BatchNormalization(axis=axis) (x)
    
    x = UpSampling3D(size=(2, 2, 3), data_format=keras_dformat)(x)
    x = ZeroPadding3D((1, 0, 3))(x)
    x = Conv3D(6, (3,3,8), data_format=keras_dformat, padding='valid', kernel_initializer='he_uniform')(x)
    x = LeakyReLU() (x)
    x = Conv3D(1, (2,2,2), data_format=keras_dformat, padding='valid', kernel_initializer='glorot_normal')(x)
    x = ReLU() (x)
    
    #Model(input=[latent], output=fake_x).summary()
    return Model(inputs=[latent], outputs=x)

