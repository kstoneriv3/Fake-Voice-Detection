'''Example of VAE on MNIST dataset using CNN
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon



# MNIST dataset
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#
#image_size = x_train.shape[1]
#x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
#x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
#x_train = x_train.astype('float32') / 255
#x_test = x_test.astype('float32') / 255

# network parameters
def cvae_training(x_train):
    #image_size = x_train.shape[1]
    #for x in x_train:
    #    x=np.reshape(x,[x.shape[0], x.shape[1], 1])
    x_train=K.stack(x_train)
    x_train = K.reshape(x_train, [-1, x_train.shape[1].value, x_train.shape[2].value, 1])
    input_shape = (x_train.shape[1].value, x_train.shape[2].value,1)
    batch_size = 20
    kernel_size = 3
    filters = 8
    latent_dim = 16
    epochs = 30
    
    
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    x=Conv2D(filters=filters,kernel_size=(1,input_shape[1]),strides=1,activation='tanh')(x)
    for i in range(2):
        filters *= 2
        x = Conv2D(filters=filters,
                   kernel_size=(kernel_size,1),
                   activation='tanh',
                   strides=(2,1),
                   padding='same')(x)
    
    # shape info needed to build decoder model
    shape = K.int_shape(x)
    
    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(64, activation='tanh')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)
    
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2]*shape[3], activation='tanh')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    filters//=2;
    for i in range(2):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=(kernel_size,1),
                            activation='tanh',
                            strides=(2,1),
                            padding='same')(x)
        filters //= 2
        
    outputs=Conv2DTranspose(filters=2,kernel_size=(1,input_shape[1]),strides=1,activation='tanh')(x)
    #x_log_var=Conv2DTranspose(filters=1,kernel_size=(1,input_shape[1]),strides=1,activation='tanh')(x);  
    #x_log_var=Flatten()(x_log_var)

   # x_log_var=1.0
#    outputs = Conv2DTranspose(filters=1,
#                              kernel_size=(kernel_size,1),
#                              activation='tanh',
#                              padding='same',
#                              name='decoder_output')(x)
    
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)
    
    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')
    reconstruction_loss = K.sum((K.flatten(inputs)-K.flatten(outputs[:,:,:,0]))/(K.exp(K.flatten(outputs[:,:,:,1])))+K.flatten(outputs[:,:,:,1]))
    #reconstruction_loss *= input_shape[0] * input_shape[1]
    #reconstruction_loss=K.binary_crossentropy(x, x_decoded_mean)
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()
    vae.fit(x_train,x_train,epochs=epochs,steps_per_epoch=40)
    return vae

def log_prob(model,samples):
    #samples=K.stack(samples)
    samples=K.reshape(samples,[1,samples.shape[0],samples.shape[1],1])
    #probs=np.zeros(100)
    #for i in range(100):
    #    probs[i]=-
    return -model.evaluate(samples,samples,steps=1)
            #validation_data=(x_test, None))


#if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    help_ = "Load h5 model trained weights"
#    parser.add_argument("-w", "--weights", help=help_)
#    help_ = "Use mse loss instead of binary cross entropy (default)"
#    parser.add_argument("-m", "--mse", help=help_, action='store_true')
#    args = parser.parse_args()
#    models = (encoder, decoder)
#    data = (x_test, y_test)
#
#    # VAE loss = mse_loss or xent_loss + kl_loss
#    if args.mse:
#        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
#    else:
#        reconstruction_loss = binary_crossentropy(K.flatten(inputs),
#                                                  K.flatten(outputs))
#
#    reconstruction_loss *= image_size * image_size
#    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
#    kl_loss = K.sum(kl_loss, axis=-1)
#    kl_loss *= -0.5
#    vae_loss = K.mean(reconstruction_loss + kl_loss)
#    vae.add_loss(vae_loss)
#    vae.compile(optimizer='rmsprop')
#    vae.summary()
#    plot_model(vae, to_file='vae_cnn.png', show_shapes=True)
#
#    if args.weights:
#        vae.load_weights(args.weights)
#    else:
#        # train the autoencoder
#        vae.fit(x_train,
#                epochs=epochs,
#                batch_size=batch_size,
#                validation_data=(x_test, None))
#        vae.save_weights('vae_cnn_mnist.h5')
#
#plot_results(models, data, batch_size=batch_size, model_name="vae_cnn")