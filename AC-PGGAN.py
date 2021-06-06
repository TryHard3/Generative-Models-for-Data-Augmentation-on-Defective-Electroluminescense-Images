import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend
from keras.layers import Embedding
from keras.layers import Concatenate

import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot
from math import sqrt
from PIL import Image
import os

from tensorflow.keras import backend

import cv2
from numpy import load
import os
import pickle as pkl
import numpy as np
from PIL import Image
import PIL
import numpy as np
import h5py
import scipy.io
import json
import PIL
import matplotlib.image as mpimg
from os import walk, getcwd
import glob
from functools import partial

import matplotlib.pyplot as plt
from matplotlib import image
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
from keras.initializers import RandomNormal
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D, Concatenate
from keras.layers import merge
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from PIL import Image
from tensorflow.python.framework.ops import disable_eager_execution
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# split a dataset into train and test sets
#from sklearn.datasets import make_blobs
#from sklearn.model_selection import train_test_split

def image_to_feature_vector(image, size=(256, 256)):
    return cv2.resize(image, size).flatten()

FILTERS = [512, 512, 512, 512, 256, 128, 64]

# Normalizes the feature vector for the pixel(axis=-1)
class PixelNormalization(Layer):
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        mean_square = tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
        l2 = tf.math.rsqrt(mean_square + 1.0e-8)
        normalized = inputs * l2
        return normalized

    def compute_output_shape(self, input_shape):
        return input_shape

# Calculate the average standard deviation of all features and spatial location.
# Concat after creating a constant feature map with the average standard deviation
class MinibatchStdev(Layer):
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)
    
    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(inputs - mean), axis=0, keepdims=True) + 1e-8)
        average_stddev = tf.reduce_mean(stddev, keepdims=True)
        shape = tf.shape(inputs)
        minibatch_stddev = tf.tile(average_stddev, (shape[0], shape[1], shape[2], 1))
        combined = tf.concat([inputs, minibatch_stddev], axis=-1)
        
        return combined
    
    def compute_output_shape(self, input_shape):
        input_shape = input_shape
        input_shape[-1] += 1
        return tuple(input_shape)

# Perform Weighted Sum
# Define alpha as backend.variable to update during training
class WeightedSum(Add):
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha, name='ws_alpha')
    
    def _merge_function(self, inputs):
        assert (len(inputs) == 2)
        output = ((1.0 - self.alpha) * inputs[0] + (self.alpha * inputs[1]))
        return output

# Scale by the number of input parameters to be similar dynamic range  
# For details, refer to https://prateekvishnu.medium.com/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528
# stddev = sqrt(2 / fan_in)
class WeightScaling(Layer):
    def __init__(self, shape, gain = np.sqrt(2), **kwargs):
        super(WeightScaling, self).__init__(**kwargs)
        shape = np.asarray(shape)
        shape = tf.constant(shape, dtype=tf.float32)
        fan_in = tf.math.reduce_prod(shape)
        self.wscale = gain*tf.math.rsqrt(fan_in)
      
    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.float32)
        return inputs * self.wscale
    
    def compute_output_shape(self, input_shape):
        return input_shape

class Bias(Layer):
    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value = b_init(shape=(input_shape[-1],), dtype='float32'), trainable=True)  

    def call(self, inputs, **kwargs):
        return inputs + self.bias
    
    def compute_output_shape(self, input_shape):
        return input_shape  

def WeightScalingDense(x, filters, gain, use_pixelnorm=False, activate=None):
    init = RandomNormal(mean=0., stddev=1.)
    in_filters = backend.int_shape(x)[-1]
    x = layers.Dense(filters, use_bias=False, kernel_initializer=init, dtype='float32')(x)
    x = WeightScaling(shape=(in_filters), gain=gain)(x)
    x = Bias(input_shape=x.shape)(x)
    if activate=='LeakyReLU':
        x = layers.LeakyReLU(0.2)(x)
    elif activate=='tanh':
        x = layers.Activation('tanh')(x)
    elif activate=='sigmoid':
        x = layers.Activation('sigmoid')(x)
    
    if use_pixelnorm:
        x = PixelNormalization()(x)
    return x

def WeightScalingConv(x, filters, kernel_size, gain, use_pixelnorm=False, activate=None, strides=(1,1)):
    init = RandomNormal(mean=0., stddev=1.)
    in_filters = backend.int_shape(x)[-1]
    x = layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False, padding="same", kernel_initializer=init, dtype='float32')(x)
    x = WeightScaling(shape=(kernel_size[0], kernel_size[1], in_filters), gain=gain)(x)
    x = Bias(input_shape=x.shape)(x)
    if activate=='LeakyReLU':
        x = layers.LeakyReLU(0.2)(x)
    elif activate=='tanh':
        x = layers.Activation('tanh')(x)
    
    if use_pixelnorm:
        x = PixelNormalization()(x)
    return x 

# https://keras.io/examples/generative/wgan_gp/
class PGAN(Model):
    def __init__(
        self,
        latent_dim,
        d_steps=1,
        gp_weight=10.0,
        drift_weight=0.001,
    ):
        super(PGAN, self).__init__()
        self.latent_dim = latent_dim
        self.d_steps = d_steps
        self.gp_weight = gp_weight
        self.drift_weight = drift_weight
        self.n_depth = 0
        self.discriminator = self.init_discriminator()
        self.discriminator_wt_fade = None
        self.generator = self.init_generator()
        self.generator_wt_fade = None

    def call(self, inputs):
        return

    def init_discriminator(self):
        img_input = layers.Input(shape = (4,4,3))
        
        img_input = tf.cast(img_input, tf.float32)
        
        # fromRGB
        x = WeightScalingConv(img_input, filters=FILTERS[0], kernel_size=(1,1), gain=np.sqrt(2), activate='LeakyReLU')
        
        # Add Minibatch end of discriminator
        x = MinibatchStdev()(x)

        x = WeightScalingConv(x, filters=FILTERS[0], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU')
        x = WeightScalingConv(x, filters=FILTERS[0], kernel_size=(4,4), gain=np.sqrt(2), activate='LeakyReLU', strides=(4,4))

        x = layers.Flatten()(x)
        # Gain should be 1, cos it's a last layer 
        out1 = WeightScalingDense(x, filters=1, gain=1.)

        #AC
        out2 = WeightScalingDense(x, filters=1, gain=1, activate='sigmoid')
        d_model = Model(img_input, [out1, out2], name='discriminator')

        return d_model

    # Fade in upper resolution block
    def fade_in_discriminator(self):
        #for layer in self.discriminator.layers:
        #    layer.trainable = False
        input_shape = self.discriminator.input.shape
        # 1. Double the input resolution. 
        input_shape = (input_shape[1]*2, input_shape[2]*2, input_shape[3])
        img_input = layers.Input(shape = input_shape)
        img_input = tf.cast(img_input, tf.float32)

        # 2. Add pooling layer 
        #    Reuse the existing “formRGB” block defined as “x1".
        x1 = layers.AveragePooling2D()(img_input)
        x1 = self.discriminator.layers[1](x1) # Conv2D FromRGB
        x1 = self.discriminator.layers[2](x1) # WeightScalingLayer
        x1 = self.discriminator.layers[3](x1) # Bias
        x1 = self.discriminator.layers[4](x1) # LeakyReLU

        # 3.  Define a "fade in" block (x2) with a new "fromRGB" and two 3x3 convolutions. 
        #     Add an AveragePooling2D layer
        x2 = WeightScalingConv(img_input, filters=FILTERS[self.n_depth], kernel_size=(1,1), gain=np.sqrt(2), activate='LeakyReLU')

        x2 = WeightScalingConv(x2, filters=FILTERS[self.n_depth], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU')
        x2 = WeightScalingConv(x2, filters=FILTERS[self.n_depth-1], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU')

        x2 = layers.AveragePooling2D()(x2)

        # 4. Weighted Sum x1 and x2 to smoothly put the "fade in" block. 
        x = WeightedSum()([x1, x2])

        # Define stabilized(c. state) discriminator 
        #original
        #for i in range(5, len(self.discriminator.layers)):
            #x2 = self.discriminator.layers[i](x2)
               
        #Tager fra 5 til len(self.discriminator.layers) - 1
        #self made
        
        #for layer in range(5, len(self.discriminator.layers)):
            #print(self.discriminator.layers[layer])
        
        #print('Suspect 1:')
        for i in range(5, len(self.discriminator.layers)):
            #Hvis det er det sidste layer, så skal den køre videre på det flatten-layer som er -4 (eller -5?) tilbage
            
            #Skræddersyet til nætværket, kan ikke håndtere at der ændre i netværket
            if (i == len(self.discriminator.layers) - 8):
                flatten = self.discriminator.layers[i](x2)
                print(self.discriminator.layers[i])
                continue
            elif (i == len(self.discriminator.layers) - 7):
                x1 = self.discriminator.layers[i](flatten)
                continue
            elif (i == len(self.discriminator.layers) - 6):
                x2 = self.discriminator.layers[i](flatten)
                continue
            elif (i == len(self.discriminator.layers) - 5):
                x1 = self.discriminator.layers[i](x1)
                continue
            elif (i == len(self.discriminator.layers) - 4):
                x2 = self.discriminator.layers[i](x2)
                continue
            elif (i == len(self.discriminator.layers) - 3):
                x1 = self.discriminator.layers[i](x1)
                continue
            elif (i == len(self.discriminator.layers) - 2):
                x2 = self.discriminator.layers[i](x2)
                continue
            elif (i == len(self.discriminator.layers) - 1):
                x1 = self.discriminator.layers[i](x1)
                continue
            
            x2 = self.discriminator.layers[i](x2)
            
        self.discriminator_stabilize = Model(img_input, [x2, x1], name='discriminator')

        #print('Suspect 1:')
        # 5. Add existing discriminator layers. 
        #Skræddersyet til nætværket, kan ikke håndtere at der ændre i netværket
        for i in range(5, len(self.discriminator.layers)):
            
            if (i == len(self.discriminator.layers) - 8):
                flatten = self.discriminator.layers[i](x)
                continue
            elif (i == len(self.discriminator.layers) - 7):
                x1 = self.discriminator.layers[i](flatten)
                continue
            elif (i == len(self.discriminator.layers) - 6):
                x = self.discriminator.layers[i](flatten)
                continue
            elif (i == len(self.discriminator.layers) - 5):
                x1 = self.discriminator.layers[i](x1)
                continue
            elif (i == len(self.discriminator.layers) - 4):
                x = self.discriminator.layers[i](x)
                continue
            elif (i == len(self.discriminator.layers) - 3):
                x1 = self.discriminator.layers[i](x1)
                continue
            elif (i == len(self.discriminator.layers) - 2):
                x = self.discriminator.layers[i](x)
                continue
            elif (i == len(self.discriminator.layers) - 1):
                x1 = self.discriminator.layers[i](x1)
                continue
            
            x = self.discriminator.layers[i](x)
        
        self.discriminator = Model(img_input, [x, x1], name='discriminator')

        self.discriminator.summary()

    # Change to stabilized(c. state) discriminator 
    def stabilize_discriminator(self):
        self.discriminator = self.discriminator_stabilize
        self.discriminator.summary()


    def init_generator(self, n_classes=2):
        #######Label-Input Block:
        in_label = layers.Input(shape=(1,))
        
        #Outputter 3, 50
        in_label1 = Embedding(n_classes, 50)(in_label)
        in_filters = backend.int_shape(in_label1)[-1]
        x = WeightScaling(shape=(in_filters), gain=np.sqrt(2)/4)(in_label1)
        x = Bias(input_shape=x.shape)(x)
        
        n_nodes = 4*4
        #WeightsScalingDense
        #label_activation = layers.LeakyReLU(0.2)(in_label1)
        #in_label2 = Dense(n_nodes)(in_label1)
        in_label2 = WeightScalingDense(x, filters=n_nodes, gain=np.sqrt(2)/4, use_pixelnorm=True)
        in_label3 = Flatten()(in_label2)
        #Reshape
        in_label4 = layers.Reshape((4, 4, 1))(in_label3)
        #######
        
        ######## Input Block:
        noise = layers.Input(shape=(self.latent_dim,))
        input_dim = PixelNormalization()(noise)
        # Actual size(After doing reshape) is just FILTERS[0], so divide gain by 4
        #use_pixelnorm=True
        input_dim = WeightScalingDense(input_dim, filters=4*4*FILTERS[0], gain=np.sqrt(2)/4, activate='LeakyReLU', use_pixelnorm=True)
        input_dim = layers.Reshape((4, 4, FILTERS[0]))(input_dim)
        #######
        
        #Concatenate filter of 1024? or 768? Or maybe just 513 (4,4,512 and 4,4,1) or even 516 (4,4,512 and 4,4,4)
        
        #Outputter (4,4,513)
        merge = Concatenate()([input_dim, in_label4])
        
        #use_pixelnorm=True
        x = WeightScalingConv(merge, filters=FILTERS[0], kernel_size=(4,4), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True)
        #use_pixelnorm=True
        x = WeightScalingConv(x, filters=FILTERS[0], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True)
        
        # Add "toRGB", the original paper uses linear as actiavation. 
        # Gain should be 1, cos it's a last layer 
        x = WeightScalingConv(x, filters=3, kernel_size=(1,1), gain=1., activate='tanh', use_pixelnorm=False)

        g_model = Model([noise, in_label], x, name='generator')
        g_model.summary()
        return g_model

    # Fade in upper resolution block
    def fade_in_generator(self):
        #for layer in self.generator.layers:
        #    layer.trainable = False
        # 1. Get the node above the “toRGB” block 
        block_end = self.generator.layers[-5].output
        # 2. Double block_end       
        block_end = layers.UpSampling2D((2,2))(block_end)

        # 3. Reuse the existing “toRGB” block defined as“x1”. 
        x1 = self.generator.layers[-4](block_end) # Conv2d
        x1 = self.generator.layers[-3](x1) # WeightScalingLayer
        x1 = self.generator.layers[-2](x1) # Bias
        x1 = self.generator.layers[-1](x1) #tanh

        # 4. Define a "fade in" block (x2) with two 3x3 convolutions and a new "toRGB".

        #use_pixelnorm=False
        x2 = WeightScalingConv(block_end, filters=FILTERS[self.n_depth], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True)
        #use_pixelnorm=False
        x2 = WeightScalingConv(x2, filters=FILTERS[self.n_depth], kernel_size=(3,3), gain=np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True)
        
        x2 = WeightScalingConv(x2, filters=3, kernel_size=(1,1), gain=1., activate='tanh', use_pixelnorm=False)

        # Define stabilized(c. state) generator
        self.generator_stabilize = Model(self.generator.input, x2, name='generator')

        # 5.Then "WeightedSum" x1 and x2 to smoothly put the "fade in" block.
        x = WeightedSum()([x1, x2])
        self.generator = Model(self.generator.input, x, name='generator')

        self.generator.summary()



    # Change to stabilized(c. state) generator 
    def stabilize_generator(self):
        self.generator = self.generator_stabilize

        self.generator.summary()


    def compile(self, d_optimizer, g_optimizer):
        super(PGAN, self).compile(run_eagerly=True)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    #Changed the input
    def train_step(self, data):
        if len(data) == 3:
            real_images, labels, sample_weight = data
        else:
            sample_weight = None
            real_images, labels = data
        
        batch_size = real_images.get_shape()[0]
        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        
        #Train the discriminator a extra amount of times (Property of WGAN-GP)
        
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.convert_to_tensor(tf.random.normal(shape=(batch_size, self.latent_dim)))
            random_fake_labels = np.random.randint(2, size=(batch_size, 1))
            
            #THis is how we log all the computations as we are going thorugh, and allows us to take the deriviative.
            
            #Categorical corss-entropy
            
            #Wasserstein and gradient penalty
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector and the label
                fake_images = self.generator([random_latent_vectors, random_fake_labels], training=True)
                
                # Get the logits for the fake images aswell as a prediction for what kind of fail there is
                fake_logits, fake_Label_pred = self.discriminator(fake_images, training=True)
                
                # Get the logits for the real images aswell as a prediction for what kind of fail there is on the image.
                real_logits, real_Label_pred = self.discriminator(real_images, training=True)

                cce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                fake_Label_loss = cce(random_fake_labels, fake_Label_pred)
                real_Label_loss = cce(labels, real_Label_pred)

                label_loss = (tf.reduce_mean(fake_Label_loss) + tf.reduce_mean(real_Label_loss)) / 2
                
                # Calculate the discriminator loss using the fake and real image logits
                #Wasserstein
                d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)

                # Calculate the drift for regularization
                drift = tf.reduce_mean(tf.square(real_logits))

                # Add the gradient penalty to the original discriminator loss aswell as the loss for the labels.
                #Added
                
                d_loss = d_cost + self.gp_weight * gp + self.drift_weight * drift + label_loss
                
            
            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_fake_labels = np.random.randint(2, size=(batch_size, 1))

        #Generatore loss
        with tf.GradientTape() as tape:
            # Generate fake images using the generator and the associated label
            generated_images = self.generator([random_latent_vectors, random_fake_labels], training=True)
                                                                   
            # Get the discriminator logits for fake images aswell as a prediction to what kind of failure
            #Added gen_label_pred
            gen_img_logits, gen_Label_pred = self.discriminator(generated_images, training=True)
            cce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            
            fake_Label_loss = cce(random_fake_labels, gen_Label_pred)
            
            # Calculate the generator loss
            g_loss = -tf.reduce_mean(gen_img_logits) + tf.reduce_mean(fake_Label_loss)
            
            #Calculate the loss of the predicted label and the actual label, using the sparse crossentropy
            #Added
            
        # Get the gradients w.r.t the generator loss
        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))
        #return {'d_label_loss1': d_loss1, 'd_label_loss2': d_loss2, 'd_normal_loss': d_loss3, 'g_label_loss1': fake_Label_loss, 'g_normal_loss2': g_loss}
        return {'d_loss': d_loss, 'd_label_loss': label_loss, 'g_loss': g_loss, 'g_label_loss': fake_Label_loss}

# Create a Keras callback that periodically saves generated images and updates alpha in WeightedSum layers
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=16, latent_dim=100, prefix=''):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.random_latent_vectors = tf.random.normal(shape=[num_img, self.latent_dim], seed=9434)
        self.random_labels = np.random.randint(2, size=(num_img, 1))
        self.steps_per_epoch = 0
        self.epochs = 0
        self.steps = self.steps_per_epoch * self.epochs
        self.n_epoch = 0
        self.prefix = prefix

    def set_prefix(self, prefix=''):
        self.prefix = prefix
  
    def set_steps(self, steps_per_epoch, epochs):
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.steps = self.steps_per_epoch * self.epochs

    def on_epoch_begin(self, epoch, logs=None):
        self.n_epoch = epoch


    def on_epoch_end(self, epoch, logs=None):
        #Få Labels til at passe ind.
        
        samples = self.model.generator([self.random_latent_vectors, self.random_labels])
        samples = (samples * 0.5) + 0.5
        n_grid = int(sqrt(self.num_img))

        fig, axes = pyplot.subplots(n_grid, n_grid, figsize=(4*n_grid, 4*n_grid))
        sample_grid = np.reshape(samples[:n_grid * n_grid], (n_grid, n_grid, samples.shape[1], samples.shape[2], samples.shape[3]))
        
        for i in range(n_grid):
            for j in range(n_grid):
                axes[i][j].set_axis_off()
                samples_grid_i_j = Image.fromarray((sample_grid[i][j] * 255).astype(np.uint8))
                samples_grid_i_j = samples_grid_i_j.resize((128,128))
                axes[i][j].imshow(np.array(samples_grid_i_j))
        title = '/user/student.aau.dk/lharde18/Data-output/cbk/Final15/plot_{%s}_{%s05d}.png' % (self.prefix, self.epochs)
        pyplot.savefig(title, bbox_inches='tight')
        print(f'\n saved {title}')
        pyplot.close(fig)
  

    def on_batch_begin(self, batch, logs=None):
        # Update alpha in WeightedSum layers
        alpha = ((self.n_epoch * self.steps_per_epoch) + batch) / float(self.steps - 1)
        #print(f'\n {self.steps}, {self.n_epoch}, {self.steps_per_epoch}, {alpha}')
        for layer in self.model.generator.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)
        for layer in self.model.discriminator.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)

def scale_dataset(images, new_shape):
    images_list = []
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = cv2.resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)


# Normalilze [-1, 1] input images
def preprocessing_image(img):
    img = img.astype('float32')
    img = (img - 127.5) / 127.5
    return img

labels = load('/user/student.aau.dk/lharde18/Data/labeltest.npy')
x_train1 = load('/user/student.aau.dk/lharde18/Data/datatest.npy')
print(labels.shape)
print(x_train1.shape)

print('kuk')
#DEFINE FILEPATH AND PARAMETERS
# can use celeb A mask dataset on https://github.com/switchablenorms/CelebAMask-HQ 
DATA_ROOT = '/user/student.aau.dk/lharde18/Data-output'
NOISE_DIM = 100
# Set the number of batches, epochs and steps for trainining.
# Look 800k images(16x50x1000) per each lavel
BATCH_SIZE = [16, 16, 16, 16, 16, 16, 4]
EPOCHS = 50
STEPS_PER_EPOCH = len(x_train1)//BATCH_SIZE[0]

#Ændre det så det scaler dataset ned, og bruger det fra memory.
#4*(2**n_depth), 4*(2**n_depth)

train_data = scale_dataset(x_train1, (4*(2**0), 4*(2**0)))
print('Scaled the data')

# Instantiate the optimizer for both networks
# learning_rate will be equalized per each layers by the WeightScaling scheme
generator_optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)



cbk = GANMonitor(num_img=64, latent_dim=NOISE_DIM, prefix='0_init')
cbk.set_steps(steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS)

# Instantiate the PGAN(PG-GAN) model.
pgan = PGAN(
        latent_dim = NOISE_DIM, 
        d_steps = 1,
)

checkpoint_path = f"ckpts/pgan_{cbk.prefix}.ckpt"

# Compile models
pgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
)

# Draw models
#tf.keras.utils.plot_model(pgan.generator, to_file=f'generator_{pgan.n_depth}.png', show_shapes=True)
#tf.keras.utils.plot_model(pgan.discriminator, to_file=f'discriminator_{pgan.n_depth}.png', show_shapes=True)

# Start training the initial generator and discriminator
print('Training')

pgan.fit(train_data, labels, steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, callbacks=[cbk])
pgan.save_weights(checkpoint_path)

# Train faded-in / stabilized generators and discriminators

for n_depth in range(1, 7):
    # Set current level(depth)
    pgan.n_depth = n_depth

    # Set parameters like epochs, steps, batch size and image size
    STEPS_PER_EPOCH = len(x_train1)//BATCH_SIZE[n_depth]
    steps_per_epoch = STEPS_PER_EPOCH
    epochs = int(EPOCHS*(BATCH_SIZE[0]/BATCH_SIZE[n_depth]))
        
    train_data = scale_dataset(x_train1, (4*(2**n_depth), 4*(2**n_depth)))
    
    cbk.set_prefix(prefix=f'{n_depth}_fade_in')
    cbk.set_steps(steps_per_epoch=steps_per_epoch, epochs=epochs)

    # Put fade in generator and discriminator
    pgan.fade_in_generator()
    pgan.fade_in_discriminator()

    # Draw fade in generator and discriminator
    #tf.keras.utils.plot_model(pgan.generator, to_file=f'generator_{n_depth}_fade_in.png', show_shapes=True)
    #tf.keras.utils.plot_model(pgan.discriminator, to_file=f'discriminator_{n_depth}_fade_in.png', show_shapes=True)

    pgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
    )
    # Train fade in generator and discriminator
    pgan.fit(train_data, labels, steps_per_epoch = steps_per_epoch, epochs = epochs, callbacks=[cbk])

    # Save models
    checkpoint_path = f"ckpts/pgan_{cbk.prefix}.ckpt"
    pgan.save_weights(checkpoint_path)

    # Change to stabilized generator and discriminator
    cbk.set_prefix(prefix=f'{n_depth}_stabilize')
    pgan.stabilize_generator()
    pgan.stabilize_discriminator()

    # Draw stabilized generator and discriminator
    #tf.keras.utils.plot_model(pgan.generator, to_file=f'generator_{n_depth}_stabilize.png', show_shapes=True)
    #tf.keras.utils.plot_model(pgan.discriminator, to_file=f'discriminator_{n_depth}_stabilize.png', show_shapes=True)
    pgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
    )
    
    # Train stabilized generator and discriminator
    pgan.fit(train_data, labels, steps_per_epoch = steps_per_epoch, epochs = epochs, callbacks=[cbk])
        
    # Save models
    checkpoint_path = f"ckpts/pgan_{cbk.prefix}.ckpt"
    pgan.save_weights(checkpoint_path)

pgangen = pgan.generator
pdisc = pgan.discriminator
pgangen.save('/user/student.aau.dk/lharde18/Data-output/pgangenFinal15')
pdisc.save('/user/student.aau.dk/lharde18/Data-output/pdiscFinal15')            

