"""
by Lei
"""

import time
import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
import math
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Softmax, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant, glorot_normal
from tensorflow.keras.optimizers import Adam

def sample_gumbel_01(shape, eps=1e-10):
    """Sample from Gumbel(0, 1) distribution"""
    U = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)
    return -tf.math.log(-tf.math.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """Draw a sample from the Gumbel-Softmax distribution"""
    # logits: [batch_size, n_classes], unnormalized log-probs
    y = logits + sample_gumbel_01(tf.shape(logits))
    return tf.nn.softmax(y / temperature, axis=-1) # sum of each line equals 1

def gumbel_softmax(logits, temperature, hard=False):
    """
    logits: [batch_size, n_classes], unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
      y_hard = tf.one_hot(tf.math.argmax(y, axis=-1), depth=y.shape[1], dtype=y.dtype)
      # ref: https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html#torch.nn.functional.gumbel_softmax
      # https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f?permalink_comment_id=3037101#gistcomment-3037101
      # use stop_gradient trick to forward the gradient w.r.t. y_hard to y
      return tf.stop_gradient(y_hard - y) + y
    else:
      return y

# This is from https://github.com/mfbalin/Concrete-Autoencoders
class ConcreteSelect(Layer):
    
    def __init__(self, output_dim, start_temp = 10.0, min_temp = 0.1, alpha = 0.99999, **kwargs):
        self.output_dim = output_dim
        self.start_temp = start_temp
        self.min_temp = K.constant(min_temp)
        self.alpha = K.constant(alpha)
        super(ConcreteSelect, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.temp = self.add_weight(name = 'temp', shape = [], initializer = Constant(self.start_temp), trainable = False)
        self.logits = self.add_weight(name = 'logits', shape = [self.output_dim, input_shape[1]], initializer = glorot_normal(), trainable = True)
        super(ConcreteSelect, self).build(input_shape)
        
    def call(self, X, training = None):
        uniform = K.random_uniform(self.logits.shape, K.epsilon(), 1.0)
        gumbel = -K.log(-K.log(uniform))
        temp = K.update(self.temp, K.maximum(self.min_temp, self.temp * self.alpha))
        noisy_logits = (self.logits + gumbel) / temp
        samples = K.softmax(noisy_logits)
        
        discrete_logits = K.one_hot(K.argmax(self.logits), self.logits.shape[1])
        
        self.selections = K.in_train_phase(samples, discrete_logits, training)
        Y = K.dot(X, K.transpose(self.selections))
        
        return Y
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)



def conv_layer(tensor, kernel_size, in_channel, out_channel, stride = 1, padding='SAME', name = None):
    ''':
    2D Convolutional layer in TF
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable(
            "weights",
            shape=[kernel_size, kernel_size, in_channel, out_channel],
            initializer=tf.random_normal_initializer(stddev=0.02)
        )
        bias = tf.get_variable(
            "bias",
            shape=[out_channel],
            initializer=tf.constant_initializer(0.0)
        )
        conv = tf.nn.conv2d(
            tensor,
            weights,
            strides=[1, stride, stride, 1],
            padding=padding
        )
        conv = tf.nn.bias_add(conv, bias)
        return conv

def deconv_layer(tensor, kernel_size, in_channel, out_channel, stride, padding='SAME', name = None):
    '''
    2D deconvolutional layer in TF
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable(
            'weights',
            shape = [kernel_size, kernel_size, out_channel, in_channel],
            initializer = tf.random_normal_initializer(stddev=0.02),
        )

        bias = tf.get_variable(
            'bias',
            shape = [out_channel],
            initializer = tf.constant_initializer(0.0)
        )

        deconv = tf.nn.conv2d_transpose(
            tensor, 
            weights,
            output_shape = [out_channel],
            strides = [1, stride, stride, 1],
            padding = padding    
            )

        deconv = tf.nn.bias_add(deconv, bias)
        return deconv
    
def fc_layer(tensor, in_dims, out_dims, name):
    '''
    Fully-Connected layer in TF
    '''
    tensor = tf.reshape(tensor, shape = [-1, tensor.get_shape().as_list()[-1]])
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE) as scope:
        weights = tf.get_variable(
            "weights",
            shape=[in_dims, out_dims],
            initializer=tf.random_normal_initializer(stddev=0.02)
        )
        bias = tf.get_variable(
            "bias",
            shape=[out_dims],
            initializer=tf.constant_initializer(0.0)
        )
        fc = tf.nn.bias_add(tf.matmul(tensor, weights), bias)
        return fc

def bn(tensor, is_training, name):
    '''
    Batch Normalization in TF
    Batch Normalization is not apllied is the model is training
    '''
    return tf.layers.batch_normalization(tensor, training=is_training, name=name)
 
if __name__ == "__main__":
    t = 0.5
    p = [0.1, 0.5, 0.4]
    temperature = 10
    logits = [[-2, 2, 0], [3, 1, 3]]
    import pdb; pdb.set_trace()
    gumbel_softmax(logits, t, True) 
   