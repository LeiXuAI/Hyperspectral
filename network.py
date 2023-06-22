import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from models import conv_layer
from tensorflow import einsum
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange

import data

# This is a binary band selection network for hyperspectral images
class WBBSN:

    def __init__(self, batch_size, epoch, window_size, num_band, select_num_band, num_class) -> None:
        self.batch_size = batch_size
        self.epoch = epoch
        self.window_size = window_size
        self.num_band = num_band
        self.select_num_band = select_num_band
        self.num_class = num_class
        
        tf.reset_default_graph()
        tf.set_random_seed(133)

    def _build_graph(self):
        self.X = tf.placeholder(tf.float32, shape=[self.batch_size, self.window_size, self.window_size, self.num_band], name='image')
        self.Y = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_class], name='gt')

    def net(self, input_img, is_training=True):
        # spatical attention layers
        x = tf.transpose(input_img, perm=[0, 3, 1, 2]) # batch, channel, h, w
        num_sample, num_band, h, w = x.shape
        self.spatial_conv1 = conv_layer(x, kernel_size=1, 
                            in_channel=num_band, out_channel=num_band//4, 
                            stride = 1, padding='SAME', name = None)
        self.spatial_conv2 = conv_layer(x, kernel_size=1, 
                            in_channel=num_band, out_channel=num_band//4, 
                            stride = 1, padding='SAME', name = None)
        self.spatial_conv3 = conv_layer(x, kernel_size=1, 
                            in_channel=num_band, out_channel=num_band, 
                            stride = 1, padding='SAME', name = None) 
        
        self.spatial_conv1 = rearrange(self.spatial_conv1, 'b c h w -> b c (h w)')
        self.spatial_conv2 = rearrange(self.spatial_conv2, 'b c h w -> b c (h w)') 

# This is pixel level k bands selection network based on https://github.com/mfbalin/Concrete-Autoencoders/
 class PBBSN():

