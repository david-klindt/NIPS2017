## Convolutional Neural Network Model

# Imports
import numpy as np
# Tensorflow needs to be LAST import
import tensorflow as tf
from tensorflow.contrib import layers


class ModelGraph:
    def __init__(self,
                 s,#        s - sizes of [image(heigth=width), image(depth=channel), mask]
                 sK,#        s - sizes of [kernel1, kernel2, ...]
                 reg,#        regularization weights [Kernel,Mask,Weights]
                 init_scales,#        # rows: kernel,mask,weights; columns: mean, sd
                 N,#        N - number of neurons
                 num_kern,#        num_kern - number of kernels per conv layer
                 act_fn,#      activation functions for each kernel = 'ID' or 'relu'
                 kernel_constraint=None,#  constraint on read out weights - 'norm' == 1
                 weights_constraint=None,#  constraint on read out weights - 'abs','norm','absnorm'
                 mask_constraint=None,#  constraint on mask - 'abs'
                 final_relu=False,
                 batch_norm=True,#whether to use batchnorm
                 bn_cent=False,#add offset after batchnorm? default=no
                 tf_seed=None,#tensorflow random seed
                 np_seed=None,#numpy random seed
                 init_kernel=np.array([]),#       init_* - explicit initial values
                 init_weights=np.array([]),
                 init_mask=np.array([])):
        
        #set np random seed
        np.random.seed(np_seed)
        
        self.graph = tf.Graph()#new tf graph
        with self.graph.as_default():#use it as default
            
            #set random seed
            tf.set_random_seed(tf_seed)
            
            #input tensor of shape NCHW
            self.X = tf.placeholder(tf.float32,shape=[None,s[1],s[0],s[0]])
            #output: N x None
            self.Y = tf.placeholder(tf.float32)
            
            #batch normalization settings
            self.is_train = tf.placeholder(tf.bool)             
            
            #use tf.layers when kernel weights unconstrained
            if kernel_constraint==None:
                
                if batch_norm:
                    normalizer = layers.batch_norm
                    bn_params = dict(decay=.998,
                                     center=bn_cent,
                                     scale=False,
                                     is_training=self.is_train,
                                     data_format='NCHW',
                                     variables_collections=['batch_norm_ema'])
                else:
                    normalizer = None
                    bn_params = None

                # Convolutional layers
                self.conv = []# list with conv outputs
                self.WK = []# list with conv weights
                for c in range(len(num_kern)):
                    # Inputs used by layer
                    if c==0:
                        inputs = tf.layers.dropout(self.X,
                                                   rate=reg[3],
                                                   training=self.is_train)
                    else:
                        inputs = tf.layers.dropout(self.conv[c-1],
                                                   rate=reg[3],
                                                   training=self.is_train)

                    # Activation function
                    if act_fn[c] == 'ID':
                        act = tf.identity
                    elif act_fn[c] == 'relu':
                        act = tf.nn.relu
                    else:
                        raise ValueError('activation function not defined')

                    # scope of variables in this layer
                    scope = 'conv'+str(c)

                    # Layer
                    self.conv.append(layers.convolution2d(
                        inputs=inputs,
                        data_format='NCHW',
                        num_outputs=num_kern[c],
                        kernel_size=sK[c],
                        stride=1,
                        padding='VALID',
                        activation_fn=act,
                        normalizer_fn=normalizer,
                        normalizer_params=bn_params,
                        weights_initializer=tf.random_normal_initializer(mean=init_scales[0,0],
                                            stddev=init_scales[0,1]),
                        #tf.constant_initializer(init_kernel),
                        #trainable=False,
                        scope=scope))

                    #WK Kernel - filter / tensor of shape H-W-InChannels-OutChannels
                    with tf.variable_scope(scope, reuse=True):
                        self.WK.append(tf.get_variable('weights'))
                        if kernel_constraint == 'norm':
                            self.WK[-1] /= (1e-5 + tf.sqrt(tf.reduce_sum(tf.square(self.WK[-1]),
                                                                         [0,1], keep_dims=True)))
                      
            #if kernels normalized in one layer net, do manually:
            elif kernel_constraint=='norm':
                self.WK = tf.get_variable(
                                'kernels',
                                shape=[sK[0], sK[0], s[1], num_kern[0]],
                                initializer=tf.truncated_normal_initializer(
                                            mean=init_scales[0,0],
                                            stddev=init_scales[0,1]))
                               
                self.WK = [self.WK / (1e-5 + tf.sqrt(tf.reduce_sum(tf.square(self.WK), [0,1], keep_dims=True)))]
                self.conv = [tf.nn.conv2d(self.X, self.WK[-1], [1, 1, 1, 1],
                            padding='VALID',data_format='NCHW')]
                    
            # Batch_norm update op
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
            ##  Location layer - each neuron has one location mask
            #WM - Mask
            if init_mask.size:
                self.WM_init = init_mask
            else:
                self.WM_init = tf.random_normal([s[2]**2,N],init_scales[1,0],init_scales[1,1])

            
            self.WM = tf.Variable(self.WM_init,dtype=tf.float32, name='WM')
            
            if mask_constraint == 'abs':
                self.WM = tf.abs(self.WM)
                
            self.mask = tf.reshape(tf.matmul(tf.reshape(self.conv[-1],[-1,s[2]**2]),
                            self.WM),[-1,num_kern[-1],N])
                
            #Weighing Layer - again factorized per neuron
            #WW - Read Out Weights
            if init_weights.size:
                self.WW_init = init_weights
            else:
                self.WW_init = tf.random_normal([num_kern[-1],N],
                               init_scales[2,0],init_scales[2,1])
            self.WW = tf.Variable(self.WW_init,dtype=tf.float32, name='WW')

            #Apply constraint to read out weights
            if weights_constraint == 'abs':
                self.WW = tf.abs(self.WW)
            if weights_constraint == 'norm':
                self.WW /= (1e-5 + tf.sqrt(tf.reduce_sum(tf.square(self.WW),0,keep_dims=True)))
            if weights_constraint == 'absnorm':
                self.WW = tf.abs(self.WW) / (1e-5+tf.sqrt(tf.reduce_sum(tf.square(self.WW),0,keep_dims=True)))
                
            #when only one feature, skip feature weighing
            if num_kern[-1]==1:
                self.Y_ = tf.transpose(tf.squeeze(self.mask))#N x D
            else:
                #Predicted Output
                self.Y_ = tf.squeeze(tf.transpose(tf.reduce_sum(tf.multiply(self.mask,
                                self.WW), 1, keep_dims=True)))#N x D
            
            if final_relu:
                self.Y_ = tf.transpose(tf.contrib.layers.bias_add(tf.transpose(self.Y_),
                                                     activation_fn = tf.nn.relu))

            #Regularization
            self.regK = tf.contrib.layers.apply_regularization(
                            tf.contrib.layers.l2_regularizer(1e-4),
                            weights_list=self.WK)# L2 norm on conv weights
            self.regM = tf.reduce_mean(tf.reduce_sum(tf.abs(self.WM),0)) #L1 Loss on mask
            self.regW = tf.reduce_mean(tf.reduce_sum(tf.abs(self.WW),0)) #L1 Loss on read out weights

            #Define a loss function
            self.res = self.Y_-self.Y# residuals
            self.MSE = tf.reduce_mean(tf.reduce_mean(self.res * self.res,1))
            self.loss = self.MSE + reg[0]*self.regK + reg[1]*self.regM + reg[2]*self.regW
            #self.poisson = tf.reduce_sum(tf.nn.log_poisson_loss(tf.log(self.Y_),self.Y))

            #Define a training graph
            self.step_size= tf.placeholder(tf.float32)
            self.training = tf.train.AdamOptimizer(self.step_size).minimize(self.loss)

            # Create a saver.
            self.saver = tf.train.Saver()