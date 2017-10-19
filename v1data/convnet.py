import numpy as np
import os
from scipy import stats, signal
import tensorflow as tf
import hashlib
import inspect
import random
from tensorflow.contrib import layers
import time


def smoothness_regularizer_2d(W, weight=1.0):
    with tf.variable_scope('smoothness'):
        lap = tf.constant([[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]])
        lap = tf.expand_dims(tf.expand_dims(lap, 2), 3)
        out_channels = W.get_shape().as_list()[2]
        W_lap = tf.nn.depthwise_conv2d(tf.transpose(W, perm=[3, 0, 1, 2]),
                                       tf.tile(lap, [1, 1, out_channels, 1]),
                                       strides=[1, 1, 1, 1], padding='SAME')
        penalty = tf.reduce_sum(tf.reduce_sum(tf.square(W_lap), [1, 2]) / tf.transpose(tf.reduce_sum(tf.square(W), [0, 1])))
        penalty = tf.identity(weight * penalty, name='penalty')
        tf.add_to_collection('smoothness_regularizer_2d', penalty)
        return penalty


def group_sparsity_regularizer_2d(W, weight=1.0):
    with tf.variable_scope('group_sparsity'):
        penalty = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(W), [0, 1])))
        penalty = tf.identity(weight * penalty, name='penalty')
        tf.add_to_collection('group_sparsity_regularizer_2d', penalty)
        return penalty


def elu(x):
    return tf.log(tf.exp(x) + 1, name='elu')
    

def inv_elu(x):
    return np.log(np.exp(x) - 1)


def poisson(prediction, response):
    return tf.reduce_mean(tf.reduce_sum(prediction - response * tf.log(prediction + 1e-5), 1), name='poisson')


def envelope(w, k=25):
    t = np.linspace(-2.5, 2.5, k, endpoint=True)
    u, v = np.meshgrid(t, t)
    win = np.exp(-(u ** 2 + v ** 2) / 2)
    sub = lambda x: x - np.median(x)
    return np.array([signal.convolve2d(sub(wi) ** 2, win, 'same') for wi in w])


def est_rf_location(x, y, k=25):
    zscore = lambda x: (x - x.mean()) / x.std()
    x = zscore(x[:,:,:,0])
    w = np.tensordot(y, x, axes=[[0], [0]])
    e = envelope(w, k)
    s = e.shape
    e = np.reshape(e, [s[0], -1])
    max_idx = np.argmax(e, axis=1)
    x = max_idx % s[2]
    y = max_idx // s[2]
    return x, y



class Net:

    def __init__(self, data=None, log_dir=None, log_hash=None, global_step=None):
        self.data = data
        log_dir_ = os.path.dirname(inspect.stack()[0][1])
        log_dir = os.path.join(log_dir_, 'train_logs', 'cnn_tmp' if log_dir is None else log_dir)
        if log_hash == None: log_hash = '%010x' % random.getrandbits(40)
        self.log_dir = os.path.join(log_dir, log_hash)
        self.log_hash = log_hash
        self.seed = int.from_bytes(log_hash[:4].encode('utf8'), 'big')
        self.global_step = 0 if global_step == None else global_step
        self.session = None
        self.best_loss = 1e100

        # placeholders
        if data is None: return
        with tf.Graph().as_default() as self.graph:
            self.is_training = tf.placeholder(tf.bool)
            self.learning_rate = tf.placeholder(tf.float32)
            self.images = tf.placeholder(tf.float32, shape=[None, data.px_y, data.px_x, 1])
            self.responses = tf.placeholder(tf.float32, shape=[None, data.num_neurons])


    def initialize(self):
        self.session = tf.Session(graph=self.graph)
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)


    def __del__(self):
        try:
            if not self.session == None:
                self.session.close()
                self.writer.close()
        except:
            pass


    def close(self):
        self.session.close()


    def save(self):
        self.saver.save(self.session, os.path.join(self.log_dir, 'model.ckpt'))


    def load(self):
        self.saver.restore(self.session, os.path.join(self.log_dir, 'model.ckpt'))


    def train(self,
              max_iter=5000,
              learning_rate=0.005,
              batch_size=256,
              val_steps=100,
              early_stopping_steps=5):
        with self.graph.as_default():
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            imgs_val, res_val = self.data.val()
            not_improved = 0
            for i in range(self.global_step + 1, self.global_step + max_iter + 1):

                # training step
                imgs_batch, res_batch = self.data.minibatch(batch_size)
                self.global_step = i
                feed_dict = {self.images: imgs_batch,
                             self.responses: res_batch,
                             self.is_training: True,
                             self.learning_rate: learning_rate}
                t = time.time()
                self.session.run([self.train_step, update_ops], feed_dict)
                # validate/save periodically
                if not i % val_steps:
                    result = self.eval(images=imgs_val, responses=res_val)
                    if result[0] < self.best_loss:
                        self.best_loss = result[0]
                        self.save()
                        not_improved = 0
                    else:
                        not_improved += 1
                    if not_improved == early_stopping_steps:
                        self.global_step -= early_stopping_steps * val_steps
                        self.load()
                        not_improved = 0
                        break
                    yield (i, result)


    def eval(self, images=None, responses=None):
        if images is None or responses is None:
            images, responses = self.data.test()
        ops = self.get_test_ops()
        feed_dict = {self.images: images,
                     self.responses: responses,
                     self.is_training: False}
        result = self.session.run(ops, feed_dict)
        return result


    def get_test_ops(self):
        return [self.poisson, self.total_loss, self.prediction]



class ConvNet(Net):

    def __init__(self, *args, **kwargs):
        super(ConvNet, self).__init__(*args, **kwargs)
        self.conv = []
        self.W = []
        self.readout_sparseness_regularizer = 0.0


    def build(self,
              filter_sizes,
              out_channels,
              strides,
              paddings,
              smooth_weights,
              sparse_weights,
              readout_sparse_weight,
              fully_connected_readout=False,
              fixed_rfs=False):

        with self.graph.as_default():
            
            # convolutional layers
            for i, (filter_size,
                    out_chans,
                    stride,
                    padding,
                    smooth_weight,
                    sparse_weight) in enumerate(zip(filter_sizes,
                                                    out_channels,
                                                    strides,
                                                    paddings,
                                                    smooth_weights,
                                                    sparse_weights)):
                x = self.images if not i else self.conv[i-1]
                bn_params = {'decay': 0.9, 'is_training': self.is_training}
                scope = 'conv{}'.format(i)
                reg = lambda w: smoothness_regularizer_2d(w, smooth_weight) + \
                                group_sparsity_regularizer_2d(w, sparse_weight)
                c = layers.convolution2d(inputs=x,
                                         num_outputs=out_chans,
                                         kernel_size=int(filter_size),
                                         stride=int(stride),
                                         padding=padding,
                                         activation_fn=elu if i < len(filter_sizes) - 1 else None,
                                         normalizer_fn=layers.batch_norm,
                                         normalizer_params=bn_params,
                                         weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                         weights_regularizer=reg,
                                         scope=scope,
                )
                with tf.variable_scope(scope, reuse=True):
                    W = tf.get_variable('weights')
                self.W.append(W)
                self.conv.append(c)

            # initialize biases
            images, responses = self.data.train()
            b = inv_elu(responses.mean(axis=0))
                
            # readout layer
            if not fully_connected_readout:
                sz = c.get_shape()
                px_x_conv = int(sz[2])
                px_y_conv = int(sz[1])
                px_conv = px_x_conv * px_y_conv
                conv_flat = tf.reshape(c, [-1, px_conv, out_channels[-1], 1])
                if not fixed_rfs:
                    self.W_spatial = tf.get_variable('W_spatial',
                                                     shape=[px_conv, self.data.num_neurons],
                                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                else:
                    # instead of learning the spatial mask, here we extract RF 
                    # locations from STA (similar to Batty et al. 2017)
                    crop = (images.shape[1] - px_x_conv) // 2
                    rf_x, rf_y = est_rf_location(images, responses, k=25)
                    rf_x = np.maximum(np.minimum(rf_x - crop, px_x_conv - 1), 0)
                    rf_y = np.maximum(np.minimum(rf_y - crop, px_y_conv - 1), 0)
                    w_init = np.zeros([self.data.num_neurons, px_x_conv, px_y_conv])
                    for i, (x, y) in enumerate(zip(rf_x, rf_y)):
                        w_init[i,y,x] = 1
                    w_init = np.reshape(w_init, [w_init.shape[0], -1]).T
                    self.W_spatial = tf.get_variable('W_spatial',
                                                     shape=[px_conv, self.data.num_neurons],
                                                     initializer=tf.constant_initializer(w_init),
                                                     trainable=False)
                    
                W_spatial_flat = tf.reshape(self.W_spatial, [px_conv, 1, 1, self.data.num_neurons])
                W_spatial_flat = tf.abs(W_spatial_flat)
                h_spatial = tf.nn.conv2d(conv_flat, W_spatial_flat, strides=[1, 1, 1, 1], padding='VALID')
                self.W_features = tf.get_variable('W_features',
                                                  shape=[out_channels[-1], self.data.num_neurons],
                                                  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                self.W_features = tf.abs(self.W_features)
                self.h_out = tf.reduce_sum(tf.multiply(h_spatial, self.W_features), [1, 2])

                # L1 regularization for readout layer
                self.readout_sparseness_regularizer = readout_sparse_weight * tf.reduce_sum(
                    tf.reduce_sum(tf.abs(self.W_spatial), 0) * \
                    tf.reduce_sum(tf.abs(self.W_features), 0)
                )
                tf.losses.add_loss(self.readout_sparseness_regularizer, tf.GraphKeys.REGULARIZATION_LOSSES)

                # output nonlinearity
                self.b_out = tf.get_variable('b_out',
                                             shape=[self.data.num_neurons],
                                             dtype=tf.float32,
                                             initializer=tf.constant_initializer(b))
                self.prediction = elu(self.h_out + self.b_out)
                
            else: # fully connected readout as in McIntosh et al. 2017
                c = layers.flatten(c)
                c = layers.dropout(c, 0.5, is_training=self.is_training)
                self.prediction = layers.fully_connected(
                    c,
                    self.data.num_neurons,
                    activation_fn=elu,
                    biases_initializer=tf.constant_initializer(b),
                    weights_regularizer=layers.l2_regularizer(0.01))
                
                # L1 on activity
                self.readout_sparseness_regularizer = readout_sparse_weight * tf.reduce_sum(
                    tf.reduce_mean(self.prediction, 0))
                tf.losses.add_loss(self.readout_sparseness_regularizer, tf.GraphKeys.REGULARIZATION_LOSSES)                
            
            # loss
            self.poisson = poisson(self.prediction, self.responses)
            tf.losses.add_loss(self.poisson)
            self.total_loss = tf.losses.get_total_loss()

            # regularizers
            self.smoothness_regularizer = tf.add_n(tf.get_collection('smoothness_regularizer_2d'))
            self.group_sparsity_regularizer = tf.add_n(tf.get_collection('group_sparsity_regularizer_2d'))

            # optimizer
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

            # initialize TF session
            self.initialize()


    def get_test_ops(self):
        return [self.poisson,
                self.readout_sparseness_regularizer,
                self.group_sparsity_regularizer,
                self.smoothness_regularizer,
                self.total_loss,
                self.prediction]

    
class LNP(Net):

    def build(self, smooth_reg_weight, sparse_reg_weight):
        self.smooth_reg_weight = smooth_reg_weight
        self.sparse_reg_weight = sparse_reg_weight
        with self.graph.as_default():
            tmp = tf.contrib.layers.convolution2d(self.images, self.data.num_neurons, self.data.px_x, 1, 'VALID',
                                                  activation_fn=tf.exp,
                                                  normalizer_fn=None,
                                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                  weights_regularizer=lambda w: smoothness_regularizer_2d(w, smooth_reg_weight) + \
                                                                                tf.contrib.layers.l1_regularizer(sparse_reg_weight)(w),
                                                  biases_initializer=tf.constant_initializer(value=-1.0),
                                                  scope='lnp')
            with tf.variable_scope('lnp', reuse=True):
                self.weights = tf.get_variable('weights')
                self.biases = tf.get_variable('biases')
            self.prediction = tf.squeeze(tmp, squeeze_dims=[1, 2])
            self.poisson = poisson(self.prediction, self.responses)
            self.total_loss = self.poisson + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)
            self.initialize()

    def get_test_ops(self):
        return [self.poisson, self.total_loss, self.prediction]

