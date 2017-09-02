from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

class Net(object):
    """Base Net class
    """
    def __init__(self, common_params, net_params):
        """The init function for Net class

        :param common_params: a params dict
        :param net_params: a params dictl
        """
        #pretrained variable collection
        self.pretrain_collection = []
        #trainable variable collection
        self.trainable_collection = []

    def _variable_on_cpu(self, name, shape, initializer, pretrain = True, train = True):
        """Helper to create a Variable stored on CPU memory

        :param name: name of the Variable
        :param shape: list of ints
        :param initializer: initializer of Variable
        :return: the Variable
        """
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer= initializer, dtype= tf.float32)
            if pretrain:
                self.pretrain_collection.append(var)
            if train:
                self.trainable_collection.append(var)

            return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd, pretrain = True, train = True):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with truncated normal distribution
        A weight decay is added only if one is specified.

        :param name: name of the variable
        :param shape: shape of the variable
        :param stddev: standard devision of a truncated normal distribution
        :param wd: add L2 loss weight decay multiply by this float. If None, weight decay is not added.
        :return: the Variable
        """
        var = self._variable_on_cpu(name, shape, initializer= tf.truncated_normal_initializer(stddev = stddev,
                                                                                              dtype= tf.float32))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name= 'weight_decay')
            tf.add_to_collection('losses', weight_decay)

        return var

    def conv2d(self, scope, input, kernel_size, stride = 1, pretrain = True, train = True):
        """convolution layer

        :param scope: variable scope name
        :param input: 4-D tensor [batch_size, height, width, depth]
        :param kernel_size: shape of convolutional kernel, [height, width, input_channel, output_channel]
        :param stride: the stride of convolution operation
        :return: 4-D tensor [batch_size, height, width, depth]
        """

        with tf.variable_scope(scope) as scope:
            kernel = self._variable_with_weight_decay('weights', shape= kernel_size,
                                                      stddev= 5e-2, wd = self.weight_decay, pretrain = pretrain,
                                                      train= train)
            conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding= 'SAME')
            bias = self._variable_on_cpu('biases', kernel_size[3], initializer= tf.constant_initializer(0.0),
                                         pretrain = pretrain, train = train)
            conv1 = tf.nn.bias_add(conv, bias)
            conv1 = self.leaky_relu(conv1)

        return conv1

    def max_pool(self, input, kernel_size, stride):
        """Max pooling layer

        :param input: 4-D Tensor [batch_size, height, width, depth]
        :param kernel_size: shape of convolutional kernel, [height, width, input_channel, output_channel]
        :param stride: the stride of convolution operation
        :return: 4-D Tensor [batch_size, height, width, depth]
        """
        return tf.nn.max_pool(input, [1, kernel_size[0], kernel_size[1], 1], [1, stride, stride, 1], padding= 'SAME')

    def local(self, scope, input, in_dimension, out_dimension, leaky = True, pretrain = True, train = True):
        """Fully connected layer

        :param scope: variable_scope name
        :param input: [batch_size, :]
        :param out_dimension: int 32
        :return: 2-D Tensor [batch_size, out_dimensions]
        """
        with tf.variable_scope(scope) as scope:
           reshape = tf.reshape(input, [self.batch_size, -1])
           weight = self._variable_with_weight_decay('weights', shape= [in_dimension, out_dimension],
                                                     stddev= 0.04, wd = self.weight_decay, pretrain=pretrain,
                                                     train= train)
           bias = self._variable_on_cpu('biases', [out_dimension], initializer= tf.constant_initializer(0),
                                        pretrain=pretrain, train= train)
           local = tf.matmul(reshape, weight) + bias

           if leaky:
               local = self.leaky_relu(local)
           else:
               local = tf.identity(local, name= scope.name)

        return local

    def leaky_relu(self, input, alpha = 0.1, dtype = tf.float32):
        """Leaky relu

        :param input: Tensor
        :param alpha: float
        :return: Tensor
        """
        x = tf.cast(input, dtype=dtype)
        bool_mask = (x > 0)
        mask = tf.cast(bool_mask, dtype= dtype)

        return mask * x + alpha * (1 - mask) * x

    def inference(self, images):
        """Build the YOLO model

        :param images: 4-D Tensor [batch_size, image_height, image_width, channles}
        :return: predicts: 4-D Tensor [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell]
        """
        raise NotImplementedError

    def loss(self, predicts, labels, objects_num):
        """Add Loss to all the trainable variable

        :param predicts: 4-D Tensor [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
        :param labels: 3-D Tensor of [batch_size, max_objects, 5]
        :param objects_num: 1-D Tesor [batch_size]
        :return:
        """
        raise NotImplementedError


