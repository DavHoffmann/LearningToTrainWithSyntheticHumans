# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


from __future__ import absolute_import

from tf_pose import network_base
import tensorflow as tf
from functools import reduce
import numpy as np
import logging

logger = logging.getLogger('vgg_teacher')

DEFAULT_PADDING = 'SAME'

class vgg_teacher(network_base.BaseNetwork):
    def setup(self, outpusize=10, nr_vectmaps=28, trainVgg=True):
        (self.feed('vgg_features')

             .conv(3, 3, 256, 1, 1, name='teacher_conv4_3')
             .conv(3, 3, 128, 1, 1, name='teacher_conv4_4')


             .max_pool(2, 2, 2, 2, name='teacher_pool4', padding='VALID')
             .conv(3, 3, 64, 1, 1, name='teacher_conv5_1')
             .conv(3, 3, 64, 1, 1, name='teacher_conv5_2')
             .max_pool( 2, 2, 2, 2, name='teacher_pool5', padding='VALID')

             .fc( 512, name= "teacher_fully_connected6")
             .fc( 512, name="teacher_fully_connected_7")
         )


        (self.feed('teacher_fully_connected_7')
              .fc(outpusize, name="visibility_ratio_fc")
             .softmax_vgg( name="visibility_ratio_total_probs")
         )


    def get_fc(self):
        fcs = []
        keys = list(self.layers.keys())
        keys.sort()
        for layer_name in keys:
            if '_fc' in layer_name:
                fcs.append(self.layers[layer_name])
        return fcs

    def get_probabilities(self):
        probs = []
        keys = list(self.layers.keys())
        keys.sort()
        for layer_name in keys:
            if 'probs' in layer_name:
                probs.append(self.layers[layer_name])
        return probs

    def get_output_names(self):
        names = []
        keys = list(self.layers.keys())
        keys.sort()
        for layer_name in keys:
            if 'probs' in layer_name:

                layer_name = self.layers[layer_name]
                names.append( layer_name.name.split('_probs')[0] )
        return names

    @network_base.layer
    def softmax_vgg(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())

        return tf.nn.softmax(input, name=name)


    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool_vgg(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())

        return count

