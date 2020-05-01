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
#first 10 layers of vgg19

class vgg_conv4_2(network_base.BaseNetwork):
    def setup(self, outputsize=10, trainable=True, nr_vectmaps=28, trainVgg=True):
        (self.feed('image')
             .normalize_vgg(name='preprocess')
             .conv(3, 3, 64, 1, 1, name='conv1_1',trainable=trainable)
             .conv(3, 3, 64, 1, 1, name='conv1_2',trainable=trainable)
             .max_pool(2, 2, 2, 2, name='pool1_stage1', padding='VALID')
             .conv(3, 3, 128, 1, 1, name='conv2_1',trainable=trainable)
             .conv(3, 3, 128, 1, 1, name='conv2_2',trainable=trainable)
             .max_pool(2, 2, 2, 2, name='pool2_stage1', padding='VALID')
             .conv(3, 3, 256, 1, 1, name='conv3_1',trainable=trainable)
             .conv(3, 3, 256, 1, 1, name='conv3_2',trainable=trainable)
             .conv(3, 3, 256, 1, 1, name='conv3_3',trainable=trainable)
             .conv(3, 3, 256, 1, 1, name='conv3_4',trainable=trainable)
             .max_pool(2, 2, 2, 2, name='pool3_stage1', padding='VALID')
             .conv(3, 3, 512, 1, 1, name='conv4_1',trainable=trainable)
                 .conv(3, 3, 512, 1, 1, name='conv4_2',trainable=trainable))
