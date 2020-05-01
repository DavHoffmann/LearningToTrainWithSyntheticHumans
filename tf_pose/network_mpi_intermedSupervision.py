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


class NetworkMpi_interSup(network_base.BaseNetwork):
    def setup(self, nr_vectmaps=28, nr_keypoints=16, trainVgg=True):
        (self.feed('image')
             .normalize_vgg(name='preprocess')
             .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=trainVgg)
             .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=trainVgg)
             .max_pool(2, 2, 2, 2, name='pool1_stage1', padding='VALID')
             .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=trainVgg)
             .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=trainVgg)
             .max_pool(2, 2, 2, 2, name='pool2_stage1', padding='VALID')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .conv(3, 3, 256, 1, 1, name='conv3_4')
             .max_pool(2, 2, 2, 2, name='pool3_stage1', padding='VALID')
             .conv(3, 3, 512, 1, 1, name='conv4_1' )
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 256, 1, 1, name='conv4_3_CPM')
             .conv(3, 3, 128, 1, 1, name='conv4_4_CPM')          # *****

             .conv(3, 3, 128, 1, 1, name='conv5_1_CPM_L1')
             .conv(3, 3, 128, 1, 1, name='conv5_2_CPM_L1')
             .conv(3, 3, 128, 1, 1, name='conv5_3_CPM_L1')
             .conv(1, 1, 512, 1, 1, name='conv5_4_CPM_L1')
             .conv(1, 1, nr_vectmaps, 1, 1, relu=False, name='conv5_5_CPM_L1'))

        (self.feed('conv4_4_CPM')
             .conv(3, 3, 128, 1, 1, name='conv5_1_CPM_L2')
             .conv(3, 3, 128, 1, 1, name='conv5_2_CPM_L2')
             .conv(3, 3, 128, 1, 1, name='conv5_3_CPM_L2')
             .conv(1, 1, 512, 1, 1, name='conv5_4_CPM_L2')
             .conv(1, 1, nr_keypoints, 1, 1, relu=False, name='conv5_5_CPM_L2'))

        (self.feed('conv5_5_CPM_L1',
                   'conv5_5_CPM_L2',
                   'conv4_4_CPM')
             .concat(3, name='concat_stage2')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage2_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage2_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage2_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage2_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage2_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage2_L1')
             .conv(1, 1, nr_vectmaps, 1, 1, relu=False, name='Mconv7_stage2_L1'))

        (self.feed('concat_stage2')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage2_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage2_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage2_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage2_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage2_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage2_L2')
             .conv(1, 1, nr_keypoints, 1, 1, relu=False, name='Mconv7_stage2_L2'))

        (self.feed('Mconv7_stage2_L1',
                   'Mconv7_stage2_L2',
                   'conv4_4_CPM')
             .concat(3, name='concat_stage3')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage3_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage3_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage3_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage3_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage3_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage3_L1')
             .conv(1, 1, nr_vectmaps, 1, 1, relu=False, name='Mconv7_stage3_L1'))

        (self.feed('concat_stage3')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage3_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage3_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage3_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage3_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage3_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage3_L2')
             .conv(1, 1, nr_keypoints, 1, 1, relu=False, name='Mconv7_stage3_L2'))

        (self.feed('Mconv7_stage3_L1',
                   'Mconv7_stage3_L2',
                   'conv4_4_CPM')
             .concat(3, name='concat_stage4')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage4_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage4_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage4_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage4_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage4_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage4_L1')
             .conv(1, 1, nr_vectmaps, 1, 1, relu=False, name='Mconv7_stage4_L1'))

        (self.feed('concat_stage4')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage4_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage4_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage4_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage4_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage4_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage4_L2')
             .conv(1, 1, nr_keypoints, 1, 1, relu=False, name='Mconv7_stage4_L2'))

        (self.feed('Mconv7_stage4_L1',
                   'Mconv7_stage4_L2',
                   'conv4_4_CPM')
             .concat(3, name='concat_stage5')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage5_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage5_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage5_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage5_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage5_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage5_L1')
             .conv(1, 1, nr_vectmaps, 1, 1, relu=False, name='Mconv7_stage5_L1'))

        (self.feed('concat_stage5')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage5_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage5_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage5_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage5_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage5_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage5_L2')
             .conv(1, 1, nr_keypoints, 1, 1, relu=False, name='Mconv7_stage5_L2'))

        (self.feed('Mconv7_stage5_L1',
                   'Mconv7_stage5_L2',
                   'conv4_4_CPM')
             .concat(3, name='concat_stage6')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage6_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage6_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage6_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage6_L1')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage6_L1')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage6_L1')
             .conv(1, 1, nr_vectmaps, 1, 1, relu=False, name='Mconv7_stage6_L1'))

        (self.feed('concat_stage6')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage6_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage6_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage6_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage6_L2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage6_L2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage6_L2')
             .conv(1, 1, nr_keypoints, 1, 1, relu=False, name='Mconv7_stage6_L2'))

        with tf.variable_scope('Openpose'):
            (self.feed('Mconv7_stage6_L2',
                       'Mconv7_stage6_L1')
                 .concat(3, name='concat_stage7'))

    def loss_l1_l2(self):
         l1s = []
         l2s = []
         for layer_name in self.layers.keys():
              if (('Mconv7' in layer_name or 'conv5_5_CPM' in layer_name) and '_L1' in layer_name):
                   l1s.append(self.layers[layer_name])
              if (('Mconv7' in layer_name or 'conv5_5_CPM' in layer_name) and '_L2' in layer_name):
                   l2s.append(self.layers[layer_name])

         return l1s, l2s

    def loss_last(self):
         return self.get_output('Mconv7_stage6_L1'), self.get_output('Mconv7_stage6_L2')

    def restorable_variables(self):
         return None
