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
# This file is originally from https://github.com/ildoonet/tf-pose-estimation/ and was modified

import os

import tensorflow as tf

from tf_pose.network_mpi_intermedSupervision import  NetworkMpi_interSup
from tf_pose.vgg_teacher import vgg_teacher
from tf_pose.vgg_conv4_2 import vgg_conv4_2


def _get_base_path():
    if not os.environ.get('OPENPOSE_MODEL', ''):
        return './models'
    return os.environ.get('OPENPOSE_MODEL')


def get_network(type, placeholder_input, sess_for_load=None, trainable=True, outputsize=None, nr_keypoints=16, nr_vectmaps=28, trainVgg=True):
    #all these have not been tested and will not work out of the box

    if type == 'vgg':
        net = CmuNetwork({'image': placeholder_input}, trainable=trainable)
        pretrain_path = 'numpy/openpose_vgg16.npy'
        last_layer = 'Mconv7_stage6_L{aux}'
    elif type == 'cmu_mpi_vgg_interSup':
        net = NetworkMpi_interSup({'image':placeholder_input}, trainable=trainable, nr_vectmaps=nr_vectmaps, nr_keypoints=nr_keypoints, trainVgg=trainVgg)
        pretrain_path =  'numpy/openpose_vgg19_10layers.npy'
        last_layer = 'Mconv7_stage6_L{aux}'
    elif type == 'vgg_teacher':
        net = vgg_teacher({'vgg_features':placeholder_input}, trainable=trainable, output_size=outputsize)
        pretrain_path = 'numpy/openpose_vgg19_10layers.npy'
        last_layer = 'meanAndVar'
    elif type =='vgg_conv4_2':
        net = vgg_conv4_2({'image':placeholder_input}, trainable=trainable)
        pretrain_path = 'numpy/openpose_vgg19_10layers.npy'
        last_layer = 'conv4_2'
    else:
        raise Exception('Invalid Mode.')


    pretrain_path_full = os.path.join(_get_base_path(), pretrain_path)
    if sess_for_load is not None:
        if type == 'cmu' or type == 'vgg' or type=='cmu_mpi_vgg'or type=='cmu_mpi':
            if not os.path.isfile(pretrain_path_full):
                raise Exception('Model file doesn\'t exist, path=%s' % pretrain_path_full)
            net.load(os.path.join(_get_base_path(), pretrain_path), sess_for_load)
        elif not(type =='heatmapTeacher'):
            s = '%dx%d' % (placeholder_input.shape[2], placeholder_input.shape[1])
            ckpts = {
                'mobilenet': 'trained/mobilenet_%s/model-246038' % s,
                'mobilenet_thin': 'trained/mobilenet_thin_%s/model-449003' % s,
                'mobilenet_fast': 'trained/mobilenet_fast_%s/model-189000' % s,
                'mobilenet_accurate': 'trained/mobilenet_accurate/model-170000'
            }
            ckpt_path = os.path.join(_get_base_path(), ckpts[type])
            loader = tf.train.Saver()
            try:
                loader.restore(sess_for_load, ckpt_path)
            except Exception as e:
                raise Exception('Fail to load model files. \npath=%s\nerr=%s' % (ckpt_path, str(e)))

    return net, pretrain_path_full, last_layer


def get_graph_path(model_name):
    dyn_graph_path = {
        'cmu': './models/graph/cmu/graph_opt.pb',
        'mobilenet_thin': './models/graph/mobilenet_thin/graph_opt.pb'
    }
    graph_path = dyn_graph_path[model_name]
    for path in (graph_path, os.path.join(os.path.dirname(os.path.abspath(__file__)), graph_path), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', graph_path)):
        if not os.path.isfile(path):
            continue
        return path
    raise Exception('Graph file doesn\'t exist, path=%s' % graph_path)


def model_wh(resolution_str):
    width, height = map(int, resolution_str.split('x'))
    if width % 16 != 0 or height % 16 != 0:
        raise Exception('Width and height should be multiples of 16. w=%d, h=%d' % (width, height))
    return int(width), int(height)
