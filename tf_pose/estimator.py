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
# This file originally from https://github.com/ildoonet/tf-pose-estimation/ was modified

import logging

import slidingwindow as sw

import cv2
import numpy as np
import tensorflow as tf
import time

from tf_pose import common
from tf_pose.common import CocoPart, MPIIPart
from tf_pose.tensblur.smoother import Smoother
from networks import get_network
from scipy.io import savemat

import numpy as np

import matplotlib.pyplot as plt
import os

try:
    # from tf_pose.pafprocess import pafprocess
    from tf_pose.pafprocess_mpi import pafprocess_mpi as pafprocess
except ModuleNotFoundError as e:
    print(e)
    print('you need to build c++ library for pafprocess. See : '
          'https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess')
    exit(-1)

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score', 'dataset')

    def __init__(self, pairs, dataset='Coco'):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        self.dataset = dataset
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1, pair.coord1[0], pair.coord1[1],
                                                   pair.score, self.dataset)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2, pair.coord2[0], pair.coord2[1],
                                                   pair.score, self.dataset)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()


class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score', 'dataset')
    orderMPI = [9, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5, 6]

    def __init__(self, uidx, part_idx, x, y, score, dataset='Coco'):
        self.uidx = uidx
        self.part_idx = self.orderMPI[part_idx]
        self.x, self.y = x, y
        self.score = score
        self.dataset = dataset

    def get_part_name(self):
        if self.dataset == 'Coco':
            return CocoPart(self.part_idx)
        else:
            return MPIIPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()


class PoseEstimator:
    def __init__(self):
        pass

    @staticmethod
    def estimate_paf(peaks, heat_mat, paf_mat, dataset='COCO'):
        pafprocess.process_paf(peaks, heat_mat, paf_mat)
        if dataset == 'COCO':
            nr_parts = 18
        elif dataset == 'MPI':
            nr_parts = 15  # 14?
        humans = []
        for human_id in range(pafprocess.get_num_humans()):
            human = Human([], dataset)
            is_added = False

            for part_idx in range(nr_parts):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    continue

                is_added = True
                human.body_parts[part_idx] = BodyPart('%d-%d' % (human_id, part_idx), part_idx,
                                                      float(pafprocess.get_part_x(c_idx)) /
                                                      heat_mat.shape[1],
                                                      float(pafprocess.get_part_y(c_idx)) /
                                                      heat_mat.shape[0],
                                                      pafprocess.get_part_score(c_idx),
                                                      dataset=dataset)

            if is_added:
                score = pafprocess.get_score(human_id)
                human.score = score
                humans.append(human)

        return humans


class TfPoseEstimator:

    def __init__(self, graph_path, target_size=(320, 240), tf_config=None, dataset='Coco',
                 nr_vectmaps=28):
        self.target_size = target_size
        self.scaled_img_size = None  # out of use
        self.pad = tf.placeholder(dtype=tf.int32, shape=(4,))

        logger.info('loading graph from %s(default size=%dx%d)' % (
            graph_path, target_size[0], target_size[1]))
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, name='TfPoseEstimator')
        self.persistent_sess = tf.Session(graph=self.graph, config=tf_config)

        if dataset == 'COCO':
            nr_keypoints = 19
            nr_vectmaps = 38
        elif dataset == 'MPI':
            nr_keypoints = 16

        if dataset == 'Coco':
            self.tensor_image = self.graph.get_tensor_by_name('TfPoseEstimator/image:0')
            self.tensor_output = self.graph.get_tensor_by_name(
                'TfPoseEstimator/Openpose/concat_stage7:0')
            self.tensor_heatMat = self.tensor_output[:, :, :, :nr_keypoints]
            self.tensor_pafMat = self.tensor_output[:, :, :, nr_keypoints:]
        else:
            if not ('caffe') in graph_path:
                self.tensor_image = self.graph.get_tensor_by_name(
                    'TfPoseEstimator/preprocess_divide:0')
                self.tensor_heatMat = self.graph.get_tensor_by_name(
                    'TfPoseEstimator/Mconv7_stage6_L2/BiasAdd:0')
                self.tensor_pafMat = self.graph.get_tensor_by_name(
                    'TfPoseEstimator/Mconv7_stage6_L1/BiasAdd:0')
            elif 'standalone' in graph_path:
                self.tensor_image = self.graph.get_tensor_by_name('TfPoseEstimator/input:0')
                self.tensor_heatMat = self.graph.get_tensor_by_name(
                    'TfPoseEstimator/Mconv7_stage6_L2/BiasAdd:0')
                self.tensor_pafMat = self.graph.get_tensor_by_name(
                    'TfPoseEstimator/Mconv7_stage6_L1/BiasAdd:0')
            else:
                self.tensor_image = self.graph.get_tensor_by_name('TfPoseEstimator/Placeholder:0')
                self.tensor_heatMat = self.graph.get_tensor_by_name(
                    'TfPoseEstimator/Mconv7_stage6_L2/BiasAdd:0')
                self.tensor_pafMat = self.graph.get_tensor_by_name(
                    'TfPoseEstimator/Mconv7_stage6_L1/BiasAdd:0')

        self.upsample_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='upsample_size')

        if dataset == 'Coco':
            self.tensor_heatMat_up = tf.image.resize_area(
                self.tensor_output[:, :, :, :nr_keypoints], self.upsample_size,
                align_corners=False, name='upsample_heatmat')
            self.tensor_pafMat_up = tf.image.resize_area(
                self.tensor_output[:, :, :, nr_keypoints:], self.upsample_size,
                align_corners=False, name='upsample_pafmat')
        else:
            self.tensor_heatMat = tf.image.resize_images(self.tensor_heatMat,
                                                         tf.shape(self.tensor_image)[1:3],
                                                         method=tf.image.ResizeMethod.BICUBIC)
            self.tensor_pafMat = tf.image.resize_images(self.tensor_pafMat,
                                                        tf.shape(self.tensor_image)[1:3],
                                                        method=tf.image.ResizeMethod.BICUBIC)

            np = self.tensor_heatMat.shape[-1] + self.tensor_pafMat.shape[-1]
            # if pad[0] < 0
            self.tensor_heatMat = tf.cond(self.pad[0] < 0,
                                          lambda: tf.concat([tf.concat([tf.zeros((1, -
                                          self.pad[0], tf.shape(self.tensor_heatMat)[1], tf.shape(
                                              self.tensor_heatMat)[-1])), tf.ones((1, -
                                          self.pad[0], tf.shape(self.tensor_heatMat)[1], 1))],
                                                                       axis=3),
                                                             self.tensor_heatMat], axis=0),
                                          lambda: self.tensor_heatMat[:, self.pad[0]:, :, :])

            self.tensor_pafMat = tf.cond(self.pad[0] < 0,
                                         lambda: tf.concat([tf.concat([tf.zeros((1, -
                                         self.pad[0], tf.shape(self.tensor_pafMat)[1], tf.shape(
                                             self.tensor_pafMat)[-1])), tf.ones((1, -
                                         self.pad[0], tf.shape(self.tensor_pafMat)[1], 1))],
                                                                      axis=3), self.tensor_pafMat],
                                                           axis=0),
                                         lambda: self.tensor_pafMat[:, self.pad[0]:, :, :])

            # if pad[1] < 0
            self.tensor_heatMat = tf.cond(self.pad[1] < 0, lambda: tf.concat(
                [tf.concat([tf.zeros((1, tf.shape(self.tensor_heatMat)[0], -
                self.pad[1], tf.shape(self.tensor_heatMat)[-1])),
                            tf.ones((1, tf.shape(self.tensor_heatMat)[0], -
                            self.pad[1], 1))], axis=3), self.tensor_heatMat], axis=1),
                                          lambda: self.tensor_heatMat[:, :, self.pad[1]:, :])

            self.tensor_pafMat = tf.cond(self.pad[1] < 0, lambda: tf.concat(
                [tf.concat([tf.zeros((1, tf.shape(self.tensor_pafMat)[0], -
                self.pad[1], tf.shape(self.tensor_pafMat)[-1])),
                            tf.ones((1, tf.shape(self.tensor_pafMat)[0], -
                            self.pad[1], 1))], axis=3), self.tensor_pafMat], axis=1),
                                         lambda: self.tensor_pafMat[:, :, self.pad[1]:, :])

            # if pad[2] < 0
            self.tensor_heatMat = tf.cond(self.pad[2] < 0,
                                          lambda: tf.concat([tf.concat([tf.zeros((1, -
                                          self.pad[2], tf.shape(self.tensor_heatMat)[1], tf.shape(
                                              self.tensor_heatMat)[-1])), tf.ones((1, -
                                          self.pad[2], tf.shape(self.tensor_heatMat)[1], 1))],
                                                                       axis=3),
                                                             self.tensor_heatMat], axis=0),
                                          lambda: tf.cond(tf.equal(self.pad[2], 0),
                                                          lambda: self.tensor_heatMat[:, :, :, :],
                                                          lambda: self.tensor_heatMat[:,
                                                                  :-self.pad[2], :, :]))

            self.tensor_pafMat = tf.cond(self.pad[2] < 0,
                                         lambda: tf.concat([tf.concat([tf.zeros((1, -
                                         self.pad[2], tf.shape(self.tensor_pafMat)[1], tf.shape(
                                             self.tensor_pafMat)[-1])), tf.ones((1, -
                                         self.pad[2], tf.shape(self.tensor_pafMat)[1], 1))],
                                                                      axis=3), self.tensor_pafMat],
                                                           axis=0),
                                         lambda: tf.cond(tf.equal(self.pad[2], 0),
                                                         lambda: self.tensor_pafMat[:, :, :, :],
                                                         lambda: self.tensor_pafMat[:,
                                                                 :-self.pad[2], :, :]))
            # if pad[3] < 0
            self.tensor_heatMat = tf.cond(self.pad[3] < 0, lambda: tf.concat(
                [tf.concat([tf.zeros((1, tf.shape(self.tensor_heatMat)[0], -
                self.pad[3], tf.shape(self.tensor_heatMat)[-1])),
                            tf.ones((1, tf.shape(self.tensor_heatMat)[0], -
                            self.pad[3], 1))], axis=3), self.tensor_heatMat], axis=1),
                                          lambda: tf.cond(tf.equal(self.pad[3], 0),
                                                          lambda: self.tensor_heatMat[:, :, :, :],
                                                          lambda: self.tensor_heatMat[:, :,
                                                                  :-self.pad[3], :]))

            self.tensor_pafMat = tf.cond(self.pad[1] < 0, lambda: tf.concat(
                [tf.concat([tf.zeros((1, tf.shape(self.tensor_pafMat)[0], -
                self.pad[3], tf.shape(self.tensor_pafMat)[-1])),
                            tf.ones((1, tf.shape(self.tensor_pafMat)[0], -
                            self.pad[3], 1))], axis=3), self.tensor_pafMat], axis=1),
                                         lambda: tf.cond(tf.equal(self.pad[3], 0),
                                                         lambda: self.tensor_pafMat[:, :, :, :],
                                                         lambda: self.tensor_pafMat[:, :,
                                                                 :-self.pad[3], :]))

            # SCALE THE image
            self.tensor_heatMat_up = tf.image.resize_images(self.tensor_heatMat[:, :, :, :],
                                                            self.upsample_size,
                                                            method=tf.image.ResizeMethod.BICUBIC)
            self.tensor_pafMat_up = tf.image.resize_images(self.tensor_pafMat[:, :, :, :],
                                                           self.upsample_size,
                                                           method=tf.image.ResizeMethod.BICUBIC)  #

        self.tensor_heatMat_up_scale = tf.placeholder(dtype=tf.float32,
                                                      shape=self.tensor_heatMat_up.get_shape())

        smoother = Smoother({'data': self.tensor_heatMat_up}, 25, 3.0)
        gaussian_heatMat = smoother.get_output()

        # doing the non maximum supression
        max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3),
                                          pooling_type='MAX', padding='SAME')
        self.tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor),
                                     gaussian_heatMat, tf.zeros_like(gaussian_heatMat))

        smoother_final = Smoother({'data': self.tensor_heatMat_up_scale}, 25, 3.0)
        gaussian_heatMat_final = smoother_final.get_output()

        max_pooled_in_tensor_final = tf.nn.pool(gaussian_heatMat_final, window_shape=(3, 3),
                                                pooling_type='MAX', padding='SAME')
        self.tensor_peaks_final = tf.where(
            tf.equal(gaussian_heatMat_final, max_pooled_in_tensor_final), gaussian_heatMat_final,
            tf.zeros_like(gaussian_heatMat_final))

        self.heatMat = self.pafMat = None

        # warm-up
        self.persistent_sess.run(tf.variables_initializer([v for v in tf.global_variables() if
                                                           v.name.split(':')[0] in [
                                                               x.decode('utf-8') for x in
                                                               self.persistent_sess.run(
                                                                   tf.report_uninitialized_variables())]]))

    def __del__(self):
        pass


    @staticmethod
    def _quantize_img(npimg):
        npimg_q = npimg + 1.0
        npimg_q /= (2.0 / 2 ** 8)
        # npimg_q += 0.5
        npimg_q = npimg_q.astype(np.uint8)
        return npimg_q


    def get_subplot(self, npimg, humans, imgcopy=False, dataset='COCO', draw_line=True, model='',
                    return_candidates=False, footOnly=False):
        plt.ioff()
        fig = plt.figure(figsize=(npimg.shape[1], npimg.shape[0]))
        orderMPI = [9, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5]
        a = fig.add_subplot(2, 1, 1)
        a.set_title('Heatmap & Detections (thr=0.05)' + model, fontsize=25)
        plt.imshow(cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB))

        if footOnly:
            tmp = np.amax(self.heatMat[:, :, -3:-1], axis=2)
        else:
            tmp = np.amax(self.heatMat[:, :, :-1], axis=2)
        plt.imshow(tmp, cmap=plt.cm.jet, alpha=0.8)
        plt.colorbar()

        image_h, image_w = npimg.shape[:2]
        for human in humans:
            # draw point
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5),
                          int(body_part.y * image_h + 0.5))
                plt.scatter(int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5),
                            color=np.array(common.CocoColors[i]) / 255)
        candidates = []
        for i in range(self.peaks.shape[2] - 1):

            peak = np.where(self.peaks[:, :, i] > 0.05)
            for j in range(peak[0].shape[0]):

                if i < 14:
                    candidates.append(np.array(
                        [peak[0][j], peak[1][j], self.peaks[peak[0][j], peak[1][j], i],
                         orderMPI[i]]))
                if j == 1:
                    plt.scatter(peak[1][j], peak[0][j], color=np.array(common.CocoColors[i]) / 255,
                                label=i, s=20 * 4)
                else:
                    plt.scatter(peak[1][j], peak[0][j], color=np.array(common.CocoColors[i]) / 255,
                                s=20 * 4)

        plt.legend(bbox_to_anchor=(-0.1, 0.5), loc='center left', borderaxespad=0.,
                   prop={'size': 12})

        plt.legend()
        supl2 = self.draw_humans(cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB), humans, imgcopy=False,
                                 dataset=dataset, draw_line=True)
        a = fig.add_subplot(2, 1, 2)
        a.set_title('Result Inference ' + model, fontsize=25)
        plt.imshow(supl2)
        plt.colorbar()

        if return_candidates:
            return fig, candidates
        else:
            return fig


    def get_drawing(self, npimg, humans, imgcopy=False, dataset='COCO', draw_line=True, model='',
                    return_candidates=False, footOnly=False):
        plt.ioff()
        fig = plt.figure(figsize=(npimg.shape[1], npimg.shape[0]))
        orderMPI = [9, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5]
        supl2 = self.draw_humans(cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB), humans, imgcopy=False,
                                 dataset=dataset, draw_line=True)
        a = fig.add_subplot(2, 1, 2)
        a.set_title('Result Inference ' + model, fontsize=25)
        plt.imshow(supl2)
        plt.colorbar()

        return fig


    def get_subplot_drawingOnly(self, npimg, humans, imgcopy=False, dataset='COCO', draw_line=True,
                                model='', return_candidates=False, footOnly=False, path=None,
                                model_name=None, counter=0):
        plt.ioff()

        fig = plt.figure(figsize=(npimg.shape[1] / 100., npimg.shape[0] / 100.))

        orderMPI = [9, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5]
        plt.imshow(cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB))
        candidates = []
        supl2 = self.draw_humans(cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB), humans, imgcopy=False,
                                 dataset=dataset, draw_line=True)
        plt.tight_layout()
        plt.axis('off')
        plt.imshow(supl2)

        plt.savefig(
            './' + 'im_' + str(counter) + 'png')
        if return_candidates:
            return fig, candidates
        else:
            return fig


    def get_subplot_groups(self, npimg, humans, imgcopy=False, dataset='COCO', draw_line=True,
                           model='', return_candidates=False, footOnly=False,
                           group_boundaries=None):
        plt.ioff()
        fig = plt.figure(figsize=(20, 20))
        orderMPI = [9, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5]
        a = fig.add_subplot(2, 1, 1)
        a.set_title('Heatmap & Detections (thr=0.05)' + model, fontsize=25)
        plt.imshow(cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB))

        tmp = np.amax(self.heatMat[:, :, :-1], axis=2)
        plt.imshow(tmp, cmap=plt.cm.jet, alpha=0.8)
        plt.colorbar()

        image_h, image_w = npimg.shape[:2]
        candidates = []
        for i in range(self.peaks.shape[2] - 1):

            peak = np.where(self.peaks[:, :, i] > 0.05)
            for j in range(peak[0].shape[0]):

                if i < 14:
                    candidates.append(np.array(
                        [peak[0][j], peak[1][j], self.peaks[peak[0][j], peak[1][j], i],
                         orderMPI[i]]))
                if j == 1:
                    plt.scatter(peak[1][j], peak[0][j], color=np.array(common.CocoColors[i]) / 255,
                                label=i, s=20 * 4)
                else:
                    plt.scatter(peak[1][j], peak[0][j], color=np.array(common.CocoColors[i]) / 255,
                                s=20 * 4)

        plt.legend(bbox_to_anchor=(-0.1, 0.5), loc='center left', borderaxespad=0.,
                   prop={'size': 12})
        plt.tight_layout()

        supl2 = self.draw_humans(cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB), humans, imgcopy=False,
                                 dataset=dataset, draw_line=True)
        cv2.rectangle(supl2, (group_boundaries[0], group_boundaries[2]),
                      (group_boundaries[1], group_boundaries[3]),
                      np.array(common.CocoColors[0]) / 255, thickness=3)
        print(group_boundaries)

        a = fig.add_subplot(2, 1, 2)
        a.set_title('Result Inference ' + model, fontsize=25)
        plt.imshow(supl2)
        plt.colorbar()
        plt.tight_layout()
        if return_candidates:
            return fig, candidates
        else:
            return fig


    @staticmethod
    def draw_humans(npimg_in, humans, imgcopy=False, dataset='COCO', draw_line=True):
        # if imgcopy:
        npimg = np.copy(npimg_in)
        image_h, image_w = npimg.shape[:2]
        centers = {}
        for human in humans:
            # draw point
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5),
                          int(body_part.y * image_h + 0.5))
                centers[i] = center
                cv2.circle(npimg, center, 3, np.array(common.CocoColors[i]) / 255, thickness=3,
                           lineType=8, shift=0)

            if draw_line:
                # draw line
                if dataset == 'COCO':
                    pairsrender = common.CocoPairsRender
                else:
                    pairsrender = common.MPIPairs

                for pair_order, pair in enumerate(pairsrender):
                    if pair[0] not in human.body_parts.keys() or pair[
                        1] not in human.body_parts.keys():
                        continue

                    cv2.line(npimg, centers[pair[0]], centers[pair[1]],
                             np.array(common.CocoColors[pair_order]) / 255, 3)

        image_new = npimg

        return image_new


    def _get_scaled_img(self, npimg, scale, use_pad=False, box_size=None):
        get_base_scale = lambda s, w, h: max(self.target_size[0] / float(h),
                                             self.target_size[1] / float(w)) * s
        img_h, img_w = npimg.shape[:2]

        if scale is None:
            if npimg.shape[:2] != (self.target_size[1], self.target_size[0]):
                # resize
                npimg = cv2.resize(npimg, self.target_size, interpolation=cv2.INTER_CUBIC)
            return [npimg], [(0.0, 0.0, 1.0, 1.0)]

        elif isinstance(scale, float) and use_pad:
            print('scale' + str(scale))
            print(npimg.shape)
            npimg = cv2.resize(npimg, dsize=None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)
            bbox = [box_size, np.max([box_size, np.shape(npimg)[1]])]
            # pad height
            pad = np.zeros((4,), dtype=int)
            h, w, _ = np.shape(npimg)
            h = np.min([bbox[0], h])
            bbox[0] = np.ceil(bbox[0] / 8) * 8
            bbox[1] = np.max([bbox[1], w])
            bbox[1] = np.ceil(bbox[1] / 8) * 8
            pad[0] = int(np.floor((bbox[0] - h) / 2))  # up
            pad[1] = int(np.floor((bbox[1] - w) / 2))  # left
            pad[2] = int(bbox[0] - h - pad[0])  # down
            pad[3] = int(bbox[1] - w - pad[1])  # right

            pad_up = np.ones((pad[0], np.shape(npimg)[1], np.shape(npimg)[2])) * 0.5
            npimg = np.concatenate([pad_up, npimg], axis=0)
            pad_left = np.ones((np.shape(npimg)[0], pad[1], np.shape(npimg)[2])) * 0.5
            npimg = np.concatenate([pad_left, npimg], axis=1)
            pad_down = np.ones((pad[2], np.shape(npimg)[1], np.shape(npimg)[2])) * 0.5
            npimg = np.concatenate([npimg, pad_down], axis=0)
            pad_right = np.ones((np.shape(npimg)[0], pad[3], np.shape(npimg)[2])) * 0.5
            npimg = np.concatenate([npimg, pad_right], axis=1)

            return npimg, pad

        elif isinstance(scale, float):
            # scaling with center crop
            base_scale = get_base_scale(scale, img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale,
                               interpolation=cv2.INTER_CUBIC)

            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.target_size[1] or npimg.shape[1] < self.target_size[0]:
                newimg = np.zeros((max(self.target_size[1], npimg.shape[0]),
                                   max(self.target_size[0], npimg.shape[1]), 3), dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg

            windows = sw.generate(npimg, sw.DimOrder.HeightWidthChannel, self.target_size[0],
                                  self.target_size[1], 0.2)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(
                    indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(
                    indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))

            return rois, ratios

        elif isinstance(scale, tuple) and len(scale) == 2:
            base_scale = get_base_scale(scale[0], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale,
                               interpolation=cv2.INTER_CUBIC)
            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.target_size[1] or npimg.shape[1] < self.target_size[0]:
                newimg = np.zeros((max(self.target_size[1], npimg.shape[0]),
                                   max(self.target_size[0], npimg.shape[1]), 3), dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg

            window_step = scale[1]

            windows = sw.generate(npimg, sw.DimOrder.HeightWidthChannel, self.target_size[0],
                                  self.target_size[1], window_step)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(
                    indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(
                    indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))

            return rois, ratios
        elif isinstance(scale, tuple) and len(scale) == 3:
            base_scale = get_base_scale(scale[2], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale,
                               interpolation=cv2.INTER_CUBIC)
            ratio_w = self.target_size[0] / float(npimg.shape[1])
            ratio_h = self.target_size[1] / float(npimg.shape[0])

            want_x, want_y = scale[:2]
            ratio_x = want_x - ratio_w / 2.
            ratio_y = want_y - ratio_h / 2.
            ratio_x = max(ratio_x, 0.0)
            ratio_y = max(ratio_y, 0.0)
            if ratio_x + ratio_w > 1.0:
                ratio_x = 1. - ratio_w
            if ratio_y + ratio_h > 1.0:
                ratio_y = 1. - ratio_h

            roi = self._crop_roi(npimg, ratio_x, ratio_y)

            return [roi], [(ratio_x, ratio_y, ratio_w, ratio_h)]


    def _crop_roi(self, npimg, ratio_x, ratio_y):
        target_w, target_h = self.target_size
        h, w = npimg.shape[:2]
        x = max(int(w * ratio_x - .5), 0)
        y = max(int(h * ratio_y - .5), 0)
        cropped = npimg[y:y + target_h, x:x + target_w]

        cropped_h, cropped_w = cropped.shape[:2]
        if cropped_w < target_w or cropped_h < target_h:
            npblank = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)

            copy_x, copy_y = (target_w - cropped_w) // 2, (target_h - cropped_h) // 2
            npblank[copy_y:copy_y + cropped_h, copy_x:copy_x + cropped_w] = cropped
        else:
            return cropped


    def inference(self, npimg, resize_to_default=True, upsample_size=1.0, data_set='COCO',
                  model=None, scales=None, bbox=None, id=None):
        if npimg is None:
            raise Exception('The image is not valid. Please check your image exists.')

        if resize_to_default:
            upsample_size = [int(self.target_size[1] / 8 * upsample_size),
                             int(self.target_size[0] / 8 * upsample_size)]
        else:
            upsample_size = [int(npimg.shape[0] / 8 * upsample_size),
                             int(npimg.shape[1] / 8 * upsample_size)]

        if self.tensor_image.dtype == tf.quint8:
            npimg = TfPoseEstimator._quantize_img(npimg)
            pass

        logger.debug('inference+ original shape=%dx%d' % (npimg.shape[1], npimg.shape[0]))

        peaks_scale = []
        heatMat_up_scale = []
        pafMat_up_scale = []
        for scale in scales:
            img = npimg
            if resize_to_default:
                img = self._get_scaled_img(npimg, None)[0][0]
            else:
                img, pad = self._get_scaled_img(npimg, scale, use_pad=not (resize_to_default),
                                                box_size=bbox)
            if 'caffe' in model:
                img = img * 255
            peaks, heatMat_up, pafMat_up = self.persistent_sess.run(
                [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
                feed_dict={self.tensor_image: [img], self.upsample_size: upsample_size,
                           self.pad: pad})
            peaks_scale.append(peaks[0])
            heatMat_up_scale.append(heatMat_up[0])
            pafMat_up_scale.append(pafMat_up[0])

        final_heatMat = np.zeros_like(heatMat_up_scale[0])

        for heatMat in heatMat_up_scale:
            final_heatMat += heatMat
        final_heatMat /= len(heatMat_up_scale)
        final_heatMat = np.reshape(final_heatMat, (
            -1, final_heatMat.shape[0], final_heatMat.shape[1], final_heatMat.shape[2]))
        final_pafMat = np.zeros_like(pafMat_up_scale[0])

        for pafMat in pafMat_up_scale:
            final_pafMat += pafMat
        final_pafMat /= len(pafMat_up_scale)

        peaks = self.persistent_sess.run([self.tensor_peaks_final],
                                         feed_dict={self.tensor_heatMat_up_scale: final_heatMat})
        peaks = peaks[0][0, :, :, :]
        self.peaks = peaks
        self.heatMat = final_heatMat[0]
        self.pafMat = final_pafMat

        logger.debug('inference- heatMat=%dx%d pafMat=%dx%d' % (
            self.heatMat.shape[1], self.heatMat.shape[0], self.pafMat.shape[1],
            self.pafMat.shape[0]))
        t = time.time()

        humans = PoseEstimator.estimate_paf(peaks, self.heatMat, self.pafMat, data_set)
        logger.debug('estimate time=%.5f' % (time.time() - t))

        return humans


    if __name__ == '__main__':
        import pickle

        f = open('./etcs/heatpaf1.pkl', 'rb')
        data = pickle.load(f)
        logger.info('size={}'.format(data['heatMat'].shape))
        f.close()

        t = time.time()
        humans = PoseEstimator.estimate_paf(data['peaks'], data['heatMat'], data['pafMat'])
        dt = time.time() - t;
        t = time.time()
        logger.info('elapsed #humans=%d time=%.8f' % (len(humans), dt))
