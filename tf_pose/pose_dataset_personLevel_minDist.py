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
import math
import multiprocessing
import struct
import sys
import threading

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from contextlib import contextmanager

import os
import random
import requests
import cv2
import numpy as np
import time
import json
from collections import defaultdict
from math import log, exp
import tensorflow as tf


import collections
import itertools
from tensorpack.dataflow import MultiThreadMapData, MultiProcessMapDataZMQ
from tensorpack.dataflow.image import MapDataComponent
from tensorpack.dataflow.common import BatchData, MapData
from tensorpack.dataflow.parallel import PrefetchData, PrefetchDataZMQ
from tensorpack.dataflow.base import RNGDataFlow, DataFlowTerminated
#import matplotlib.pyplot as plt
# from pycocotools.coco import COCO
try:
    from tf_pose.pose_augment_distSampling import pose_flip, pose_rotation, pose_to_img, pose_crop_random, \
    pose_resize_shortestedge_random, pose_resize_shortestedge_fixed, pose_crop_center, pose_random_scale, pose_crop_person_center, pose_cpm_original_scale
except:
    from pose_augment_distSampling import pose_flip, pose_rotation, pose_to_img, pose_crop_random, \
        pose_resize_shortestedge_random, pose_resize_shortestedge_fixed, pose_crop_center, pose_random_scale, pose_crop_person_center, pose_cpm_original_scale


logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger('pose_dataset')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

mplset = False
mpiMaskPath = './masks'



class CocoMetadata:

    @staticmethod
    def parse_float(four_np):
        assert len(four_np) == 4
        return struct.unpack('<f', bytes(four_np))[0]

    @staticmethod
    def parse_floats(four_nps, adjust=0):
        assert len(four_nps) % 4 == 0
        return [(CocoMetadata.parse_float(four_nps[x*4:x*4+4]) + adjust) for x in range(len(four_nps) // 4)]

    def __init__(self, idx, img_url, img_meta, annotations, sigma):
        # __coco_parts = 57
        self.__parts = 19
        self.__vecs = list(zip(
            [2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16],
            [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
        ))
        self.idx = idx
        self.img_url = img_url
        self.img = None
        self.sigma = sigma

        self.height = int(img_meta['height'])
        self.width = int(img_meta['width'])

        joint_list = []
        for ann in annotations:
            if ann.get('num_keypoints', 0) == 0:
                continue

            kp = np.array(ann['keypoints'])
            xs = kp[0::3]
            ys = kp[1::3]
            vs = kp[2::3]

            joint_list.append([(x, y) if v >= 1 else (-1000, -1000) for x, y, v in zip(xs, ys, vs)])

        self.joint_list = []
        transform = list(zip(
            [1, 6, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4],
            [1, 7, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
        ))
        for prev_joint in joint_list:
            new_joint = []
            for idx1, idx2 in transform:
                j1 = prev_joint[idx1-1]
                j2 = prev_joint[idx2-1]

                if j1[0] <= 0 or j1[1] <= 0 or j2[0] <= 0 or j2[1] <= 0:
                    new_joint.append((-1000, -1000))
                else:
                    new_joint.append(((j1[0] + j2[0]) / 2, (j1[1] + j2[1]) / 2))

            new_joint.append((-1000, -1000))
            self.joint_list.append(new_joint)


    def get_heatmap(self, target_size):
        heatmap = np.zeros((self.__parts, self.height, self.width), dtype=np.float32)

        for joints in self.joint_list:
            for idx, point in enumerate(joints):
                if point[0] < 0 or point[1] < 0:
                    continue
                CocoMetadata.put_heatmap(heatmap, idx, point, self.sigma)

        heatmap = heatmap.transpose((1, 2, 0))

        # background
        heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)

        if target_size:
            heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_AREA)

        return heatmap.astype(np.float16)

    @staticmethod
    def put_heatmap(heatmap, plane_idx, center, sigma):
        center_x, center_y = center
        _, height, width = heatmap.shape[:3]

        th = 4.6052
        delta = math.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))

        x1 = int(min(width, center_x + delta * sigma))
        y1 = int(min(height, center_y + delta * sigma))


        exp_factor = 1 / 2.0 / sigma / sigma
        arr_heatmap = heatmap[plane_idx, y0:y1, x0:x1]

        if y1 == 368:
            y_vec = (np.arange(y0, y1 ) - center_y) ** 2

        else:
            y_vec = (np.arange(y0, y1 ) - center_y) ** 2  # y1 included
        if x1 == 368:
            x_vec = (np.arange(x0, x1 ) - center_x) ** 2
        else:
            x_vec = (np.arange(x0, x1 ) - center_x) ** 2
        xv, yv = np.meshgrid(x_vec, y_vec)
        arr_sum = exp_factor * (xv + yv)
        arr_exp = np.exp(-arr_sum)
        arr_exp[arr_sum > th] = 0

        mm = np.maximum(arr_heatmap, arr_exp)
        nn = np.minimum(mm, np.ones_like(mm))
        heatmap[plane_idx, y0:y1 , x0:x1 ] = nn


    def get_vectormap(self, target_size):
        vectormap = np.zeros((self.__vects*2, self.height, self.width), dtype=np.float32)
        countmap = np.zeros((self.__parts, self.height, self.width), dtype=np.int16)
        for joints in self.joint_list:
            for plane_idx, (j_idx1, j_idx2) in enumerate(self.__vecs):
                j_idx1 -= 1
                j_idx2 -= 1

                center_from = joints[j_idx1]
                center_to = joints[j_idx2]

                if center_from[0] < -50 or center_from[1] < -50 or center_to[0] < -50 or center_to[1] < -50 or center_from[0] > self.width+50 or center_from[1] > self.height+50 or center_to[0] > self.width+50 or center_to[1] > self.height+50:
                    continue

                CocoMetadata.put_vectormap(vectormap, countmap, plane_idx, center_from, center_to)

        vectormap = vectormap.transpose((1, 2, 0))
        nonzeros = np.nonzero(countmap)
        for p, y, x in zip(nonzeros[0], nonzeros[1], nonzeros[2]):
            if countmap[p][y][x] <= 0:
                continue
            vectormap[y][x][p*2+0] /= countmap[p][y][x]
            vectormap[y][x][p*2+1] /= countmap[p][y][x]

        if target_size:
            vectormap = cv2.resize(vectormap, target_size, interpolation=cv2.INTER_AREA)

        return vectormap.astype(np.float16)

    @staticmethod
    def put_vectormap(vectormap, countmap, plane_idx, center_from, center_to, threshold=8):
        _, height, width = vectormap.shape[:3]

        vec_x = center_to[0] - center_from[0]
        vec_y = center_to[1] - center_from[1]
        try:
            min_x = max(0, int(min(center_from[0], center_to[0]) - threshold))
            min_y = max(0, int(min(center_from[1], center_to[1]) - threshold))
        except:
            print(center_from)
            print(center_to)

        max_x = min(width, int(max(center_from[0], center_to[0]) + threshold))
        max_y = min(height, int(max(center_from[1], center_to[1]) + threshold))

        norm = math.sqrt(vec_x ** 2 + vec_y ** 2)
        if norm == 0:
            return

        vec_x /= norm
        vec_y /= norm

        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                bec_x = x - center_from[0]
                bec_y = y - center_from[1]
                dist = abs(bec_x * vec_y - bec_y * vec_x)

                if dist > threshold:
                    continue

                countmap[plane_idx][y][x] += 1

                vectormap[plane_idx*2+0][y][x] = vec_x
                vectormap[plane_idx*2+1][y][x] = vec_y



class MPIMetadata(CocoMetadata):

    def __init__(self, idx, img_url, img_meta, annotations, objpos, scale, randpers, synthList, sigma, hyperparams=None, dataSet='MPI', left_foot=None,
                 right_foot=None, fullGraph=False, mixedDataset=False, volumetricMask=False, occlusionPrediction=False):

        self.__parts = 16
        if not(fullGraph):
            self.__vecs = list(zip(
                [1, 2, 3, 4, 2, 6, 7, 2, 15,  9, 10, 15, 12, 13],
                [2, 3, 4, 5, 6, 7, 8, 15, 9, 10, 11, 12, 13, 14]
            ))
        else:
            combs = list(itertools.combinations(np.arange(1,15),2))
            combs = np.array(combs) #just get all combinations without repetitions
            self.__vecs = list(zip(
            combs[:,0],
            combs[:,1]
            ))

        self.idx = idx
        self.img_url = img_url
        self.img = None
        self.mask = None
        self.sigma = sigma
        self.objpos = objpos
        self.scale = scale
        self.randpers = randpers
        self.mixedDataset = mixedDataset
        self.synthList = synthList
        self.heatMask = None
        self.pafMask = None
        self.volumetricMask = volumetricMask
        self.cropped_list = None
        self.occlusionPrediction = occlusionPrediction


        self.height = int(img_meta['height'])
        self.width = int(img_meta['width'])

        self.dataSet = dataSet
        self.hyperparams = hyperparams
        self.nr_joints = 0
        self.first_crop = True


        joint_list = []
        for ann in annotations:
            if np.sum(ann) == 0:
                continue

            kp = np.array(ann)

            xs = kp[:, 0]
            ys = kp[:, 1]
            vs = kp[:, 2]

            joint_list.append([(x, y, v) for x, y, v in zip(xs, ys, vs)])

        self.joint_list = []
        self.joint_visibility = []

        transform = list(zip(
            [9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 7],
            [9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 6]
        ))
        for prev_joint in joint_list:
            new_joint = []
            new_visibility = []
            for idx1, idx2 in transform:
                try:
                    j1 = prev_joint[idx1]
                    j2 = prev_joint[idx2]
                except:
                    print(prev_joint)

                if (j1[0] == 0 and j1[1] == 0) or (j2[0] == 0 or j2[1] == 0):
                    new_joint.append((-np.inf, -np.inf))
                    new_visibility.append(False)
                else:
                    new_joint.append(((j1[0] + j2[0]) / 2, (j1[1] + j2[1]) / 2))
                    new_visibility.append(j1[0]==j2[0])

            new_joint.append((-np.inf, -np.inf))
            self.joint_list.append(new_joint)
            self.joint_visibility.append(new_visibility)



    def get_heatmap(self, target_size):

        if self.occlusionPrediction:
            occMapMultiplyer = 2
        else:
            occMapMultiplyer = 1
        heatmap = np.zeros((self.__parts*occMapMultiplyer, self.height, self.width), dtype=np.float32)
        if self.volumetricMask:
            fullMask = np.zeros((self.__parts*occMapMultiplyer, self.height, self.width), dtype=np.float32)
        self.nr_joints = 0
        for i, joints in enumerate(self.joint_list):
            for idx, point in enumerate(joints):
                if self.cropped_list[i][idx] or ((point[0] < 0 or point[1] < 0) or (point[0]>=self.height and point[1]>=self.width)):
                    continue
                self.nr_joints += 1
                MPIMetadata.put_heatmap(heatmap, idx, point, self.sigma)
                if self.occlusionPrediction:
                    if not(self.joint_visibility[i]):
                        MPIMetadata.put_heatmap(heatmap, idx+self.__parts, point, self.sigma)
                if self.volumetricMask:
                    if self.synthList[i]:
                        MPIMetadata.put_heatmap(fullMask, idx, point, self.sigma)
                        self.nr_joints -= 1
                    if self.occlusionPrediction:
                        MPIMetadata.put_heatmap(fullMask, idx +self.__parts, point, self.sigma)
        if self.nr_joints == 0:
            self.nr_joints = 1

        heatmap = heatmap.transpose((1, 2, 0))
        if self.volumetricMask:
            fullMask = fullMask.transpose((1, 2, 0))


        heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)
        if self.volumetricMask:
            fullMask[:, :, -1] = np.clip(1- np.amax(fullMask, axis=2), 0.0, 1.0)

        if target_size:
            if np.shape(heatmap)[2] > 3:
                he = np.zeros((target_size[0], target_size[1], heatmap.shape[2]))
                if self.volumetricMask:
                    fuma = np.zeros((target_size[0], target_size[1], fullMask.shape[2]))
                for mapNr in range(heatmap.shape[2]):
                    he[:,:,mapNr] = cv2.resize(heatmap[:,:,mapNr],target_size, interpolation=cv2.INTER_AREA)
                    if self.volumetricMask:
                        fuma[:,:,mapNr] = cv2.resize(fullMask[:,:,mapNr], (target_size[0], target_size[1]), interpolation=cv2.INTER_AREA)
                heatmap = he
                if self.volumetricMask:
                    fullMask = fuma
            else:
                heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_AREA)
                if self.volumetricMask:
                    fullMask = cv2.resize(fullMask, (target_size[0], target_size[1]), interpolation=cv2.INTER_AREA)

        if self.volumetricMask:
            for i in range(fullMask.shape[2]):
                #th, fullMask[:,:,i] = cv2.threshold(fullMask[:,:, i], 30, 1, cv2.THRESH_BINARY)
                fullMask[:, :, i] = fullMask[:, :, i]>0.25
            # mask = mask > 0
            # only when using multiplication for masking
            fullMask = np.array(fullMask, dtype=bool)
            fullMask = np.logical_not(fullMask)
            fullMask = np.array(fullMask, dtype=np.float32)

            fullMask = fullMask.reshape(46, 46, fullMask.shape[2])
            self.heatMask = fullMask


        return heatmap.astype(np.float16)


    def get_nr_joints(self):
        return self.nr_joints

    def get_vectormap(self, target_size):
        vectormap = np.zeros((len(self.__vecs)*2, self.height, self.width), dtype=np.float32)
        pafMask = np.zeros((len(self.__vecs)*2, self.height, self.width), dtype=np.float32)
        countmap = np.zeros((len(self.__vecs)*2, self.height, self.width), dtype=np.int16)
        countmapMask = np.zeros((len(self.__vecs) * 2, self.height, self.width), dtype=np.int16)
        count = -1
        for i, joints in enumerate(self.joint_list):
            for plane_idx, (j_idx1, j_idx2) in enumerate(self.__vecs):
                j_idx1 -= 1
                j_idx2 -= 1
#
                center_from = joints[j_idx1]
                center_to = joints[j_idx2]
                if self.cropped_list[i][j_idx1] or self.cropped_list[i][j_idx2] or np.abs(center_from[0])==np.inf or np.abs(center_from[1]) == np.inf or np.abs(center_to[0])==np.inf or np.abs(center_to[1]) == np.inf:
                    continue

                MPIMetadata.put_vectormap(vectormap, countmap, plane_idx, center_from, center_to)
                if self.volumetricMask:
                    if self.synthList[i]:
                        MPIMetadata.put_vectormap(pafMask, countmapMask, plane_idx, center_from, center_to)

        vectormap = vectormap.transpose((1, 2, 0))
        if self.volumetricMask:
            pafMask = pafMask.transpose((1, 2, 0))
        nonzeros = np.nonzero(countmap)
        for p, y, x in zip(nonzeros[0], nonzeros[1], nonzeros[2]):
            if countmap[p][y][x] <= 0:
                continue
            vectormap[y][x][p*2+0] /= countmap[p][y][x]
            vectormap[y][x][p*2+1] /= countmap[p][y][x]
            if self.volumetricMask:
                pafMask[y][x][p*2+0] /= countmap[p][y][x]
                pafMask[y][x][p*2+1] /= countmap[p][y][x]


        if target_size:
            if np.shape(vectormap)[2] > 3:
                he = np.zeros((target_size[0], target_size[1], vectormap.shape[2]))
                pafMa = np.zeros((target_size[0], target_size[1], vectormap.shape[2]))
                for mapNr in range(vectormap.shape[2]):
                    he[:,:,mapNr] = cv2.resize(vectormap[:,:,mapNr],target_size, interpolation=cv2.INTER_AREA)
                    if self.volumetricMask:
                        pafMa[:,:,mapNr] = cv2.resize(pafMask[:,:,mapNr], (target_size[0], target_size[1]), interpolation=cv2.INTER_AREA)
                vectormap = he
                if self.volumetricMask:
                    pafMask = pafMa
            else:
                vectormap = cv2.resize(vectormap, target_size, interpolation=cv2.INTER_AREA)

        if self.volumetricMask:
            for i in range(pafMask.shape[2]):
                pafMask[:,:,i] = np.abs(pafMask[:,:,i])>0.25

            pafMask = np.array(pafMask, dtype=bool)
            pafMask = np.logical_not(pafMask)
            pafMask = np.array(pafMask, dtype=np.float32)

            pafMask = pafMask.reshape(46, 46, pafMask.shape[2])
            self.pafMask = pafMask

        return vectormap.astype(np.float16)


    def get_mask(self, target_size=None):
        mask = self.mask
        if target_size:
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_AREA)
        th, mask = cv2.threshold(mask, 30, 1, cv2.THRESH_BINARY)
        mask = np.array(mask,dtype=bool)
        mask = np.logical_not(mask)
        mask = np.array(mask, dtype=np.float32)


        mask = mask.reshape(46,46,1)

        if self.volumetricMask:
            mask = np.tile(mask, self.pafMask.shape[2]+self.heatMask.shape[2])
            fullMask = np.concatenate([self.heatMask, self.pafMask],axis=2)
            fullMask = np.array(fullMask,dtype=bool)
            mask = np.array(mask * fullMask, dtype=np.float32)

        return mask


class CocoPose(RNGDataFlow):
    @staticmethod
    def display_image(inp, heatmap, vectmap, as_numpy=False):
        global mplset
        mplset = True
        import matplotlib.pyplot as plt

        fig = plt.figure()
        a = fig.add_subplot(2, 2, 1)
        a.set_title('Image')
        plt.imshow(CocoPose.get_bgimg(inp))

        a = fig.add_subplot(2, 2, 2)
        a.set_title('Heatmap')
        plt.imshow(CocoPose.get_bgimg(inp, target_size=(heatmap.shape[1], heatmap.shape[0])), alpha=0.5)
        tmp = np.amax(heatmap, axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        tmp2 = vectmap.transpose((2, 0, 1))
        tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
        tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

        a = fig.add_subplot(2, 2, 3)
        a.set_title('Vectormap-x')
        plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        a = fig.add_subplot(2, 2, 4)
        a.set_title('Vectormap-y')
        plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        if not as_numpy:
            plt.show()
        else:
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            fig.clear()
            plt.close()
            return data

    @staticmethod
    def get_bgimg(inp, target_size=None):
        inp = cv2.cvtColor(inp.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if target_size:
            inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)
        return inp

    def __init__(self, path, img_path=None, is_train=True, decode_img=True, only_idx=-1):
        self.is_train = is_train
        self.decode_img = decode_img
        self.only_idx = only_idx

        if is_train:
            whole_path = os.path.join(path, 'person_keypoints_train2017.json')
        else:
            whole_path = os.path.join(path, 'person_keypoints_val2017.json')
        self.img_path = (img_path if img_path is not None else '') + ('train2017/' if is_train else 'val2017/')
        self.coco = COCO(whole_path)

        logger.info('%s dataset %d' % (path, self.size()))

    def size(self):
        return len(self.coco.imgs)

    def get_data(self):
        idxs = np.arange(self.size())
        if self.is_train:
            self.rng.shuffle(idxs)
        else:
            pass

        keys =   list(self.coco.imgs.keys())
        for idx in idxs:
            img_meta = self.coco.imgs[keys[idx]]
            img_idx = img_meta['id']
            ann_idx = self.coco.getAnnIds(imgIds=img_idx)

            if 'http://' in self.img_path:
                img_url = self.img_path + img_meta['file_name']
            else:
                img_url = os.path.join(self.img_path, img_meta['file_name'])

            anns = self.coco.loadAnns(ann_idx)
            meta = CocoMetadata(idx, img_url, img_meta, anns, sigma=8.0)

            total_keypoints = sum([ann.get('num_keypoints', 0) for ann in anns])
            if total_keypoints == 0 and random.uniform(0, 1) > 0.2:
                continue

            yield [meta]


class MPIIPose(RNGDataFlow):
    def __init__(self, path, img_path=None, is_train=True, decode_img=True, only_idx=-1, hyperparams=None, dataSet='MPI', construction_dict=None, picked_bins=None, gm_params=None, person_level=False, useBins=False, params=None, fullyConnected=False, mixedDataset=False, volumetricMask=False, occlusionPrediction=False):
        self.is_train = is_train
        self.decode_img = decode_img
        self.only_idx = only_idx
        self.img_path = img_path
        self.hyperparams = hyperparams
        self.dataSet = dataSet
        self.picked_bins = picked_bins
        self.gm_params = gm_params
        self.person_level = person_level
        self.useBins = useBins
        self.fullyConnected = fullyConnected
        self.mixedDataset = mixedDataset
        self.volumetricMask = volumetricMask
        self.occlusionPrediction = occlusionPrediction

        ####### build the construction dictionary#############################
        # with grouping parameters per person
        if construction_dict is None:
            whole_path = os.path.join(path)

            if os.path.isdir(whole_path):
                annolist_file = []
                for file in os.listdir(whole_path):
                    with open(os.path.join(whole_path,file), 'r') as infile:
                        try:
                            fi = json.load(infile)
                        except:
                            import ipdb
                            ipdb.set_trace()

                        fi = fi['root']
                        annolist_file = annolist_file + fi
            else:
                with open(whole_path, 'r') as infile:
                    annolist_file = json.load(infile)
                annolist_file = annolist_file['root']

            dataset,  cats, imgs = dict(),dict(),dict()
            imgToAnns, anns, objpos, scale, personIdx, synthList = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
            if not(picked_bins is None):
                camera_angle, median_dist, std_dist, median_scale, std_scale, nrOfPeople, visibility_ratio_total, cropped_ratio_total, minDist, cropped_person = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
                camera_angle_to_img, median_dist_to_img, std_dist_to_img, median_scale_to_img,std_scale_to_img, nrOfPeople_to_img, visibility_ratio_total_to_img, cropped_ratio_total_to_img, probs_per_bin, visibility_ratio_person = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list),defaultdict(list),defaultdict(list)


            #in json every person has own entry, here all annotations are added for each image
            for ann in annolist_file:
                try:
                    int(ann['isValidation'])
                except:
                    logger.warning('possibly empty entry found')
                    continue
                if mixedDataset:
                    isSynth = bool(ann['isSynth'])
                    if isSynth:
                        continue
                if (is_train and int(ann['isValidation']) == 0) or (not(is_train) and int(ann['isValidation']) == 1) :
                    if person_level:
                        if 'cropped_person' in ann.keys():
                            if mixedDataset:
                                if ann['objpos'][0]<=0 or ann['objpos'][1]<=0 or ann['objpos'][0]>=ann['img_width'] or ann['objpos'][1]>=ann['img_height']:
                                    continue
                            else:
                                if ann['objpos'][0] <= 0 or ann['objpos'][1] <= 0 or ann['objpos'][0] >= ann['img_width'] or ann['objpos'][1] >= ann['img_height'] or ann['visibility_person'] == 0 or ann['cropped_person'] == 1:
                                    continue

                        identifier = ann['img_paths']+'_'+str(ann['people_index'])
                        personIdx[identifier] = 0
                    else:
                        identifier = ann['img_paths']
                        personIdx[identifier] = None

                    imgToAnns[identifier].append(ann['joint_self'])
                    anns[identifier].append( ann['joint_self'] )
                    objpos[identifier].append( ann['objpos'])
                    scale[identifier].append(ann['scale_provided'])


                    if person_level:
                        if 'minDist' in ann.keys():
                            if not(isinstance(ann['scale_provided_other'], list)):
                                objposes, joints, scales = [ann['objpos_other']], [ann['joint_others']], [ann['scale_provided_other']]
                            else:
                                objposes, joints, scales = ann['objpos_other'], ann['joint_others'], ann['scale_provided_other']
                            if 'minDist' in ann.keys():
                                if isinstance(ann['minDist'],dict):
                                    # logger.info('no min distance found, maybe no other object in list?')
                                    # logger.info(ann['objpos_other'])
                                    continue
                            if mixedDataset:
                                areSynth_other = ann['isSynth_other']
                                synthList[identifier].append(False) #sample only real people.
                            if 'minDist' in ann.keys():
                                cropped_person[identifier].append(ann['cropped_person'])
                                minDist[identifier].append(ann['minDist'])

                            for i, (joints_other, scale_other) in enumerate(zip( joints, scales)):
                                if isinstance(ann['joint_others'], list):
                                    # order #joints x 3 joints from 0 to 15
                                    imgToAnns[identifier].append(joints_other)
                                    anns[identifier].append(joints_other)
                                    scale[identifier].append(scale_other)
                                if mixedDataset:
                                    if isinstance(ann['isSynth_other'], list):
                                        synthList[identifier].append(bool(ann['isSynth_other'][i][0]))
                                    else:
                                        print(ann['isSynth_other'])
                                        synthList[identifier].append(ann['isSynth_other'])
                            if len(np.shape(objposes)) > 1:
                                for position in objposes:
                                    objpos[identifier].append(position)
                            else:
                                objpos[identifier].append(objposes)
                        else: #must be real data
                            if isinstance(ann['objpos_other'], dict):
                                if mixedDataset:
                                    areSynth_other = ann['isSynth_other']
                                    synthList[identifier].append(False)
                            elif isinstance(ann['objpos_other'][0], list):
                                for i, (objposes, joints_other, scale_other) in enumerate(zip(ann['objpos_other'], ann['joint_others'], ann['scale_provided_other'])):
                                    # order #joints x 3 joints from 0 to 15
                                    imgToAnns[identifier].append(joints_other)
                                    anns[identifier].append(joints_other)
                                    scale[identifier].append(scale_other)
                                    if len(np.shape(objposes)) > 1:
                                        for position in objposes:
                                            objpos[identifier].append(position)
                                    else:
                                        objpos[identifier].append(objposes)
                                    if not(len(joints_other) ==16):
                                        for missJointAnn in range(16-len(joints_other)):
                                            imgToAnns[identifier][-1].append((0.0,0.0,0.0))
                                    if mixedDataset:
                                        synthList[identifier].append(ann['isSynth_other'][i])
                            elif isinstance(ann['objpos_other'], list):
                                # order #joints x 3 joints from 0 to 15
                                imgToAnns[identifier].append(ann['joint_others'])
                                anns[identifier].append(ann['joint_others'])
                                scale[identifier].append(ann['scale_provided_other'])
                                if len(np.shape(ann['objpos_other'])) > 1:
                                    for position in ann['objpos_other']:
                                        objpos[identifier].append(position)
                                else:
                                    objpos[identifier].append(ann['objpos_other'])
                                if not(len(ann['joint_others']) ==16):
                                    for missJointAnn in range(15-len(ann['joint_others'])):
                                        imgToAnns[identifier][-1].append((0.0,0.0,0.0))


                    if not (picked_bins is None):

                        camera_angle[identifier].append(ann['camera_angle'])
                        median_dist[identifier].append(ann['median_dist'])
                        std_dist[identifier].append(ann['std_dist'])
                        median_scale[identifier].append(ann['median_scale'])
                        std_scale[identifier].append(ann['std_scale'])
                        nrOfPeople[identifier].append(ann['nr_people'])
                        visibility_ratio_total[identifier].append(ann['visibility_ratio_total'])
                        visibility_ratio_person[identifier].append(ann['visibility_person'])
                        cropped_ratio_total[identifier].append(ann['cropped_ratio_total'])

                        bins_used = 0
                        if useBins:
                            probs_per_bin[identifier] = np.zeros_like(picked_bins, dtype=bool)
                        else:
                            probs_per_bin[identifier] = np.zeros_like(picked_bins, dtype=float)
                        keys = list(gm_params.keys())
                        keys.sort()
                        for key in gm_params:
                            if key == 'visibility_ratio_total' or key =='visiblility_ratio_total':
                                ann_key = 'visibility_person'
                            elif key == 'scale_person':
                                ann_key = 'scale_provided'
                            else:
                                ann_key = key

                            if ann[ann_key] == '_NaN_':
                                ann[ann_key] = 10000000  # just make sure it never gets sampled.
                            if useBins:
                                idx = np.digitize(ann[ann_key], gm_params[key]['means'][0:-1])
                                probs_per_bin[identifier][bins_used:len(gm_params[key]['means'])+bins_used][idx] = True
                                bins_used += len(gm_params[key]['means'])
                            else:
                                if not(isinstance(gm_params[key], dict)):
                                    probs_per_bin[identifier][bins_used:len(gm_params[key].means)+bins_used] = gaus_pdf(float(ann[ann_key]), gm_params[key].means, gm_params[key].var)
                                    bins_used += len(gm_params[key].means)
                                else:
                                    try:
                                        probs_per_bin[identifier][bins_used:len(gm_params[key]['means'])+bins_used] = gaus_pdf(float(ann[ann_key]), gm_params[key]['means'], gm_params[key]['vars'])
                                        bins_used += len(gm_params[key]['means'])
                                    except:
                                        print('whops')

                    if not(ann['img_paths'] in list(imgs.keys())):
                        img_meta = {}
                        img_meta['file_name'] = ann['img_paths']
                        img_meta['height'] = ann['img_height']
                        img_meta['width'] = ann['img_width']
                        img_meta['id'] = identifier
                        imgs[identifier] = img_meta

            self.key = list(imgs.keys())
            self.anns = anns
            self.imgs = imgs
            self.imgToAnns = imgToAnns
            self.objpos = objpos
            self.scale = scale
            self.personIdx = personIdx
            self.synthList = synthList

            if not(picked_bins is None):
                self.camera_angle_to_img = camera_angle_to_img
                self.median_dist_to_img = median_dist_to_img
                self.std_dist_to_img = std_dist_to_img
                self.median_scale_to_img = median_scale_to_img
                self.std_scale_to_img = std_scale_to_img
                self.nrOfPeople_to_img = nrOfPeople_to_img
                self.visibility_ratio_total_to_img = visibility_ratio_total
                self.visibility_ratio_person = visibility_ratio_person
                self.cropped_ratio_total_to_img = cropped_ratio_total_to_img

                self.cropped_person =  cropped_person
                self.minDist = minDist

                keys = list(self.key)


                probs = np.zeros((len(keys), len(picked_bins)))
                for imgnr, key in enumerate(keys):
                    probs[imgnr, :] = probs_per_bin[key]
                self.probs=probs

                if useBins:
                    self.probs = np.array(probs, dtype=bool)
        else:
            self.is_train = construction_dict['is_train']
            self.decode_img = construction_dict['decode_img']
            self.only_idx = construction_dict['only_idx']
            self.img_path = construction_dict['img_path']
            self.hyperparams = construction_dict['hyperparams']
            self.dataSet = construction_dict['dataSet']
            self.anns = construction_dict['anns']
            self.imgs = construction_dict['imgs']
            self.imgToAnns  = construction_dict['imgToAnns']
            self.objpos  = construction_dict['objpos']
            self.scale = construction_dict['scale']
            self.key = construction_dict['key']
            if self.person_level:
                self.personIdx = construction_dict['personIdx']

            if self.mixedDataset:
                self.synthList = construction_dict['synthList']

            if not(picked_bins is None):
                self.camera_angle_to_img = construction_dict['camera_angle_to_img']
                self.median_dist_to_img = construction_dict['median_dist_to_img']
                self.std_dist_to_img = construction_dict['std_dist_to_img']
                self.median_scale_to_img = construction_dict['median_scale_to_img']
                self.std_scale_to_img = construction_dict['std_scale_to_img']
                self.nrOfPeople_to_img = construction_dict['nrOfPeople_to_img']
                self.visibility_ratio_total_to_img = construction_dict['visibility_ratio_total_to_img']
                self.visibility_ratio_person = construction_dict['visibility_person']
                self.cropped_ratio_total_to_img = construction_dict['cropped_ratio_total_to_img']
                self.probs = construction_dict['probs']
                self.cropped_person = construction_dict['cropped_person']
                self.minDist = construction_dict['minDist']

        logger.info('%s dataset %d' % (path, self.size()))


    def get_dictionary(self):
        poseData = {'is_train':self.is_train,
                    'decode_img':self.decode_img,
                    'only_idx':self.only_idx,
                    'img_path':self.img_path,
                    'hyperparams':self.hyperparams,
                    'dataSet':self.dataSet,
                    'anns':self.anns,
                    'imgs':self.imgs,
                    'imgToAnns':self.imgToAnns,
                    'objpos':self.objpos,
                    'scale':self.scale,
                    'key':self.key}

        if self.person_level:
            poseData['personIdx'] = self.personIdx

        if not(self.picked_bins is None):
            poseData['camera_angle_to_img'] = self.camera_angle_to_img
            poseData['median_dist_to_img'] = self.median_dist_to_img
            poseData['std_dist_to_img'] = self.std_dist_to_img
            poseData['median_scale_to_img'] = self.median_scale_to_img
            poseData['std_scale_to_img'] = self.std_scale_to_img
            poseData['nrOfPeople_to_img'] = self.nrOfPeople_to_img
            poseData['visibility_ratio_total_to_img'] = self.visibility_ratio_total_to_img
            poseData['visibility_person'] = self.visibility_ratio_person
            poseData['cropped_ratio_total_to_img'] = self.cropped_ratio_total_to_img
            poseData['probs'] = self.probs
            poseData['minDist'] = self.minDist
            poseData['cropped_person'] = self.cropped_person

        if self.mixedDataset:
            poseData['synthList'] = self.synthList

        return poseData


    def get_combinations(self, bins1, bins2):
        params = [bins1, bins2]
        r = [[]]
        for x in params:
            r = [i + [y] for y in x for i in r]

        return r

    def size(self):
        return len(self.imgs)

    def size2(self,iterable):
        d = collections.deque(enumerate(iterable, 1), maxlen=1)
        return d[0][0] if d else 0

    def get_data(self):
        def _isArrayLike(obj):
            return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


        idxs = np.arange(self.size())

        keys = list(self.key)

        if not(self.picked_bins is None):
            if self.useBins:
                empty = True
                while empty:
                    inBin = np.squeeze( self.probs[:,self.picked_bins] )# is a boolean index if in bin or not if useBins is true
                    if len(np.shape(inBin)) > 1:
                        inBin = np.sum(inBin, axis=1)
                        inBin = inBin>0
                    if np.sum(inBin) <= 32:
                        logger.warning('bin was empty - detected in get '
                                       'data, try to resolve issue before, else picked does not correspond to bin really picked')
                        self.picked_bins = np.zeros_like(self.picked_bins, dtype=bool)
                        self.picked_bins[np.random.choice(np.arange(0,len(self.picked_bins)))] = True
                        self.empty = True
                    else:
                        empty = False
                idxs = idxs[inBin]
                self.rng.shuffle(idxs)

        else:
            self.rng.shuffle(idxs)


        for idx in idxs:
            if self.useBins:
                idx = np.random.choice(idxs)

            img_meta = self.imgs[keys[idx]]
            img_idx = img_meta['id']
            anns = self.imgToAnns[img_idx]
            objpos = self.objpos[img_idx]
            scale = self.scale[img_idx]
            randpers = self.personIdx[img_idx]
            if self.mixedDataset:
                synthList = self.synthList[img_idx]
            else:
                synthList = np.zeros((len(anns),), dtype=bool)

            if 'http://' in self.img_path:
                img_url = self.img_path + img_meta['file_name']
            else:
                if self.dataSet =='MPI_synth':
                    img_url = img_meta['file_name']
                else:
                    img_url = os.path.join(self.img_path, img_meta['file_name'])

            meta = MPIMetadata(idx, img_url, img_meta, anns, objpos, scale, randpers, synthList=synthList, sigma=7.0, hyperparams=self.hyperparams, fullGraph=self.fullyConnected, mixedDataset=self.mixedDataset, volumetricMask=self.volumetricMask,occlusionPrediction=self.occlusionPrediction)

            total_keypoints = sum([np.sum(ann) for ann in anns])
            if total_keypoints == 0 and random.uniform(0, 1) > 0.2:
                continue

            yield [meta]

def get_masklocation(url):
    parts = url.split('composition')
    url = parts[0]
    filenr = parts[-1]
    filenr = filenr.split('.')[0]
    filenr = filenr+'.txt'
    fi = url + 'img_path_BGround'+ filenr
    with open(fi,'r') as infile:
        imgname = infile.readline()
    imgname = 'mask_'+imgname
    return os.path.join(mpiMaskPath, imgname)

def gaus_pdf( X, means, vars):
    return np.exp(-(X-means)**2/(2*vars**2)) / (np.sqrt(2*math.pi * vars**2))

def read_image_url(metas):
    for meta in metas:
        img_str = None
        if 'http://' in meta.img_url:
            for _ in range(10):
                try:
                    resp = requests.get(meta.img_url)
                    if resp.status_code // 100 != 2:
                        logger.warning('request failed code=%d url=%s' % (resp.status_code, meta.img_url))
                        time.sleep(1.0)
                        continue
                    img_str = resp.content
                    break
                except Exception as e:
                    logger.warning('request failed url=%s, err=%s' % (meta.img_url, str(e)))
        else:
            img_str = open(meta.img_url, 'rb').read()
            if isinstance(meta, MPIMetadata):
                try:
                    if meta.mixedDataset:
                        mask_url = get_masklocation(meta.img_url)
                        loadMask = True
                    else:
                        p1, p2 = meta.img_url.split('images/')
                        mask_url = os.path.join(p1,'masks','mask_'+p2)
                        loadMask = True
                except:
                    mask_url = None

                    loadMask = False
        if not img_str :
            logger.warning('image not read, path=%s' % meta.img_url)
            raise Exception()

        nparr = np.fromstring(img_str, np.uint8)
        meta.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if isinstance(meta, MPIMetadata):
            if loadMask:
                im = cv2.imread(mask_url,cv2.IMREAD_UNCHANGED)
            else:
                im = np.zeros((np.shape(meta.img)[0], np.shape(meta.img)[1]))
            im = cv2.resize(im, (meta.width, meta.height), interpolation=cv2.INTER_AREA)
            th, threshed = cv2.threshold(im, 30, 255, cv2.THRESH_BINARY)
            meta.mask = threshed
    return metas


def get_dataflow(path, is_train, img_path=None, dataSet='COCO', hyperparams=None, construction_dict=None, picked_bins=None, gm_params=None, queuelength=50, person_level=False, useBins=False, params=None, synthTrain=None, fullyConnected=False, mixedDataset=False, volumetricMask=False, occlusionPrediction=False):
    if dataSet == 'COCO':
        ds = CocoPose(path, img_path, is_train)       # read data from lmdb
    else:
        ds = MPIIPose(path, img_path, is_train, hyperparams=hyperparams, dataSet=dataSet,construction_dict=construction_dict,picked_bins=picked_bins, gm_params=gm_params, person_level=person_level, useBins=useBins, params=params, fullyConnected=fullyConnected, mixedDataset=mixedDataset, volumetricMask=volumetricMask, occlusionPrediction=occlusionPrediction)

    if is_train:
        if dataSet == 'COCO':
            ds = MapData(ds, read_image_url)
            ds = MapDataComponent(ds, pose_random_scale)
            ds = MapDataComponent(ds, pose_rotation)
            ds = MapDataComponent(ds, pose_flip)
            ds = MapDataComponent(ds, pose_resize_shortestedge_random)
            ds = MapDataComponent(ds, pose_crop_random)
            ds = MapData(ds, pose_to_img)
        else:

            if not synthTrain is None:
                if synthTrain:
                    threads = 40
                    buffer = 10
                else:
                    threads = 8
                    buffer = 5
            else:
                threads = 32
                buffer = 10


            ds = MultiThreadMapData(ds, nr_thread=threads, map_func=read_image_url, buffer_size=buffer)
            ds = MapDataComponent(ds, pose_crop_person_center) #
            ds = MapDataComponent(ds, pose_cpm_original_scale)
            ds = MapDataComponent(ds, pose_rotation)
            ds = MapDataComponent(ds, pose_crop_person_center)
            ds = MapDataComponent(ds, pose_flip)
            ds = MapData(ds, pose_to_img)

        ds = PrefetchData(ds, queuelength, 3)
    else:
        if dataSet == 'COCO':
            ds = MultiThreadMapData(ds, nr_thread=1, map_func=read_image_url, buffer_size=500)
            ds = MapDataComponent(ds, pose_resize_shortestedge_fixed)
            ds = MapDataComponent(ds, pose_crop_center)
            ds = MapData(ds, pose_to_img)
        else:
            ds = MultiThreadMapData(ds, nr_thread=1, map_func=read_image_url, buffer_size=1)
            ds = MapDataComponent(ds, pose_cpm_original_scale)
            ds = MapDataComponent(ds, pose_crop_person_center)
            ds = MapData(ds, pose_to_img)

        ds = PrefetchData(ds, 4, 10)
    return ds


def _get_dataflow_onlyread(path, is_train, img_path=None, dataSet='COCO', validate=False, person_level=False):
    if dataSet == 'COCO':
        ds = CocoPose(path, img_path, is_train)
    else:
        ds = MPIIPose(path, img_path, is_train, person_level=person_level)
    ds = MapData(ds, read_image_url)
    if validate:
        ds = MapData(ds, pose_to_img)
    else:
        ds = MapData(ds, pose_to_img)
    return ds



def get_dataflow_batch(path, is_train, batchsize, img_path=None, dataSet='COCO', hyperparams=None, construction_dict=None,
                       picked_bins=None, gm_params=None,dataflow_queuelenth=100, person_level=True, use_bins=False, params=None,
                       synthTrain=None, fullyConnected=False, mixedData=False, volumetricMask=False, occlusionPrediction=False):
    logger.info('dataflow img_path=%s' % img_path)
    ds = get_dataflow(path, is_train, img_path=img_path, dataSet=dataSet, hyperparams=hyperparams, construction_dict=construction_dict, picked_bins=picked_bins, gm_params=gm_params,queuelength=dataflow_queuelenth, person_level=person_level, useBins=use_bins, params=params, synthTrain=synthTrain, fullyConnected=fullyConnected, mixedDataset=mixedData, volumetricMask=volumetricMask, occlusionPrediction=occlusionPrediction)
    ds = BatchData(ds, batchsize)
    if is_train:
        ds = PrefetchData(ds, 5, 4)
    else:

        ds = PrefetchData(ds, 1, 1)

    return ds


class DataFlowToQueue(threading.Thread):
    def __init__(self, ds, placeholders, queue_size=5):
        super().__init__()
        self.daemon = True

        self.ds = ds
        self.placeholders = placeholders
        self.queue = tf.FIFOQueue(queue_size, [ph.dtype for ph in placeholders], shapes=[ph.get_shape() for ph in placeholders])
        self.op = self.queue.enqueue(placeholders)
        self.close_op = self.queue.close(cancel_pending_enqueues=True)

        self._coord = None
        self._sess = None

        self.last_dp = None

    @contextmanager
    def default_sess(self):
        if self._sess:
            with self._sess.as_default():
                yield
        else:
            logger.warning("DataFlowToQueue {} wasn't under a default session!".format(self.name))
            yield

    def size(self):
        return self.queue.size()

    def start(self):
        self._sess = tf.get_default_session()
        super().start()

    def set_coordinator(self, coord):
        self._coord = coord

    def run(self):
        with self.default_sess():
            try:
                while not self._coord.should_stop():
                    try:
                        self.ds.reset_state()
                        while not self._coord.should_stop():
                            for dp in self.ds.get_data():
                                feed = dict(zip(self.placeholders, dp))

                                self.op.run(feed_dict=feed)
                                self.last_dp = dp
                                if self._coord.should_stop():
                                    logger.info('stop running')
                                    break
                    except (tf.errors.CancelledError, tf.errors.OutOfRangeError, DataFlowTerminated):
                        logger.info('err type1, placeholders={}'.format(self.placeholders))
                        #sys.exit(-1)
                    except Exception as e:
                        logger.error('err type2, err={}, placeholders={}'.format(str(e), self.placeholders))
                        if isinstance(e, RuntimeError) and 'closed Session' in str(e):
                            pass
                        else:
                            logger.exception("Exception in {}:{}".format(self.name, str(e)))
                        sys.exit(-1)
            except Exception as e:
                logger.exception("Exception in {}:{}".format(self.name, str(e)))
            finally:
                try:
                    self.close_op.run()
                    while not( self.queue.is_closed()):
                        logger.info('trying to close')
                        self.queue.close()

                except Exception:
                    pass
                logger.info("{} Exited.".format(self.name))

    def dequeue(self):
        return self.queue.dequeue()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    from tf_pose.pose_augment import set_network_input_wh, set_network_scale
    # set_network_input_wh(368, 368)
    set_network_input_wh(480, 320)
    set_network_scale(8)

    df = _get_dataflow_onlyread('/data/public/rw/coco/annotations', True, '/data/public/rw/coco/', 'MPII')

    from tensorpack.dataflow.common import TestDataSpeed
    TestDataSpeed(df).start()
    sys.exit(0)

    with tf.Session() as sess:
        df.reset_state()
        t1 = time.time()
        for idx, dp in enumerate(df.get_data()):
            if idx == 0:
                for d in dp:
                    logger.info('%d dp shape={}'.format(d.shape))
            print(time.time() - t1)
            t1 = time.time()
            CocoPose.display_image(dp[0], dp[1].astype(np.float32), dp[2].astype(np.float32))
            print(dp[1].shape, dp[2].shape)
            pass

    logger.info('done')
