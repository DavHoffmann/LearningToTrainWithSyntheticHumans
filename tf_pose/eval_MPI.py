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

import sys
import os
import numpy as np
import logging
import argparse
import json, re
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')


from networks import model_wh, get_graph_path
from scipy.io import loadmat, savemat
from estimator import TfPoseEstimator

import cv2
from scipy import array as array

import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

eval_size = -1


def round_int(val):
    return int(round(val))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
    parser.add_argument('--resize', type=str, default='0x0', help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=8.0, help='if provided, resize heatmaps before they are post-processed. default=8.0')
    parser.add_argument('--model', type=str, default='cmu', help='cmu_mpi / cmu / mobilenet_thin')
    parser.add_argument('--img-dir', type=str, default='./images')
    parser.add_argument('--datapath', type=str, default='./annotaions.json')
    parser.add_argument('--data-idx', type=int, default=-1)
    parser.add_argument('--data-set', type=str, default='MPI')
    parser.add_argument('--weight-path', type=str, default=None)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--visualize', type=int, default=0)
    parser.add_argument('--minival', type=int, default=0, help='0 for validation set, 1 for test set')
    parser.add_argument('--param-idx', type=int, default=-1)
    parser.add_argument('--multiscale', type=int, default=1)
    parser.add_argument('--group_validation', type=int, default=1)
    parser.add_argument('--fullGraph', type=bool, default=False)
    parser.add_argument('--basepath', type=str, default='./models_pb')
    parser.add_argument('--outpath', type=str, default='./results')
    parser.add_argument('--visOutputFolder', type=str, default='./vis')

    args = parser.parse_args()
    basepath = args.basepath
    multiscale = args.multiscale

    models = os.listdir(basepath)
    args.model = os.path.join(basepath,models[args.param_idx])

    print(args.data_set)
    print(args.minival)

    if args.minival ==0:
        path = './eval/annolist_fullVal_groups.mat'
        rectidxs = loadmat('./eval/rectidxs_multi_test_fullVal.mat', struct_as_record=False, squeeze_me=True)
        rectidxs = rectidxs['rectidxs_multi_train']
        annolist = loadmat(path, struct_as_record=False,squeeze_me=True)
        annolist = annolist['annolist_test_multi']
    elif args.minival ==1:
        path = './eval//annolist_test.mat'
        rectidxs = loadmat('./eval//rectidxs_multi_test.mat', struct_as_record=False, squeeze_me=True)
        rectidxs = rectidxs['rectidxs_multi_test']
        annolist = loadmat(path, struct_as_record=False,squeeze_me=True)
        annolist = annolist['annolist_test']

    crop_ratio      = 2.5 # 2
    bbox_ratio      = 0.25 # 0.5
    boxsize         = 368
    padvalue        = 128
    targetDist      = 41/35

    scale_search = np.array([0.7, 1, 1.3])

    vis=args.visualize
    model = args.model.split('/')[-1]
    model = model.split('.')[0]

    plt.ioff()
    w, h = model_wh(args.resize)
    if args.fullGraph == True:
        nrVectMaps= 91
    else:
        nrVectMaps = 28
    if w == 0 or h == 0:
        e = TfPoseEstimator(args.model, target_size=(368, 368), dataset=args.data_set, nr_vectmaps=nrVectMaps)
        w, h = 368, 368
    else:
        e = TfPoseEstimator(args.model, target_size=(w, h), dataset=args.data_set, nr_vectmaps=nrVectMaps)


    pred = []
    candidates = []
    nrimages = annolist.shape[0]
    for i in np.arange(0,nrimages):

        skipping = False
        imagePath   = os.path.join(args.img_dir, annolist[i].image.name)
        img_str = open(imagePath, 'rb').read()
        nparr       = np.fromstring(img_str, np.uint8)
        oriImg      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        rect    = annolist[i].annorect
        try:
            pos     = np.zeros((len(rect),2))
            scale   = np.zeros((len(rect),1))
        except:
            rect = [rect]
            pos     = np.zeros((len(rect),2))
            scale   = np.zeros((len(rect),1))

        for ridx in range(len(rect)):
            try:
                pos[ridx,:] = [rect[ridx].objpos.x, rect[ridx].objpos.y]
                scale[ridx] = rect[ridx].scale
            except:
                temppos = pos
                tempscale = scale
                pos = np.zeros((len(rect)-1, 2))
                scale = np.zeros((len(rect)-1, 1))
                pos[:ridx,:] = temppos[:ridx,:]
                tempscale[:ridx,:] = tempscale[:ridx,:]
            if not(args.minival == 4):
                try:
                    rect[ridx].annopoints.point
                except:
                    pred.append([])
                    print('Skipping: '+str(i))
                    skipping = True
                    break
        if skipping:
            continue

        minX = np.min(pos[:,0])
        minY = np.min(pos[:,1])
        maxX = np.max(pos[:,0])
        maxY = np.max(pos[:,1])

        zeroScaleIdx = scale != 0
        scale = scale[zeroScaleIdx]
        if np.sum(zeroScaleIdx) ==0:
            scale = [1]

        scale0 = targetDist/np.mean(scale)
        deltaX = boxsize/(scale0*crop_ratio)
        deltaY = boxsize/(scale0*2)

        bbox = np.zeros((4,))
        dX = deltaX * bbox_ratio
        dY = deltaY * bbox_ratio
        bbox[0] = int(np.round(np.max(minX-dX,0)))
        bbox[1] = int(np.round(np.max(minY-dY,0)))
        bbox[2] = int(np.round(np.min([maxX+dX,np.shape(oriImg)[1]])))
        bbox[3] = int(np.round(np.min([maxY+dY,np.shape(oriImg)[0]])))
        bbox = np.array(bbox, dtype=int)

        distMap = 0
        multiplier = scale_search * scale0
        pad      = np.zeros((1, len(multiplier)))
        ori_size = np.zeros((len(multiplier), 2))
        suc_flag = 0

        image_to_test = oriImg
        image_to_test = np.array(image_to_test / np.max(image_to_test), dtype=np.float32)




        result = []
        # inference the image with the specified network
        print(i)
        humans = e.inference(image_to_test, resize_to_default=False, upsample_size=args.resize_out_ratio,data_set='MPI', model=args.model, scales=multiplier, bbox=boxsize)
        if vis:
            fig, img_candidates = e.get_subplot(npimg=image_to_test, humans=humans, imgcopy=False,dataset='MPI',model=model,return_candidates=True)
        print(humans)
        truepred = np.zeros_like(e.peaks)



        if vis:
            minx= int(np.round(np.min(pos[:,0])))
            maxx = int(np.round(np.max(pos[:, 0])))
            miny = int(np.round(np.min(pos[:, 1])))
            maxy = int(np.round(np.max(pos[:, 1])))
            groups = [minx , maxx , miny , maxy ]

            candidates.append(img_candidates)
            model_name = model + '_multiscale'

            fig, img_candidates = e.get_subplot_drawingOnly(npimg=image_to_test, humans=humans, imgcopy=False, dataset='MPI', model=model, return_candidates=True, footOnly=False, path=os.path.join('./', model_name + '_nobox', str(i)),
                                                            model_name=model_name)
            if not(os.path.isdir(os.path.join(args.visOutputFolder,
                                              model_name+ '_nobox'))):
                os.mkdir(os.path.join(args.visOutputFolder,
                                      model_name + '_nobox'))
            plt.savefig(os.path.join(args.visOutputFolder,
                                     model_name + '_nobox',str(i)))
            plt.close('all')

        annorect = []
        for nrh, human in enumerate(humans):
            annopoints = []
            point = []
            for keypoint in list(human.body_parts.keys()):
                point.append((human.body_parts[keypoint].part_idx, human.body_parts[keypoint].x*oriImg.shape[1], human.body_parts[keypoint].y*oriImg.shape[0], human.body_parts[keypoint].score))

            annopoints.append(array( (array( point, dtype=[('id', 'O'), ('x', 'O'), ('y', 'O'),('score', 'O')]),),
                                      dtype=[('point', 'O')]) )
            annorect.append(annopoints)
        pred.append( annorect )



    savename = args.model.split('/')[-1]
    if not(os.path.isdir(args.outpath)):
        os.mkdir(args.outpath)
    savemat(os.path.join(args.outpath, savename.replace('pb','mat')), {'pred': pred})
    print('Done!')
