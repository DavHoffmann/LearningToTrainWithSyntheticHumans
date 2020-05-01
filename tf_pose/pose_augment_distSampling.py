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
import math
import random

import cv2
import numpy as np
from tensorpack.dataflow.imgaug.geometry import RotationAndCropValid

try:
    import pose_dataset_personLevel_minDist as pose_dataset
except:
    from tf_pose import pose_dataset_personLevel_minDist as pose_dataset

from tf_pose.common import CocoPart, MPIIPart
import logging

_network_w = 368
_network_h = 368
_scale = 8


logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger('pose_dataset')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def set_network_input_wh(w, h):
    global _network_w, _network_h
    _network_w, _network_h = w, h


def set_network_scale(scale):
    global _scale
    _scale = scale


def pose_random_scale(meta):
    scalew = random.uniform(0.8, 1.2)
    scaleh = random.uniform(0.8, 1.2)
    neww = int(meta.width * scalew)
    newh = int(meta.height * scaleh)

    dst = cv2.resize(meta.img, (neww, newh), interpolation=cv2.INTER_AREA)

    #adjust mask if given
    if isinstance(meta, pose_dataset.MPIMetadata):
        maskHight, maskWidth = meta.mask.shape
        maskneww = int(maskWidth * scalew)
        masknewh = int(maskHight * scaleh)
        msk = cv2.resize(meta.mask,(maskneww,masknewh), interpolation=cv2.INTER_AREA )
        th, msk = cv2.threshold(msk, 30, 255, cv2.THRESH_BINARY)
        meta.mask = msk

    newh = int(meta.height * scaleh)

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            adjust_joint.append((int(point[0] * scalew + 0.5), int(point[1] * scaleh + 0.5)))
        adjust_joint_list.append(adjust_joint)

    adjust_objPos = []
    future_randpers = meta.randpers
    for c, pos in enumerate(meta.objpos):
        if pos[0] < 0 or pos[1] < 0:
            if c <= meta.randpers:
                future_randpers -= 1
                logger.warning('lost person during init')
            continue
        newx, newy = int(pos[0] * scalew + 0.5), int(pos[1] *scaleh + 0.5)
        if not (newx > dst.shape[1] or newy > dst.shape[0] or newy<0 or newx<0):
            adjust_objPos.append([int(pos[0] * scalew + 0.5), int(pos[1] *scaleh + 0.5)])
        else:
            if c <= meta.randpers:
                future_randpers -= 1
            logger.warning('lost person during scale')

    meta.randpers = future_randpers
    meta.joint_list = adjust_joint_list
    meta.objpos = adjust_objPos
    meta.width, meta.height = neww, newh
    meta.img = dst
    return meta


def pose_resize_shortestedge_fixed(meta):
    ratio_w = _network_w / meta.width
    ratio_h = _network_h / meta.height
    ratio = max(ratio_w, ratio_h)

    return pose_resize_shortestedge(meta, int(min(meta.width * ratio + 0.5, meta.height * ratio + 0.5)))


def pose_resize_shortestedge_random(meta):
    ratio_w = _network_w / meta.width
    ratio_h = _network_h / meta.height
    ratio = min(ratio_w, ratio_h)
    target_size = int(min(meta.width * ratio + 0.5, meta.height * ratio + 0.5))
    random_scale = random.uniform(0.95, 1.6)
    target_size = int(target_size * random_scale)

    return pose_resize_shortestedge(meta, target_size)



def pose_resize_shortestedge(meta, target_size, mask_target_size=None):
    global _network_w, _network_h
    img = meta.img

    # adjust image
    scale = target_size / min(meta.height, meta.width)
    if meta.height < meta.width:
        newh, neww = target_size, int(scale * meta.width + 0.5)
    else:
        newh, neww = int(scale * meta.height + 0.5), target_size

    dst = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
    if isinstance(meta, pose_dataset.MPIMetadata):
        msk = cv2.resize(meta.mask, (neww, newh), interpolation=cv2.INTER_AREA)

    th, msk = cv2.threshold(msk, 30, 255, cv2.THRESH_BINARY)

    pw = ph = 0
    if neww < _network_w or newh < _network_h:
        pw = max(0, (_network_w - neww) // 2)
        ph = max(0, (_network_h - newh) // 2)
        mw = (_network_w - neww) % 2
        mh = (_network_h - newh) % 2
        color = random.randint(0, 255)
        dst = cv2.copyMakeBorder(dst, ph, ph+mh, pw, pw+mw, cv2.BORDER_CONSTANT, value=(128,128,128))
        if isinstance(meta, pose_dataset.MPIMetadata):
            msk = cv2.copyMakeBorder(msk, ph, ph + mh, pw, pw + mw, cv2.BORDER_CONSTANT, value=(0))

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            adjust_joint.append((int(point[0]*scale+0.5) + pw, int(point[1]*scale+0.5) + ph))
        adjust_joint_list.append(adjust_joint)

    adjust_objPos = []
    future_randpers = meta.randpers
    for c, pos in enumerate(meta.objpos):
        newx = int(pos[0]*scale+0.5) + pw
        newy = int(pos[1]*scale+0.5) + ph
        if not (newx > dst.shape[1] or newy > dst.shape[0] or newx<0 or newy<0):

            adjust_objPos.append([int(pos[0]*scale+0.5) + pw, int(pos[1]*scale+0.5) + ph])
        else:
            if c == meta.randpers:
                logger.warning('lost person during resize shortest')
            if c <= meta.randpers:
                future_randpers -= 1

    meta.randpers = future_randpers
    meta.objpos = adjust_objPos

    meta.joint_list = adjust_joint_list
    meta.objpos =  adjust_objPos
    meta.width, meta.height = neww + pw * 2, newh + ph * 2
    meta.img = dst
    if isinstance(meta, pose_dataset.MPIMetadata):
        meta.mask = msk
    return meta

def pose_cpm_original_scale(meta):
    if meta.hyperparams is None:
        scale_min = 0.5
        scale_max = 1.10000002384
        target_dist = 0.600000023842
    else:
        scale_min = meta.hyperparams['scale_min']
        scale_max = meta.hyperparams['scale_max']
        target_dist = meta.hyperparams['target_dist']

    if meta.first_crop:
        adjust_joint_list = []
        cropped = []
        for joint in meta.joint_list:
            adjust_joint = []
            cropped_for_person = []
            for point in joint:
                if point[0] ==0 and point[1] ==0:
                    adjust_joint.append((-np.inf, -np.inf))
                    cropped_for_person.append(True)
                    continue
                adjust_joint.append((point[0], point[1]))
                cropped_for_person.append(False)
            adjust_joint_list.append(adjust_joint)
            cropped.append(cropped_for_person)

        meta.cropped_list = cropped
        meta.joint_list = adjust_joint_list

    if not(isinstance(meta, pose_dataset.MPIMetadata)):
        return(pose_random_scale(meta))

    mask = meta.mask
    img = meta.img

    if not(meta.randpers is None):
        rand_pers = meta.randpers
    else:

        rand_pers = np.random.randint(0,np.shape(meta.objpos)[0])
        meta.randpers = rand_pers

    dice = np.random.rand()
    if dice > 1:
        scale_multiplier =1
    else:
        dice2 = np.random.rand()
        scale_multiplier = (scale_max - scale_min) * dice2+scale_min

    try:
        scale_abs = target_dist/meta.scale[rand_pers]
    except:
        logger.info(meta.scale)
        logger.info(meta.randpers)

    scale = scale_abs * scale_multiplier
    neww, newh = int(np.round(img.shape[1] * scale)), int(np.round(img.shape[0] * scale))
    img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_CUBIC)

    #adjust mask if given
    if isinstance(meta, pose_dataset.MPIMetadata):
        maskHight, maskWidth = meta.mask.shape
        maskneww = int(maskWidth * scale)
        masknewh = int(maskHight * scale)
        msk = cv2.resize(meta.mask,(maskneww,masknewh), interpolation=cv2.INTER_AREA )
        th, msk = cv2.threshold(msk, 30, 255, cv2.THRESH_BINARY)
        meta.mask = msk

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            adjust_joint.append((point[0] * scale , point[1] * scale))
        adjust_joint_list.append(adjust_joint)

    adjust_objPos = []
    future_randpers = meta.randpers
    for c, pos in enumerate(meta.objpos):
        try:
            if pos[0] < -100 or pos[1] < -100:
                if c == meta.randpers:
                    logger.warning('lost person during init')
                    logger.info(str(pos))
                if c <= meta.randpers:
                    future_randpers -= 1
                continue
        except:
            print(pos)
        newx = int(pos[0] * scale )
        newy = int(pos[1] *scale )
        if not(newx > img.shape[1] or newy > img.shape[0] or newx<0 or newy<0):
            adjust_objPos.append([int(pos[0] * scale ), int(pos[1] *scale )])
        elif c <= meta.randpers:
            future_randpers -=1
            logger.warning('lost person during scale cpm')
            logger.info(scale)
            logger.info(pos)
            logger.info([newx, newy])

    meta.randpers = future_randpers

    meta.joint_list = adjust_joint_list
    meta.objpos = adjust_objPos
    meta.img = img
    meta.width, meta.height = img.shape[1], img.shape[0]

    meta.first_crop = False # just in case first crop is not used, but meta.fist_crop is set to true by default
    return meta


def pose_crop_center(meta):
    global _network_w, _network_h
    target_size = (_network_w, _network_h)
    x = (meta.width - target_size[0]) // 2 if meta.width > target_size[0] else 0
    y = (meta.height - target_size[1]) // 2 if meta.height > target_size[1] else 0

    return pose_crop(meta, x, y, target_size[0], target_size[1])


def pose_crop_random(meta):

    global _network_w, _network_h
    target_size = (_network_w, _network_h)
    found = False
    for _ in range(50):
        x = random.randrange(0, meta.width - target_size[0]) if meta.width > target_size[0] else 0
        y = random.randrange(0, meta.height - target_size[1]) if meta.height > target_size[1] else 0

        # check whether any face is inside the box to generate a reasonably-balanced datasets
        if isinstance(meta, pose_dataset.MPIMetadata):
            for joint in meta.joint_list:
                if x <= joint[MPIIPart.Neck.value][0] < x + target_size[0] and y <= joint[MPIIPart.Neck.value][1] < y + target_size[1]:
                    logger.info('found a neck. Cropping around it')
                    found=True
                    break

        elif isinstance(meta, pose_dataset.CocoMetadata):
            for joint in meta.joint_list:
                if x <= joint[CocoPart.Nose.value][0] < x + target_size[0] and y <= joint[CocoPart.Nose.value][1] < y + target_size[1]:
                    found=True
                    break
    if found:
        return pose_crop(meta, x, y, target_size[0], target_size[1])
    else:
        logger.info('no neck found')
        return  get_center_without_objpos(meta)

def get_center_without_objpos(meta):
    joint_first_choice = []
    joint_second_choice = []
    joint_third_choice = []
    joint_last_choice = []

    for person in range(len(meta.joint_list)):
        for nr, joint in enumerate(meta.joint_list[person]):
            if nr == 14 and not(joint[0]<-100 or joint[1]<-100):
                joint_first_choice.append(joint)
            if nr == 1 and not(joint[0]<-100 or joint[1]<-100):
                joint_second_choice.append(joint)
            if nr == 0 and not(joint[0]<-100 or joint[1]<-100):
                joint_third_choice.append(joint)
            if not(joint[0]<-100 or joint[1]<-100):
                joint_last_choice.append(joint)

    if len(joint_first_choice)>0:
        dice = np.random.randint(0, len(joint_first_choice))
        return joint_first_choice[dice]
    elif len(joint_second_choice)>0:
        dice = np.random.randint(0, len(joint_second_choice))
        return joint_second_choice[dice]
    elif len(joint_third_choice)>0:
        dice = np.random.randint(0, len(joint_third_choice))
        return joint_third_choice[dice]
    elif len(joint_last_choice)>0:
        dice = np.random.randint(0, len(joint_last_choice))
        return joint_last_choice[dice]

    else:
        # may you never reach this point
        logger.warning('no one on image!')
        return [meta.height/2, meta.width/2]


def pose_crop_person_center(meta):

    global _network_w, _network_h
    img = meta.img
    mask = meta.mask

    target_size = (_network_w, _network_h)
    if meta.hyperparams is None:
        x_offset = np.random.randint(0,50)
        y_offset = np.random.randint(0,50)
    else:
        x_offset = np.random.randint(0, meta.hyperparams['center_perterb_max'])
        y_offset = np.random.randint(0, meta.hyperparams['center_perterb_max'])

    if meta.first_crop:
        target_size = (meta.width, meta.height)
        x_offset = 0
        y_offset = 0

    if meta.first_crop:
        adjust_joint_list = []
        cropped = []
        for joint in meta.joint_list:
            adjust_joint = []
            cropped_for_person = []
            for point in joint:
                if point[0] ==0 and point[1] ==0:
                    print(str(point[0])+'_'+str(point[1]))
                    adjust_joint.append((-np.inf, -np.inf))
                    cropped_for_person.append(True)
                else:
                    adjust_joint.append((point[0], point[1]))
                    cropped_for_person.append(False)
            adjust_joint_list.append(adjust_joint)
            cropped.append(cropped_for_person)

        meta.cropped_list = cropped
        meta.joint_list = adjust_joint_list


    crop_random = False
    if not(isinstance(meta, pose_dataset.MPIMetadata)):
        crop_random = True
        logger.warning('no person center in image. try to find neck instead')
        logger.warning('this might also be caused by wrong import. ')

    if np.shape(meta.objpos)[0] == 0:
        crop_random = True
        print(meta.objpos)
        logger.warning('no person center in image. try to find neck instead')
        logger.info(meta.first_crop)


    if crop_random:
        center = get_center_without_objpos(meta)
    else:
        if not(meta.randpers is None):
            if len(meta.objpos)-1 < meta.randpers:

                logger.warning('random person got out of image by augmentation. Choose a different one:')
                logger.info(meta.first_crop)
                meta.randpers = np.random.randint(0,np.shape(meta.objpos)[0])

            rand_pers = meta.randpers
            if len( meta.objpos[rand_pers]) < 2:
                center = get_center_without_objpos(meta)
                logger.warning('get_center_without_objpos')
            else:
                if meta.objpos[rand_pers][0] < 0 or meta.objpos[rand_pers][1] < 0:
                    candids = []
                    candidIdx = np.zeros((len(meta.objpos,)))
                    logger.warning('random person got out of image by augmentation_2. Choose a different one:')
                    for i in range(len(meta.objpos)):
                        if not(meta.objpos[i][0] < 0 or meta.objpos[i][1] < 0):
                            candids.append(i)
                            candidIdx[i] = 1
                    if len(candids)==0:
                        logger.warning('return pose_crop_center!!!!')
                        return pose_crop_center(meta)
                    else:
                        rand_pers = np.random.randint(0, len(candids))
                        rand_persIdx = np.where(candidIdx)
                        meta.randpers = rand_persIdx[0][rand_pers]
                        rand_pers = meta.randpers
        else:
            meta.randpers = np.random.randint(0,len(meta.objpos))
            rand_pers = meta.randpers

        center = meta.objpos[meta.randpers]
        #add random permutation to image center
        center = [center[0] + x_offset, center[1] + y_offset]
        offset_left = (center[0] - target_size[0]/2)
        offset_up = (center[1] - target_size[1]/2)
        img_save = meta.img
        if offset_left < 0:
            x_pos1 = 0
            x_pad1 = int(np.floor(- offset_left))
        else:
            x_pos1 = int(np.floor(offset_left))
            x_pad1 = 0
        if center[0] + target_size[0]/2 > img.shape[1]:
            x_pos2 = img.shape[1]
            x_pad2 = - int(np.ceil(img.shape[1] - (center[0] + target_size[0]/2)))
        else:
            x_pos2 = int(np.ceil(center[0] + target_size[0]/2))
            x_pad2 = 0

        if offset_up < 0:
            y_pos1 = 0
            y_pad1 = int(np.floor(- offset_up))
        else:
            y_pos1 = int(np.floor(offset_up))
            y_pad1 = 0
        if center[1] + target_size[1]/2 > img.shape[0]:
            y_pos2 = img.shape[0]
            y_pad2 = - int(np.ceil(img.shape[0] - (center[1] +target_size[1]/2)))
        else:
            y_pos2 = int(np.ceil(center[1] + target_size[1]/2))
            y_pad2 = 0

        img = img[y_pos1:y_pos2, x_pos1:x_pos2, :]

        if not(meta.first_crop) and (np.shape(img)[0] > target_size[1] or np.shape(img)[1] > target_size[0]):
            logger.info('warning image is not cropped correctly')
            logger.info(np.shape(img))
        mask = mask[y_pos1:y_pos2, x_pos1:x_pos2]
        x_padleft = int(np.floor(( x_pad1)))
        y_padup = int(np.floor((y_pad1)))
        x_padright = int(np.ceil(x_pad2))
        y_padbottom = int(np.ceil((y_pad2)))

        # adjust meta data
        adjust_joint_list = []
        for joint in meta.joint_list:
            adjust_joint = []
            for point in joint:
                new_x, new_y = point[0] - x_pos1 + x_padleft, point[1] - y_pos1 + y_padup
                adjust_joint.append((new_x, new_y))
            adjust_joint_list.append(adjust_joint)

        meta.joint_list = adjust_joint_list


        img = cv2.copyMakeBorder(img, y_padup, y_padbottom, x_padleft, x_padright, cv2.BORDER_CONSTANT, value=(128,128,128))
        if isinstance(meta, pose_dataset.MPIMetadata):
            mask = cv2.copyMakeBorder(mask, y_padup, y_padbottom, x_padleft, x_padright, cv2.BORDER_CONSTANT, value=255)#cv2.BORDER_CONSTANT) #masks with zeros
            meta.mask = mask

        adjust_objPos = []
        randpers_future = meta.randpers
        for c,pos in enumerate(meta.objpos):
            new_x, new_y = pos[0] - x_pos1 + x_padleft, pos[1] - y_pos1 + y_padup
            if not (new_x > img.shape[1] or new_y > img.shape[0] or new_x< 0 or new_y < 0):
                adjust_objPos.append([new_x, new_y])
            elif c <=meta.randpers:
                if meta.first_crop and meta.randpers ==c:
                    logger.warning('lost person during crop')
                elif meta.randpers ==c:
                    logger.warning('lost person during last crop')
                randpers_future -= 1
        meta.randpers = randpers_future
        meta.objpos = adjust_objPos


        meta.width, meta.height = target_size[0], target_size[1]
        meta.img = img
        if not(meta.first_crop) and not(str(np.shape(meta.img)) == '(368, 368, 3)'):
            logger.info('img shape= '+str(np.shape(meta.img)))
            logger.info(str(x_padright)+str(x_padleft)+str(y_padup)+str(y_padbottom))
        meta.first_crop = False
        return meta


def pose_crop(meta, x, y, w, h):
    # adjust image
    target_size = (w, h)

    img = meta.img
    resized = img[y:y+target_size[1], x:x+target_size[0], :]
    if isinstance(meta, pose_dataset.MPIMetadata):
        meta.mask = meta.mask[y:y+target_size[1], x:x+target_size[0]]

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            new_x, new_y = point[0] - x, point[1] - y
            adjust_joint.append((new_x, new_y))
        adjust_joint_list.append(adjust_joint)
    meta.joint_list = adjust_joint_list

    adjust_objPos = []
    future_ranpers = meta.randpers
    for c, pos in enumerate(meta.objpos):
        new_x, new_y = pos[0] - x, pos[1] - y
        if not(new_x > resized.shape[1] or new_y > resized.shape[0] or new_x<0 or new_y<0):
            adjust_objPos.append([new_x, new_y])
        elif c <= meta.randpers:
            logger.warning('lost person during pose_crop')
            future_ranpers -= 1
    meta.randpers = future_ranpers
    meta.objpos = adjust_objPos
    meta.width, meta.height = target_size
    meta.img = resized

    return meta


def pose_flip(meta):
    r = random.uniform(0, 1.0)
    if r > 0.5:
        return meta

    img = meta.img
    img = cv2.flip(img, 1)


    if isinstance(meta, pose_dataset.MPIMetadata):
        meta.mask = cv2.flip(meta.mask, 1)
        flip_list = [MPIIPart.Head, MPIIPart.Neck, MPIIPart.LShoulder, MPIIPart.LElbow, MPIIPart.LWrist, MPIIPart.RShoulder, MPIIPart.RElbow, MPIIPart.RWrist, MPIIPart.LHip, MPIIPart.LKnee,
                     MPIIPart.LAnkle, MPIIPart.RHip, MPIIPart.RKnee, MPIIPart.RAnkle, MPIIPart.Center]

    elif isinstance(meta, pose_dataset.CocoMetadata):
        # flip meta
        flip_list = [CocoPart.Nose, CocoPart.Neck, CocoPart.LShoulder, CocoPart.LElbow, CocoPart.LWrist, CocoPart.RShoulder, CocoPart.RElbow, CocoPart.RWrist,
                     CocoPart.LHip, CocoPart.LKnee, CocoPart.LAnkle, CocoPart.RHip, CocoPart.RKnee, CocoPart.RAnkle,
                     CocoPart.LEye, CocoPart.REye, CocoPart.LEar, CocoPart.REar, CocoPart.Background]


    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for cocopart in flip_list:
            point = joint[cocopart.value]
            adjust_joint.append((meta.width - point[0], point[1]))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list

    adjust_objPos = []
    future_randpers = meta.randpers
    for c, pos in enumerate(meta.objpos):
        newx = meta.width - pos[0]
        newy = pos[1]
        if not(newx > img.shape[1] or newy > img.shape[0] or newy<0 or newx<0):
            adjust_objPos.append([meta.width - pos[0], pos[1]])
        elif c <= meta.randpers:
            future_randpers -= 1
            logger.warning('lost person during flip')
    meta.randpers = future_randpers
    meta.objpos = adjust_objPos

    meta.img = img
    return meta


def pose_rotation(meta):
    if meta.hyperparams is None:
        max_rotate_degree = 40#random.uniform(-15.0, 15.0)
    else:
        max_rotate_degree = meta.hyperparams['max_rotate_degree']
    deg = random.uniform(-40.0, 40.0)
    img = meta.img

    center = (img.shape[1] * 0.5, img.shape[0] * 0.5)       # x, y
    rot_m = cv2.getRotationMatrix2D(center, deg, 1)

    if not isinstance(meta, pose_dataset.MPIMetadata):
        ret = cv2.warpAffine(img, rot_m, img.shape[1::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        neww, newh = RotationAndCropValid.largest_rotated_rect(ret.shape[1], ret.shape[0], deg)
        neww = min(neww, ret.shape[1])
        newh = min(newh, ret.shape[0])
        newx = int(center[0] - neww * 0.5)
        newy = int(center[1] - newh * 0.5)
        img = ret[newy:newy + newh, newx:newx + neww]
    else:
        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rot_m[0, 0])
        abs_sin = abs(rot_m[0, 1])

        # find the new width and height bounds
        bound_w = int(img.shape[0] * abs_sin + img.shape[1] * abs_cos)
        bound_h = int(img.shape[0] * abs_cos + img.shape[1] * abs_sin)

        # subtract old image center (bringing image back to orin) and adding
        #  the new image center coordinates
        rot_m[0, 2] += bound_w / 2 - center[0]
        rot_m[1, 2] += bound_h / 2 - center[1]

        ret = cv2.warpAffine(img, rot_m, (bound_w,bound_h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]

        img = ret
        neww, newh = ret.shape[1], ret.shape[0]
        newx = int(center[0] - neww * 0.5)
        newy = int(center[1] - newh * 0.5)



    #adjust mask
    if isinstance(meta, pose_dataset.MPIMetadata):
        msk = cv2.warpAffine(meta.mask, rot_m, (bound_w,bound_h), flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT, borderValue=255)#borderMode=cv2.BORDER_CONSTANT)
        meta.mask = msk

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            if np.abs(point[0])==np.inf or np.abs(point[1])==np.inf:
                x=np.inf
                y=np.inf
            else:
                x, y = _rotate_coord((meta.width, meta.height), (newx, newy), point, deg)
            adjust_joint.append((x, y))
        adjust_joint_list.append(adjust_joint)

    adjust_objPos = []
    future_randpers = meta.randpers
    for c, pos in enumerate(meta.objpos):
        x, y = _rotate_coord((meta.width, meta.height), (newx, newy), pos, deg)
        if not (x > img.shape[1] or y > img.shape[0] or x<0 or y<0):
            adjust_objPos.append([x, y])
        elif c <= meta.randpers:
            future_randpers -= 1
            logger.warning('lost person during rotate')
    meta.randpers = future_randpers
    meta.objpos = adjust_objPos
    meta.joint_list = adjust_joint_list
    meta.width, meta.height = neww, newh
    meta.img = img

    return meta


def _rotate_corners(shape,  point, angle):
    angle = -1 * angle / 180.0 * math.pi

    ox, oy = shape
    px, py = point[0,:], point[1,:]

    px = np.array(px)
    py = np.array(py)

    ox /= 2
    oy /= 2

    qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return int(qx), int(qy)


def _rotate_coord(shape, newxy, point, angle):
    angle = -1 * angle / 180.0 * math.pi

    ox, oy = shape
    px, py = point

    ox /= 2
    oy /= 2

    qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    new_x, new_y = newxy

    qx += ox - new_x
    qy += oy - new_y

    try:
        return int(qx + 0.5), int(qy + 0.5)
    except:
        print(qx)
        print(qy)


def pose_to_img(meta_l):
    global _network_w, _network_h, _scale

    return [
        meta_l[0].img.astype(np.float16),
        meta_l[0].get_heatmap(target_size=(_network_w // _scale, _network_h // _scale)),
        meta_l[0].get_vectormap(target_size=(_network_w // _scale, _network_h // _scale))
      , meta_l[0].get_mask(target_size=(_network_w // _scale, _network_h // _scale)),
        meta_l[0].get_nr_joints()]



def pose_to_img_validate(meta_l):
    global _network_w, _network_h, _scale

    return [
        meta_l[0].img.astype(np.float16),
        meta_l[0].get_heatmap(),
        meta_l[0].get_vectormap()
      , meta_l[0].get_mask()
    ]
