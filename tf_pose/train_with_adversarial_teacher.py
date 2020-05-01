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
# modified from https://github.com/ildoonet/tf-pose-estimation/

import matplotlib as mpl
mpl.use('Agg')

import argparse
import logging
import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from pose_augment_distSampling import set_network_input_wh, set_network_scale
from pose_dataset_personLevel_minDist import get_dataflow_batch, DataFlowToQueue, CocoPose, CocoMetadata, MPIMetadata, MPIIPose
from networks import get_network
import training_util as tut

logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def compute_gt_teacher(probabilities, picked, alpha, beta, teacher_loss_type, loss_last, loss_current, nbins_nrs, nbins):
    print(probabilities)
    print(picked)
    print(loss_last)
    print(loss_current)
    picked_probs_vect = np.repeat(probabilities[picked], nbins_nrs[picked] - 1)

    if loss_last <= loss_current:
        probabilities[picked] = probabilities[picked] + probabilities[picked] * alpha
        probabilities[np.logical_not(picked)] = probabilities[np.logical_not(picked)] - (picked_probs_vect * alpha) / nbins_nrs[np.logical_not(picked)]
    else:
        probabilities[picked] <= probabilities[picked] - probabilities[picked] * beta
        probabilities[np.logical_not(picked)] = probabilities[np.logical_not(picked)] + (picked_probs_vect * beta) / nbins_nrs[np.logical_not(picked)]

    nbins_last = 0
    for nbin in nbins:
        probabilities[nbins_last:nbin+nbins_last] = probabilities[nbins_last:nbin+nbins_last]/sum(probabilities)
        nbins_last+=nbin
    print('return')

    return probabilities




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--model', default='cmu_mpi_vgg_interSup', help='model name')
    parser.add_argument('--datapath', type=str, default='./MPI.json', help='path to json file')
    parser.add_argument('--synth_data_path', type=str, default='./purelySynthetic/', help='path to folder containing json files')
    parser.add_argument('--imgpath', type=str, default='./imagesMPI')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--epochs-till-restart', type=int, default=5)
    # paths
    parser.add_argument('--modelpath', type=str, default='./models/cmu_mpi')
    parser.add_argument('--logpath', type=str, default='./logfiles')
    parser.add_argument('--checkpoint', type=str, default='./models/')
    parser.add_argument('--tmpNpyPath', type=str, default='./tmpNpy/')
    parser.add_argument('--tmpgtPath', type=str, default='./tmpgt/')
    parser.add_argument('--gm_path', type=str, default='./tf_pose/groupBoundaries.npy')

    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)

    # optimizer
    parser.add_argument('--lr', type=str, default='0.00005')
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--beta1', type=str, default=0.8)
    parser.add_argument('--beta2', type=str, default=0.999)
    parser.add_argument('--param-idx', type=int, default=None) # 0 for minDist grouping, 1 for cameraAngle
    parser.add_argument('--decay-steps', type=int, default=20000)
    parser.add_argument('--decay_rate', type=int, default=0.33)

    # data augmentation
    parser.add_argument('--scale_min', type=float, default=0.4)
    parser.add_argument('--scale_max', type=float, default=1.6)
    parser.add_argument('--target_dist', type=float, default=1.25)
    parser.add_argument('--center_perterb_max', type=int, default=50)
    parser.add_argument('--max_rotate_degree', type=float, default=45)

    #teacher
    parser.add_argument('--initial_student', type=str, default='./models/baselineIntermediate/-70002')
    parser.add_argument('--identifier', type=str, default='')

    #dataset
    parser.add_argument('--data-set', type=str, default='MPII', help='MPII')
    parser.add_argument('--synth_real_split', type=list, default=[2, 2])
    parser.add_argument('--useBins', type=int, default=1)
    parser.add_argument('--finetune', type=int, default=1)
    parser.add_argument('--mixed_data', type=bool, default=False)
    parser.add_argument('--volumetricMask', type=bool, default=False)
    parser.add_argument('--stylized', type=bool, default=False)

    args = parser.parse_args()
    args.useBins = True #other values not supported
    args.finetune = bool(args.finetune)

    # define minimal number of samples per bin
    minimalNrOfSamples = args.batchsize
    #flag to optimize thread numbers in dataflows
    synthTrainThreads = True

    #define factor to determine sampling
    if args.param_idx<1:
        teacher_parameter_list =  ['minDist']
    elif args.param_idx < 2:
        teacher_parameter_list = ['camera_angle']

    steps_till_teacher_training = 20

    #determine the size of output layer of teacher
    nbins_all = np.array([10, 10])
    nbins = []
    param_all = ['camera_angle', 'minDist']
    for i, parameter in enumerate(param_all):
        if parameter in teacher_parameter_list:
            nbins.append(nbins_all[i])
    output_size = nbins[0]
    nbins = np.array(nbins)
    bins = np.zeros((len(teacher_parameter_list), sum(nbins)))
    picked = np.zeros((sum(nbins, )))
    nbins_nrs = []

    probabilities = []
    for n in range (len(nbins)):
        nbins_nrs = np.concatenate([nbins_nrs, np.repeat([nbins[n]], nbins[n])], axis=0)
        probabilities = np.concatenate([probabilities, np.repeat([1 / nbins[n]], nbins[n])], axis=0)

    nbins_nrs = np.array(nbins_nrs, dtype=int)
    probabilities = np.array(probabilities)
    teacher_loss_history = np.zeros((2, 10))
    random_sampling_loss_history = np.zeros((10,))
    total_teacher_loss_arr = 0
    picked_bins = {}

    loss_old = 10000
    teacher_gt = None


    # define input placeholder
    set_network_input_wh(args.input_width, args.input_height)
    scale = 4

    # build a dictionary for data augmentation hyper parameters
    augmentation_hyperparams = {}
    augmentation_hyperparams['scale_min'] = args.scale_min
    augmentation_hyperparams['scale_max'] = args.scale_max
    augmentation_hyperparams['target_dist'] = args.target_dist
    augmentation_hyperparams['center_perterb_max'] = args.center_perterb_max
    augmentation_hyperparams['max_rotate_degree'] = args.max_rotate_degree


    #for adversarial teacher
    alpha = 0.01
    beta = 0.01
    teacher_loss_type = 2
    artificial_penalty = [1, 0] #needed when empty group is selected


    # define the network
    if args.model in ['cmu', 'vgg', 'mobilenet_thin', 'mobilenet_try', 'mobilenet_try2',
                      'mobilenet_try3', 'hybridnet_try', 'cmu_mpi', 'cmu_mpi_vgg',
                      'cmu_mpi_vgg_interSup']:
        scale = 8


    logger.info('define model+')
    set_network_scale(scale)
    output_w, output_h = args.input_width // scale, args.input_height // scale
    with tf.device(tf.DeviceSpec(device_type="CPU")):
        input_node = tf.placeholder(tf.float32, shape=(args.batchsize, args.input_height, args.input_width, 3), name='image')

        if (args.data_set == 'MPII'):
            nr_keypoints = 16
            nr_vectmaps = 28  #
            if args.volumetricMask:
                mask_node = tf.placeholder(
                    tf.float32,
                    shape=(args.batchsize,  output_h,
                           output_w,
                           nr_vectmaps+nr_keypoints),
                    name='mask')
            else:
                mask_node = tf.placeholder(tf.float32,
                                           shape=(args.batchsize,
                                                  output_h,
                                                  output_w, 1),
                                           name='mask')

        vectmap_node = tf.placeholder(tf.float32,
                                      shape=(args.batchsize,
                                             output_h,
                                             output_w,
                                             nr_vectmaps),
                                      name='vectmap')
        heatmap_node = tf.placeholder(tf.float32,
                                      shape=(args.batchsize,
                                             output_h,
                                             output_w,
                                             nr_keypoints),
                                      name='heatmap')


        # define placeholders for openPose and teacher
        if args.data_set == 'MPII':
            # batshsize * imgsize*imgsize *channel
            q_inp = tf.placeholder(dtype=tf.float32,
                                   shape=(None, None, None, 3))
            q_heat = tf.placeholder(dtype=tf.float32,
                                    shape=(args.batchsize,46,46,nr_keypoints))
            q_vect = tf.placeholder(dtype=tf.float32,
                                    shape=(args.batchsize,46,46,nr_vectmaps))

            if args.volumetricMask:
                q_mask = tf.placeholder(
                    dtype=tf.float32,
                    shape=(args.batchsize, 46, 46, nr_keypoints+nr_vectmaps))
            else:
                q_mask = tf.placeholder(dtype=tf.float32,
                                        shape=(args.batchsize, 46, 46, 1))

            q_nr_joints = tf.placeholder(dtype=tf.float32,
                                         shape=(args.batchsize,))
            loss_tmOne = tf.placeholder(dtype=tf.float32,
                                        shape=(),
                                        name='loss_t-1')
            loss_t = tf.placeholder(dtype=tf.float32,
                                    shape=(),
                                    name='loss_t')
            probs_old = tf.placeholder(dtype=tf.float32,
                                       shape=np.shape(probabilities))
            picked_probs_old = tf.placeholder(dtype=tf.float32,
                                              shape=np.shape(probabilities))
            gt_teacher = tf.placeholder(tf.float32, np.shape(probabilities))
            teach_inp = tf.placeholder(dtype=tf.float32,
                                       shape=(args.batchsize, 368, 368, 3))
            loss_scale = tf.placeholder(dtype=tf.float32,
                                        shape=(), name='loss_scale')
        else:
            logger.info('not supported')
            exit(-1)


    # define model for multi-gpu (not tested for more than 1 GPU)

    q_inp_split, q_heat_split, q_vect_split, q_mask_split, q_nr_joints_split, \
    teach_inp_split = tf.split(q_inp, args.gpus), tf.split(q_heat, args.gpus), \
                      tf.split(q_vect,args.gpus), tf.split(q_mask, args.gpus), \
                      tf.split(q_nr_joints, args.gpus),\
                      tf.split(teach_inp, args.gpus)



    ###################set up networks and define losses #####################
    output_vectmap = []
    output_heatmap = []
    losses = []
    losses_masked = []
    last_losses_l1 = []
    last_losses_l2 = []
    last_losses_l1_masked = []
    last_losses_l2_masked = []
    outputs = []

    for gpu_id in range(args.gpus): #not tested for more than 1
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            # set up teacher network
            with tf.variable_scope('teacher', reuse=(gpu_id > 0)):
                vgg2, pretrain_path_vgg2, vgg_4_2_2 = get_network('vgg_conv4_2', teach_inp_split[gpu_id])
                vgg_features_2 = vgg2.get_output('conv4_2')
                teacherNet, pretrain_path_teacher, teacherLastLayer = get_network('vgg_teacher',
                                                                                  vgg_features_2,
                                                                                  outputsize=output_size)


            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                if args.finetune:
                    net, pretrain_path, last_layer = get_network(args.model,
                                                                 q_inp_split[gpu_id], trainVgg=False)
                else:
                    vgg, pretrain_path_vgg, vgg_4_2 = get_network('vgg_conv4_2', q_inp_split[gpu_id])
                    vgg_features = vgg.get_output('conv4_2')
                # initialize openPose
                vect, heat = net.loss_last()
                output_vectmap.append(vect)
                output_heatmap.append(heat)
                outputs.append(net.get_output())
                l1s, l2s = net.loss_l1_l2()

                # get loss values at each stage of OpenPose, apply masks if
                # required
                for msk in q_mask_split:
                    msk = msk[0, :, :, :]
                for idx, (l1, l2) in enumerate(zip(l1s, l2s)):
                    loss_l1 = tf.nn.l2_loss(
                        tf.concat(l1, axis=0) - q_vect_split[gpu_id],
                        name='loss_l1_stage%d_tower%d' % (idx, gpu_id))
                    loss_l2 = tf.nn.l2_loss(
                        tf.concat(l2, axis=0) - q_heat_split[gpu_id],
                        name='loss_l2_stage%d_tower%d' % (idx, gpu_id))

                    if args.volumetricMask:
                        mask_tensor_vectmaps = q_mask_split[gpu_id][:, :, :, nr_keypoints:]
                    else:
                        mask_tensor_vectmaps = tf.tile(q_mask_split[gpu_id], [1, 1, 1, nr_vectmaps])

                    masked_diff_l1 = tf.multiply((l1 - q_vect_split[gpu_id]), mask_tensor_vectmaps)
                    loss_l1_masked = tf.nn.l2_loss(masked_diff_l1, name='loss_l1_stage%d_tower%d' % (idx, gpu_id))

                    if args.volumetricMask:
                        mask_tensor_keypoints = q_mask_split[gpu_id][:, :, :, :nr_keypoints]
                    else:
                        mask_tensor_keypoints = tf.tile(q_mask_split[gpu_id], [1, 1, 1, nr_keypoints])
                    masked_diff = tf.multiply((l2 - q_heat_split[gpu_id]), mask_tensor_keypoints,
                                              name='masked_difference_pred_gt')
                    loss_l2_masked = tf.nn.l2_loss(masked_diff, name='loss_l2_stage%d_tower%d' % (idx, gpu_id))

                    losses_masked.append(tf.reduce_mean([loss_l1_masked, loss_l2_masked]))
                    losses.append(tf.reduce_mean([loss_l1, loss_l2]))

                heatDif = tf.concat(l2, axis=0) - q_heat_split[gpu_id]

                # normalize loss by nr of joints
                sample_loss_ll_l1_normed = tf.div(
                    [tf.nn.l2_loss(masked_diff_l1[i, :, :, :]) for i in range(masked_diff_l1.shape[0])],
                    q_nr_joints_split[gpu_id], name='sample_loss_ll_l1_normed')
                sample_loss_ll_l2_normed = tf.div(
                    [tf.nn.l2_loss(masked_diff[i, :, :, :]) for i in range(masked_diff.shape[0])],
                    q_nr_joints_split[gpu_id], name='sample_loss_ll_l2_normed')
                sample_loss_ll_normed = tf.reduce_mean(
                    [sample_loss_ll_l1_normed, sample_loss_ll_l2_normed],
                    axis=0)

                last_losses_l1.append(loss_l1)
                last_losses_l2.append(loss_l2)
                last_losses_l1_masked.append(loss_l1_masked)
                last_losses_l2_masked.append(loss_l2_masked)

                ###############teacher loss##################################
                with tf.variable_scope('teacher', reuse=(gpu_id > 0)):
                    masked_last_losses = tf.reduce_sum(
                        tf.add(last_losses_l1_masked, last_losses_l2_masked))
                    probs = teacherNet.get_probabilities()
                    probs = tf.concat(probs, axis=1)

                    outs = 0
                    tea_loss_list = []
                    for i, fc in enumerate(teacherNet.get_fc()):
                        gt_param = tf.ones([args.batchsize, output_size],
                                           dtype=tf.float32) * gt_teacher[outs:outs + output_size]
                        tea_loss_list.append(
                            tf.nn.softmax_cross_entropy_with_logits(logits=fc, labels=gt_param))
                        outs = nbins_all[i]
                    teacher_loss = tf.reduce_sum(tea_loss_list )

    outputs = tf.concat(outputs, axis=0)


    # process losses
    with tf.device(tf.DeviceSpec(device_type="CPU")):
        # define loss
        total_loss = tf.reduce_sum(losses) / args.batchsize
        total_loss_ll_paf = tf.reduce_sum(last_losses_l1) / args.batchsize
        total_loss_ll_heat = tf.reduce_sum(last_losses_l2) / args.batchsize
        total_loss_ll = tf.reduce_mean([total_loss_ll_paf, total_loss_ll_heat])

        # masked
        total_loss_masked = tf.reduce_sum(losses_masked) / args.batchsize
        total_loss_ll_paf_masked = tf.reduce_sum(last_losses_l1_masked) / args.batchsize
        total_loss_ll_heat_masked = tf.reduce_sum(last_losses_l2_masked) / args.batchsize
        total_loss_ll_masked = tf.reduce_mean([total_loss_ll_paf_masked,
                                               total_loss_ll_heat_masked])


        # define parameters for optimizer
        if args.data_set == 'MPII':
            step_per_epoch = 15956 // args.batchsize
        global_step = tf.Variable(0, trainable=False)
        global_step_teacher = tf.Variable(0, trainable=False)

        starter_learning_rate = float(args.lr)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   decay_steps=args.decay_steps,
                                                   decay_rate=args.decay_rate,
                                                   staircase=True)


    ###################### setting up optimizers###############################
    logger.info('set up optimizers')
    var_list = tf.all_variables()
    var_list_teacher = []
    var_list_student = []
    for var in var_list:
        if 'teacher' in var.name or '_fc' in var.name:
            var_list_teacher.append(var)
        elif not ('Variable' in var.name):
            var_list_student.append(var)

    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1,
                                       beta2=args.beta2, epsilon=1e-8)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss_masked,
                                      var_list=var_list_student,
                                      global_step=global_step,
                                      colocate_gradients_with_ops=True)

    optimizer_teacher = tf.train.AdamOptimizer('0.00005', beta1=args.beta1,
                                               beta2=args.beta2, epsilon=1e-8)
    update_ops_teacher = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op_teacher = optimizer.minimize(teacher_loss,
                                              var_list=var_list_teacher,
                                              global_step=global_step_teacher,
                                              colocate_gradients_with_ops=True)


    #################### define summaries ####################################
    logger.info('define summary')
    tf.summary.scalar("loss", total_loss)
    tf.summary.scalar("loss_lastlayer", total_loss_ll)
    tf.summary.scalar("loss_lastlayer_paf", total_loss_ll_paf)
    tf.summary.scalar("loss_lastlayer_heat", total_loss_ll_heat)
    tf.summary.scalar("loss_masked", total_loss_masked)
    tf.summary.scalar("loss_lastlayer_masked", total_loss_ll_masked)
    tf.summary.scalar("loss_lastlayer_paf_masked", total_loss_ll_paf_masked)
    tf.summary.scalar("loss_lastlayer_heat_masked", total_loss_ll_heat_masked)
    merged_summary_op = tf.summary.merge_all()

    # normal loss
    sample_train = tf.placeholder(tf.float32, shape=(12, 640, 640, 3))
    train_img = tf.summary.image('training_sample', sample_train, 4)


    #teacher losses
    teacher_update_loss = tf.placeholder(tf.float32, shape=())
    loss_teacher_update = tf.summary.scalar('teacher_update_loss', teacher_update_loss)
    teacher_objective = tf.placeholder(tf.float32, shape=())
    teacher_objectiv_summary = tf.summary.scalar('teacher_objective', teacher_objective)
    loss_tmOne_loss_t = tf.placeholder(tf.float32, shape=())
    loss_tmOne_loss_t_summary = tf.summary.scalar('loss_tmOne_loss_t', loss_tmOne_loss_t)
    merged_teacher_train_op = tf.summary.merge([loss_teacher_update, teacher_objectiv_summary,
                                                loss_tmOne_loss_t_summary])

    datasetname = args.datapath.split('/')[-1]
    saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=2.7)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)


    ################ get a training name and directories to store info#########
    loss_name = 'synth_adversary_'
    prms = ''
    for prm in teacher_parameter_list:
        prms += prm +'_'
    training_name = loss_name + '_' + prms + '_' + 'alpha_{}_beta{}_{}{}_dataSet:{}_batch:{' \
                                                   '}_lr:{}_beta1:{}_decay_steps:{}_decay_rate:{' \
                                                   '}_gpus:{}_{}x{}_batchsize:{}_%/'.format(
        alpha,
        beta,
        args.identifier,
        args.model,
        datasetname,
        args.batchsize,
        args.lr,
        args.beta1,
        args.decay_steps,
        args.decay_rate,
        args.gpus,
        args.input_width, args.input_height,
        args.batchsize )

    #get directories to safe debug information
    np_path = os.path.join(args.tmpNpyPath, training_name)
    gt_path = os.path.join(args.tmpgtPath, training_name)
    if not (os.path.isdir(args.tmpNpyPath)):
        os.mkdir(args.tmpNpyPath)
    if not (os.path.isdir(args.tmpgtPath)):
        os.mkdir(args.tmpgtPath)
    if not (os.path.isdir(np_path)):
        os.mkdir(np_path)
    if not (os.path.isdir(gt_path)):
        os.mkdir(gt_path)


    ###################### setting up the dataflows ###########################
    #set up dataflow for real training data
    df_train_real = get_dataflow_batch(args.datapath, True,
                                       args.batchsize / args.synth_real_split[1],
                                       img_path=args.imgpath,
                                       dataSet=args.data_set,
                                       hyperparams=augmentation_hyperparams,
                                       dataflow_queuelenth=30,
                                       volumetricMask=args.volumetricMask)
    df_train_real.reset_state()

    #get name for specific parameter set
    bin_names_all = teacherNet.get_output_names()
    bin_names = []
    for bin_name in bin_names_all:
        bin_name = bin_name.split('teacher/')[-1]
        if bin_name in teacher_parameter_list:
            bin_names.append(bin_name)
    for param in teacher_parameter_list:
        if not param in bin_names_all:
            bin_names.append(param)
    params = '_'.join(str(elem) for elem in teacher_parameter_list)

    # define path for dataset construction dictionary and file containing the
    # group boundaries
    const_dict_path = './tf_pose/probs_dict_personLevel_bins_fineCol_6gmm_cropFixed' \
                      + params + '.npy'
    if args.mixed_data:
        const_dict_path = './tf_pose/probs_dict_personLevel_bins_mixedData' + params + '.npy'
    # volumetric mask
    if args.volumetricMask:
        const_dict_path = './tf_pose/probs_dict_personLevel_bins_mixedData_synthList_secondPerosn' + \
                          params + '.npy'
    if args.stylized:
        const_dict_path = './tf_pose/probs_dict_personLevel_bins_stylized_Perosn' + params + '.npy'
    if args.stylized and args.mixed_data:
        const_dict_path = './tf_pose/probs_dict_personLevel_bins_stylizedAndMixed_Perosn' + params + '.npy'

    # get group boundaries
    gm_params_all = tut.load_gm_params(args.gm_path)
    gm_params = {}
    for key in gm_params_all.keys():
        if key=='visiblility_ratio_total':
            testkey='visibility_ratio_total'
        else:
            testkey = key
        if testkey in teacher_parameter_list:
            gm_params[key] = gm_params_all[key]

    # set up synthetic data dataflow
    picked = tut.get_picked(output_size=output_size, nbins_all=nbins_all,
                            bin_names=bin_names, picked_bins=picked_bins)
    logger.info(str(picked))
    # pick a different group until a non-empty group is found
    empty = True
    df_train_synth = None
    while empty:
        if df_train_synth is None:
            if os.path.isfile(const_dict_path):
                construction_dict = np.load(const_dict_path)
                construction_dict = construction_dict.item()

                df_train_synth = get_dataflow_batch(args.synth_data_path, True,
                                                    args.batchsize /
                                                    args.synth_real_split[0],
                                                    img_path=args.imgpath,
                                                    dataSet=args.data_set,
                                                    hyperparams=augmentation_hyperparams,
                                                    picked_bins=picked,
                                                    construction_dict=construction_dict,
                                                    gm_params=gm_params,
                                                    dataflow_queuelenth=30,
                                                    person_level=True,
                                                    params=teacher_parameter_list,
                                                    use_bins=args.useBins,
                                                    synthTrain=synthTrainThreads,
                                                    mixedData=args.mixed_data,
                                                    volumetricMask=args.volumetricMask)
            else:
                logger.info('construncting training data file - this might take a while')
                df_train_synth = get_dataflow_batch(args.synth_data_path, True,
                                                    args.batchsize /
                                                    args.synth_real_split[0],
                                                    img_path=args.imgpath,
                                                    dataSet=args.data_set,
                                                    hyperparams=augmentation_hyperparams,
                                                    picked_bins=picked,
                                                    gm_params=gm_params,
                                                    dataflow_queuelenth=30,
                                                    person_level=True,
                                                    use_bins=args.useBins,
                                                    params=teacher_parameter_list,
                                                    synthTrain=synthTrainThreads,
                                                    mixedData=args.mixed_data,
                                                    volumetricMask=args.volumetricMask)

                # save if no construction dict was generated before
                construction_dict = df_train_synth.ds.ds.ds.ds.ds.ds.ds.ds.ds.ds.get_dictionary()
                np.save(const_dict_path, construction_dict)

        nrOfValidSamples = tut.check_if_empty(df_train_synth, picked, args.useBins)
        if nrOfValidSamples <= minimalNrOfSamples:
            picked = tut.get_picked(output_size=output_size, nbins_all=nbins_all, bin_names=bin_names,
                                    picked_bins=picked_bins, useBins=args.useBins)
            logger.info(str(picked))
        else:
            empty = False
            df_train_synth.ds.ds.ds.ds.ds.ds.ds.ds.ds.ds.empty = False
            logger.info('updating bins')
            df_train_synth = tut.set_up_new_dataflow(df_train_synth,
                                                     args.synth_data_path, True,
                                                     args.batchsize,
                                                     args.synth_real_split[0],
                                                     img_path=args.imgpath,
                                                     dataSet=args.data_set,
                                                     hyperparams=augmentation_hyperparams,
                                                     construction_dict=construction_dict,
                                                     picked_bins=picked,
                                                     gm_params=gm_params,
                                                     dataflow_queuelenth=30,
                                                     person_level=True,
                                                     use_bins=args.useBins,
                                                     params=teacher_parameter_list,
                                                     synthTrainThreads=synthTrainThreads,
                                                     mixed_data=args.mixed_data,
                                                     volumetricMask=args.volumetricMask,
                                                     args=args)
            df_train_synth.reset_state()


    ######################## load weights ####################################
    with tf.Session(config=config) as sess:
        logger.info('model weights initialization')
        sess.run(tf.global_variables_initializer())
        if args.checkpoint:
            savedir = training_name
            args.checkpoint = os.path.join(args.checkpoint, savedir)
            if not (os.path.isdir(os.path.join(args.checkpoint))):
                os.mkdir(os.path.join(args.checkpoint))
            try:
                logger.info('Restore from checkpoint...')
                checkpoitns=tf.train.latest_checkpoint(args.checkpoint)
                saver.restore(sess, checkpoitns)
                logger.info('Restore from checkpoint...Done')
            except:
                logger.warning('no checkpoint found')
                logger.info('Restore pretrained weights...')
                if '.ckpt' in pretrain_path or not (args.initial_student is None):
                    if not (args.initial_student is None):
                        logger.info('initialize student with pretrained network')
                        print(args.initial_student)
                        loader = tf.train.Saver(var_list_student)
                        loader.restore(sess, args.initial_student)
                    else:
                        loader = tf.train.Saver(net.restorable_variables())
                        loader.restore(sess, pretrain_path)
                elif '.npy' in pretrain_path:
                    vgg.load(pretrain_path, sess, False)
                logger.info('Restore pretrained weights...Done')
        elif pretrain_path:
            if not (args.checkpoint):
                args.checkpoint = os.path.join(args.modelpath, training_name, 'model')
            logger.info('Restore pretrained weights...')
            if '.ckpt' in pretrain_path or not (args.initial_student is None):
                if not (args.initial_student is None):
                    logger.info('initialize student with pretrained network')
                    loader = tf.train.Saver(var_list_student)
                    loader.restore(sess, args.inistial_student)
                else:
                    loader = tf.train.Saver(net.restorable_variables())
                    loader.restore(sess, pretrain_path)
            elif '.npy' in pretrain_path:
                vgg.load(pretrain_path, sess, False)
                net.load(pretrain_path, sess, False)
                teacherNet.load(pretrain_path, sess, False)
            logger.info('Restore pretrained weights...Done')

        logger.info('prepare file writer')
        logger.info('logging to: ' + args.logpath + '/' + training_name)
        file_writer = tf.summary.FileWriter(args.logpath + '/' + training_name, sess.graph)
        logger.info('prepare coordinator')

        # load history for teacher loss
        history_path = os.path.join(args.tmpNpyPath, training_name, 'hist.npy')
        if os.path.exists(history_path):
            hist = np.load(history_path)
            hist = hist.item()
            teacher_loss_history = hist['teacher_loss_history']
            histcount = hist['historycount']
        else:
            histcount = 0


        #######################initalize training###############################
        logger.info('Training Started.')
        train = True
        # first iteration is slightly different
        first = True
        time_started = time.time()
        last_gs_num = last_gs_num2 = last_gs_num3 = 0
        if first:
            initial_gs_num = sess.run(global_step)

        hard_example_list = np.zeros(
            ((steps_till_teacher_training ) * args.batchsize,
             args.input_height, args.input_height, 3))
        hard_example_losses = np.zeros(
            ((steps_till_teacher_training) * args.batchsize))
        loss_mask_only_real = np.hstack(
            (np.zeros((int(np.round(args.batchsize / 2)),), dtype=bool),
             np.ones((int(np.round(args.batchsize / 2)),), dtype=bool)))
        loss_mask_only_real = np.tile(loss_mask_only_real,
                                      steps_till_teacher_training)


        ##################### training loop ################################
        teacher_train_old = None
        while train:
            reset = False
            if not first:
                logger.info('updating bins')
                df_train_synth = tut.set_up_new_dataflow(
                    df_train_synth,
                    args.synth_data_path,
                    True, args.batchsize,
                    args.synth_real_split[0],
                    img_path=args.imgpath,
                    dataSet=args.data_set,
                    hyperparams=augmentation_hyperparams,
                    construction_dict=construction_dict,
                    picked_bins=picked,
                    gm_params=gm_params,
                    dataflow_queuelenth=30,
                    person_level=True,
                    use_bins=args.useBins,
                    params=teacher_parameter_list,
                    synthTrainThreads=synthTrainThreads,
                    mixed_data=args.mixed_data,
                    volumetricMask=args.volumetricMask,
                    args=args)
                df_train_synth.reset_state()
                hard_example_list = np.zeros((
                    (steps_till_teacher_training) * args.batchsize,
                    args.input_height, args.input_height, 3))
                hard_example_losses = np.zeros(
                    ((steps_till_teacher_training) * args.batchsize))

            batch_count = 0
            for out_synth, out_real in zip(df_train_synth.get_data(), df_train_real.get_data()):
                images_train, heatmaps_train, vectmaps_train, mask_train, \
                q_nr_joints_train = np.concatenate(
                    (out_synth[0], out_real[0]), axis=0), np.concatenate(
                    (out_synth[1], out_real[1]), axis=0), np.concatenate(
                    (out_synth[2], out_real[2]), axis=0), np.concatenate(
                    (out_synth[3], out_real[3]), axis=0), np.concatenate(
                    (out_synth[4], out_real[4]), axis=0)
                q_nr_joints_train = np.array(q_nr_joints_train)

                _, gs_num, sample_loss_ll_normed_arr = sess.run(
                    [train_op, global_step, sample_loss_ll_normed],
                    feed_dict={q_inp: images_train, q_vect: vectmaps_train,
                               q_heat: heatmaps_train, q_mask: mask_train,
                               q_nr_joints: q_nr_joints_train})


                if first:
                    last_gs_num = gs_num - 1
                    last_gs_num3 = gs_num - 1
                first = False

                # save the losses of examples for the teacher network
                for i in range(images_train.shape[0]):
                    hard_example_list[batch_count * args.batchsize + i, :,
                    :, :] = images_train[i, :, :, :]
                    hard_example_losses[batch_count * args.batchsize + i] = \
                        sample_loss_ll_normed_arr[i]
                batch_count += 1

                # stopp training when args.max_epoch is reached
                if gs_num > step_per_epoch * args.max_epoch:
                    train = False
                    break

                # log progress
                if gs_num - last_gs_num >= 100:

                    train_loss, train_loss_ll, train_loss_ll_paf,  \
                    train_loss_ll_heat, train_loss_masked,  \
                    train_loss_ll_masked, train_loss_ll_paf_masked,  \
                    train_loss_ll_heat_masked, lr_val, summary =  \
                        sess.run([  total_loss, total_loss_ll,
                                    total_loss_ll_paf, total_loss_ll_heat,
                                     total_loss_masked, total_loss_ll_masked,
                                    total_loss_ll_paf_masked,
                                    total_loss_ll_heat_masked, learning_rate,
                                    merged_summary_op],
                                    feed_dict={ q_inp: images_train,
                                                q_vect: vectmaps_train,
                                                q_heat: heatmaps_train,
                                                q_mask: mask_train})


                    batch_per_sec = (gs_num - initial_gs_num) / \
                                    (time.time() - time_started)
                    logger.info(
                        'epoch=%.2f step=%d, %0.4f examples/sec '
                        'lr=%f,  loss=%g, loss_ll=%g, '
                        'loss_ll_paf=%g,  loss_ll_heat=%g,  '
                        'loss_masked=%g, loss_ll_masked=%g,  '
                        'loss_ll_paf_masked=%g,  '
                        'loss_ll_heat_masked=%g' % (
                            gs_num / step_per_epoch,  gs_num, batch_per_sec *
                            args.batchsize, lr_val,  train_loss, train_loss_ll,
                            train_loss_ll_paf, train_loss_ll_heat,
                            train_loss_masked, train_loss_ll_masked,
                            train_loss_ll_paf_masked,
                            train_loss_ll_heat_masked))
                    last_gs_num = gs_num
                    file_writer.add_summary(summary, gs_num)


                #################### train teacher############################
                if gs_num - last_gs_num3 >= steps_till_teacher_training:
                    last_gs_num3 = gs_num
                    # highest at first position
                    idxs = np.argsort(hard_example_losses[loss_mask_only_real])[::-1]

                    # compute mean over last 10 teacher loss values to get a less noisy teaching signal
                    smoothed_teachingSignal = np.mean(teacher_loss_history[0, :][teacher_loss_history[1, :] > 0])
                    loss_current = np.mean(hard_example_losses[np.logical_not(loss_mask_only_real)])

                    teacher_train = hard_example_list[loss_mask_only_real][idxs[:32], :, :, :]

                    # only train when we have values to compare to
                    if not (teacher_train_old is None):
                        logger.info('teaching the teacher')
                        scale_loss = 1
                        # as it is a constant
                        teacher_gt = compute_gt_teacher(
                            probabilities, picked, alpha, beta,
                            teacher_loss_type,
                            loss_last=smoothed_teachingSignal,
                            loss_current=loss_current,
                            nbins_nrs=nbins_nrs, nbins=nbins)
                        print('ground truth for teacher: ' + str(teacher_gt))

                        # train the teacher
                        _, _, total_teacher_loss_arr = sess.run([
                            train_op_teacher, global_step, teacher_loss],
                            feed_dict={   teach_inp: teacher_train_old,
                                          probs_old: np.squeeze(probabilities),
                                          picked_probs_old: probabilities_picked,
                                          loss_tmOne: np.squeeze(smoothed_teachingSignal),
                                          loss_t: np.squeeze(loss_current),
                                          gt_teacher: teacher_gt,
                                          loss_scale: scale_loss})

                    #fill the teacher_loss_history array at the next position
                    teacher_loss_history[0, histcount] = loss_current
                    teacher_loss_history[1, histcount] = gs_num
                    #set histcount to 0 to start overwriting values
                    histcount += 1
                    if histcount >= np.shape(teacher_loss_history)[1]:
                        histcount = 0

                    # until non empty pick found
                    while True:
                        pr = []
                        probabilities = sess.run(
                            [probs],
                            feed_dict={teach_inp: teacher_train})
                        probabilities = probabilities[0][0,:]

                        picked = tut.get_picked(output_size=output_size, nbins_all=nbins_all,
                                                bin_names=bin_names, picked_bins=picked_bins,
                                                probabilities=probabilities, nbins=nbins,
                                                useBins=args.useBins)
                        probabilities_picked = np.repeat(probabilities[picked],
                                                         output_size)
                        nrOfValidSamples = tut.check_if_empty(df_train_synth,
                                                          picked, args.useBins)
                        logger.info('examples in bin: '+str(nrOfValidSamples))
                        # in case bin is empty
                        if nrOfValidSamples <= minimalNrOfSamples:
                            # dont penalize in first iteration. just pick
                            # another bin if first iteration
                            if not (teacher_train is None):
                                logger.info(
                                    'the proposed bins don\'t contain any '
                                    'data points. retry after penalizing the teacher')

                                teacher_gt = compute_gt_teacher(
                                    probabilities,
                                    picked, alpha, beta,
                                    teacher_loss_type,
                                    loss_last=artificial_penalty[0],
                                    loss_current=artificial_penalty[1],
                                    nbins_nrs=nbins_nrs, nbins=nbins)
                                # penalize teacher to reduce probability of
                                # empty group
                                _, _, total_teacher_loss_arr = sess.run(
                                    [train_op_teacher, global_step,
                                        teacher_loss],
                                    feed_dict={teach_inp: teacher_train,
                                               probs_old: np.squeeze(
                                                   probabilities),
                                               picked_probs_old:
                                                   probabilities_picked,
                                               loss_tmOne: np.squeeze(
                                                   artificial_penalty[0]),
                                               loss_t: np.squeeze(
                                                   artificial_penalty[1]),
                                               gt_teacher: teacher_gt,
                                               loss_scale: 1})
                        else:
                            loss_old = np.copy(loss_current)
                            teacher_train_old = np.copy(teacher_train)
                            break

                    # save output probabilities and gt for debugging
                    if not (teacher_gt is None):
                        np.save(os.path.join(gt_path, str(gs_num)), teacher_gt)
                    if not (teacher_train_old is None):
                        batch_count = 0
                        loss_log = np.mean(loss_current)
                        # get summary
                        np.save(os.path.join(np_path, str(gs_num)),
                                probabilities)
                        summary = sess.run(
                            merged_teacher_train_op,
                            feed_dict={
                                teacher_update_loss: loss_log,
                                teacher_objective: total_teacher_loss_arr,
                                loss_tmOne_loss_t: np.squeeze(
                                    smoothed_teachingSignal - loss_current)})
                        file_writer.add_summary(summary, gs_num)
                        reset = True

                ###########################################
                # save checkpoint
                if gs_num - last_gs_num2 >= 1000:
                    # save weights
                    saver.save(sess, args.checkpoint, global_step=global_step)
                    hist_dict = {'historycount': histcount,
                                 'teacher_loss_history': teacher_loss_history}
                    np.save(history_path, hist_dict)

                    average_loss = average_loss_ll = average_loss_ll_paf = \
                        average_loss_ll_heat = average_loss_masked = \
                        average_loss_ll_masked = average_loss_ll_paf_masked \
                        = average_loss_ll_heat_masked = 0
                    total_cnt = 0

                    last_gs_num2 = gs_num


                # break out of sampling loop to set up new dataflow after
                # teacher was trained
                if reset:
                    break
    #######################
    saver.save(sess, args.checkpoint, global_step=global_step)
    logger.info('optimization finished. %f' % (time.time() - time_started))
    exit(0)
