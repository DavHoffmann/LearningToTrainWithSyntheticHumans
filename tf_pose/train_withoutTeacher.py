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
mpl.use('Agg')  # training mode, no screen should be open. (It will block training loop)

import argparse
import logging
import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm


from pose_augment_distSampling import set_network_input_wh, set_network_scale
from pose_dataset_personLevel_minDist import get_dataflow_batch, DataFlowToQueue, \
    CocoPose, CocoMetadata, MPIMetadata, MPIIPose
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





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--model', default='cmu_mpi_vgg_interSup', help='model name')  # changes to be applied for MPI
    parser.add_argument('--datapath', type=str, default='./MPI.json', help='path  json file')
    parser.add_argument('--synth_data_path', type=str, default='./purelySynthetic', help='path to folder containing json files')
    parser.add_argument('--imgpath', type=str, default='./imagesMPI')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--epochs-till-restart', type=int, default=5)
    # paths
    parser.add_argument('--modelpath', type=str, default='./models/cmu_mpi')
    parser.add_argument('--logpath', type=str, default='./logfiles')
    parser.add_argument('--checkpoint', type=str, default='./models/')
    parser.add_argument('--gm_path', type=str, default='./tf_pose/groupBoundaries.npy')

    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)

    # optimizer
    parser.add_argument('--lr', type=str, default='0.00005')
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--beta1', type=str, default=0.8)
    parser.add_argument('--beta2', type=str, default=0.999)
    parser.add_argument('--param-idx', type=int, default=None)
    parser.add_argument('--decay-steps', type=int, default=20000)
    parser.add_argument('--decay_rate', type=int, default=0.33)

    # data augmentation
    parser.add_argument('--scale_min', type=float, default=0.4)
    parser.add_argument('--scale_max', type=float, default=1.6)
    parser.add_argument('--target_dist', type=float, default=1.25)
    parser.add_argument('--center_perterb_max', type=int, default=50)
    parser.add_argument('--max_rotate_degree', type=float, default=45)

    #other
    parser.add_argument('--initial_student', type=str, default='./models/baselineIntermediate/-70002')
    parser.add_argument('--identifier', type=str, default='')

    #dataset
    parser.add_argument('--data-set', type=str, default='MPII', help='MPII')
    parser.add_argument('--synth_real_split', type=list, default=[2, 2])
    parser.add_argument('--paramSearch', type=int, default=0)
    parser.add_argument('--useBins', type=int, default=1)
    parser.add_argument('--prepicked', type=int, default=1)
    parser.add_argument('--mixed_data', type=bool, default=False)
    parser.add_argument('--volumetricMask', type=bool, default=False)
    parser.add_argument('--stylized', type=bool, default=False)
    parser.add_argument('--onlyReal', type=bool, default=False)

    args = parser.parse_args()

    #nothing else supported
    args.prepicked = True
    args.useBins = True


    trainingname_addition = ''
    # define minimal number of samples per bin
    minimalNrOfSamples = args.batchsize
    #flag to optimize thread numbers in dataflows
    synthTrainThreads = True

    teacher_parameter_list = ['minDist']

    steps_till_teacher_training = 20

    nbins_all = np.array([10, 10])
    nbins = []
    params_used = []
    param_all = ['camera_angle', 'minDist']
    for i, parameter in enumerate(param_all):
        if parameter in teacher_parameter_list:
            nbins.append(nbins_all[i])
            params_used.append(True)
        else:
            params_used.append(False)
    output_size = nbins[0]

    if args.prepicked:
        if args.param_idx ==0:
            prepicked = np.ones((output_size,))
            trainingname_addition ='allRandom'

        prepicked = np.array(prepicked, dtype=bool)
    nbins = np.array(nbins)
    bins = np.zeros((len(teacher_parameter_list), sum(nbins)))
    if args.prepicked:
        picked = prepicked
    nbins_nrs = []

    probabilities = []
    for n in range (len(nbins)):
        nbins_nrs = np.concatenate([nbins_nrs, np.repeat([nbins[n]], nbins[n])], axis=0)
        probabilities = np.concatenate([probabilities, np.repeat([1 / nbins[n]], nbins[n])], axis=0)


    nbins_nrs = np.array(nbins_nrs, dtype=int)
    probabilities = np.array(probabilities)
    random_sampling_loss_history = np.zeros((10,))
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


    # define the network
    if args.model in ['cmu', 'vgg', 'mobilenet_thin', 'mobilenet_try', 'mobilenet_try2',
                      'mobilenet_try3', 'hybridnet_try', 'cmu_mpi', 'cmu_mpi_vgg',
                      'cmu_mpi_vgg_interSup']:
        scale = 8

    logger.info('define model+')
    set_network_scale(scale)
    output_w, output_h = args.input_width // scale, args.input_height // scale
    with tf.device(tf.DeviceSpec(device_type="CPU")):
        input_node = tf.placeholder(tf.float32,
                                    shape=(args.batchsize, args.input_height, args.input_width, 3),
                                    name='image')
        if (args.data_set == 'MPII'):
            nr_keypoints = 16
            nr_vectmaps = 28
            if args.volumetricMask:
                mask_node = tf.placeholder(
                    tf.float32,
                    shape=(args.batchsize, output_h,
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

        if args.data_set == 'MPII':
            q_inp = tf.placeholder(dtype=tf.float32,
                                   shape=(None, None, None, 3))
            q_heat = tf.placeholder(dtype=tf.float32,
                                    shape=(args.batchsize, 46, 46, nr_keypoints))
            q_vect = tf.placeholder(dtype=tf.float32,
                                    shape=(args.batchsize, 46, 46, nr_vectmaps))

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
            picked_forPolicyGradient = tf.placeholder(dtype=tf.float32,
                                                      shape=np.shape(
                                                          probabilities))
            picked_probs_old = tf.placeholder(dtype=tf.float32,
                                              shape=np.shape(probabilities))

            teach_inp = tf.placeholder(dtype=tf.float32,
                                       shape=(args.batchsize, 46, 46, nr_keypoints))
            loss_scale = tf.placeholder(dtype=tf.float32,
                                        shape=(), name='loss_scale')

    q_inp_split, q_heat_split, q_vect_split, q_mask_split, q_nr_joints_split, \
    teach_inp_split = tf.split(q_inp, args.gpus), tf.split(q_heat, args.gpus), \
                      tf.split(q_vect, args.gpus), tf.split(q_mask, args.gpus), \
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

    for gpu_id in range(args.gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                # initialize openPose
                net, pretrain_path, last_layer = get_network(args.model,
                                                             q_inp_split[gpu_id],
                                                             nr_keypoints=nr_keypoints,
                                                             trainVgg=False)
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
                        mask_tensor_keypoints = q_mask_split[gpu_id][:,:,:, :nr_keypoints]
                    else:
                        mask_tensor_keypoints = tf.tile(q_mask_split[gpu_id], [1, 1, 1, nr_keypoints])
                    masked_diff = tf.multiply((l2 - q_heat_split[gpu_id]), mask_tensor_keypoints,
                                              name='masked_difference_pred_gt')
                    loss_l2_masked = tf.nn.l2_loss(masked_diff,
                                                   name='loss_l2_stage%d_tower%d' % (idx, gpu_id))

                    losses_masked.append(tf.reduce_mean([loss_l1_masked, loss_l2_masked]))
                    losses.append(tf.reduce_mean([loss_l1, loss_l2]))

                heatDif = tf.concat(l2, axis=0) - q_heat_split[gpu_id]

                # normalize loss by nr of joints
                sample_loss_ll_l1_normed = tf.div(
                    [tf.nn.l2_loss(masked_diff_l1[i,:,:,:]) for i in range(masked_diff_l1.shape[0])],
                    q_nr_joints_split[gpu_id], name='sample_loss_ll_l1_normed')
                sample_loss_ll_l2_normed = tf.div(
                    [tf.nn.l2_loss(masked_diff[i, :, :, :]) for i in range(masked_diff.shape[0])],
                    q_nr_joints_split[gpu_id], name='sample_loss_ll_l2_normed')
                sample_loss_ll_normed = tf.reduce_mean([sample_loss_ll_l1_normed, sample_loss_ll_l2_normed],
                                                       axis=0)

                last_losses_l1.append(loss_l1)
                last_losses_l2.append(loss_l2)
                last_losses_l1_masked.append(loss_l1_masked)
                last_losses_l2_masked.append(loss_l2_masked)

    outputs = tf.concat(outputs, axis=0)


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
        total_loss_ll_masked = tf.reduce_mean([total_loss_ll_paf_masked, total_loss_ll_heat_masked])

        # define optimizer
        if args.data_set == 'MPII':
            step_per_epoch = 15956 // args.batchsize
        global_step = tf.Variable(0, trainable=False)


        starter_learning_rate = float(args.lr)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   decay_steps=args.decay_steps,
                                                   decay_rate=args.decay_rate, staircase=True)

    logger.info('setting up optimizer-')
    var_list = tf.all_variables()
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

    datasetname = args.datapath.split('/')[-1]
    synthDataName = args.synth_data_path.split('/')[-1]
    saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=3)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    loss_name = 'no_teacher'
    prms = ''
    for prm in teacher_parameter_list:
        prms += prm +'_'

    training_name = '{}_{}_teacherArgs:{}_dataSet:{}_preselected:{}_synthData:{}_%/'.format(
        args.identifier,
        args.model,
        teacher_parameter_list[0],
        datasetname,
        trainingname_addition,
        synthDataName)


    df_train_real = get_dataflow_batch(args.datapath, True,
                                       args.batchsize / args.synth_real_split[1],
                                       img_path=args.imgpath,
                                       dataSet=args.data_set,
                                       hyperparams=augmentation_hyperparams,
                                       dataflow_queuelenth=30,
                                       volumetricMask=args.volumetricMask)
    df_train_real.reset_state()


    bin_names = []

    if args.prepicked:
        picked = prepicked

    logger.info(str(picked))

    params = '_'.join(str(elem) for elem in teacher_parameter_list)

    const_dict_path = './tf_pose/probs_dict_personLevel_bins_fineCol_6gmm_croppedFixed'+params+training_name[:-1]+'.npy'
    if args.mixed_data:
        const_dict_path = './tf_pose/probs_dict_personLevel_bins_mixedData'+params+training_name[:-1]+'.npy'
    if args.volumetricMask:
        const_dict_path = './tf_pose/probs_dict_personLevel_bins_mixedData_synthList'+params+training_name[:-1]+'.npy'
    if args.stylized:
        const_dict_path = './tf_pose/probs_dict_personLevel_bins_stylized'+params+training_name[:-1]+'.npy'
    if args.mixed_data and args.stylized:
        const_dict_path = './tf_pose/probs_dict_personLevel_bins_mixedAndStylized'+params+training_name[:-1]+'.npy'

    gm_params_all = tut.load_gm_params(args.gm_path)
    gm_params = {}
    for key in gm_params_all.keys():
        if key=='visiblility_ratio_total':
            testkey='visibility_ratio_total'
        else:
            testkey =key
        if testkey in teacher_parameter_list:
            gm_params[key] = gm_params_all[key]

    empty = True
    df_train_synth = None
    while empty:
        if df_train_synth is None:
            if os.path.isfile(const_dict_path):
                construction_dict = np.load(const_dict_path)
                construction_dict = construction_dict.item()

                df_train_synth = get_dataflow_batch(args.synth_data_path, True,
                                                    args.batchsize / args.synth_real_split[0],
                                                    img_path=args.imgpath, dataSet=args.data_set,
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

                construction_dict = df_train_synth.ds.ds.ds.ds.ds.ds.ds.ds.ds.ds.get_dictionary()
                np.save(const_dict_path, construction_dict)

        nrOfValidSamples =tut.check_if_empty(df_train_synth, picked, args.useBins)
        if nrOfValidSamples <= minimalNrOfSamples:
            if args.prepicked:
                logger.warning('all prepicked bins are empty!')
                exit(-1)

        else:
            empty = False
            df_train_synth.ds.ds.ds.ds.ds.ds.ds.ds.ds.ds.emty = False
            if args.onlyReal:
                df_train_synth = get_dataflow_batch(args.datapath, True,
                                                    args.batchsize / args.synth_real_split[1],
                                                    img_path=args.imgpath, dataSet=args.data_set,
                                                    hyperparams=augmentation_hyperparams,
                                                    dataflow_queuelenth=30,
                                                    volumetricMask=args.volumetricMask)
                df_train_synth.reset_state()
            else:
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
                                                         dataflow_queuelenth=30, person_level=True,
                                                         use_bins=args.useBins, params=teacher_parameter_list,
                                                         synthTrainThreads=synthTrainThreads,
                                                         mixed_data=args.mixed_data,
                                                         volumetricMask=args.volumetricMask,
                                                         args=args)
            df_train_synth.reset_state()


 ############################### load checkpoint###############################
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
                saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
                logger.info('Restore from checkpoint...Done')
            except:
                logger.warning('no checkpoint found')
                logger.info('Restore pretrained weights...')
                if '.ckpt' in pretrain_path or not (args.initial_student is None):
                    if not (args.initial_student is None):
                        logger.info('initialize student with pretrained network')
                        loader = tf.train.Saver(var_list_student)
                        loader.restore(sess, args.initial_student)
                    else:
                        loader = tf.train.Saver(net.restorable_variables())
                        loader.restore(sess, pretrain_path)
                elif '.npy' in pretrain_path:
                    logger.info('Restore pretrained weights...Done')
                    vgg.load(pretrain_path, sess, False)
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
            logger.info('Restore pretrained weights...Done')

        logger.info('prepare file writer')
        logger.info('logging to: ' + args.logpath + '/' + training_name)
        file_writer = tf.summary.FileWriter(args.logpath + '/' + training_name, sess.graph)

        logger.info('prepare coordinator')


        #######################################################################
        logger.info('Training Started.')
        train = True
        # first iteration is slightly different
        first = True
        time_started = time.time()
        last_gs_num = last_gs_num2 = last_gs_num3 = 0
        if first:
            initial_gs_num = sess.run(global_step)


        while train:
            reset = False
            if not first:
                if args.onlyReal:
                    df_train_synth = get_dataflow_batch(args.datapath, True,
                                                        args.batchsize / args.synth_real_split[1],
                                                        img_path=args.imgpath,
                                                        dataSet=args.data_set,
                                                        hyperparams=augmentation_hyperparams,
                                                        dataflow_queuelenth=30,
                                                        volumetricMask=args.volumetricMask)
                    df_train_synth.reset_state()
                else:
                    logger.info('updating bins')
                    df_train_synth = tut.set_up_new_dataflow(df_train_synth, args.synth_data_path,
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


            batch_count = 0
            for out_synth, out_real in zip(df_train_synth.get_data(), df_train_real.get_data()):
                images_train, heatmaps_train, vectmaps_train, mask_train, q_nr_joints_train = \
                    np.concatenate((out_synth[0], out_real[0]), axis=0), \
                    np.concatenate((out_synth[1], out_real[1]), axis=0), \
                    np.concatenate((out_synth[2], out_real[2]), axis=0), \
                    np.concatenate((out_synth[3], out_real[3]), axis=0), \
                    np.concatenate((out_synth[4], out_real[4]), axis=0)
                q_nr_joints_train = np.array(q_nr_joints_train)

                _, gs_num, sample_loss_ll_normed_arr = sess.run(
                    [train_op, global_step, sample_loss_ll_normed],
                    feed_dict={q_inp: images_train, q_vect: vectmaps_train,
                               q_heat: heatmaps_train, q_mask: mask_train,
                               q_nr_joints: q_nr_joints_train})


                first = False

                if gs_num - last_gs_num >= 100:
                    train_loss, train_loss_ll, train_loss_ll_paf, train_loss_ll_heat, \
                    train_loss_masked, train_loss_ll_masked, train_loss_ll_paf_masked, \
                    train_loss_ll_heat_masked, lr_val, summary = sess.run(
                        [total_loss, total_loss_ll, total_loss_ll_paf, total_loss_ll_heat,
                         total_loss_masked, total_loss_ll_masked, total_loss_ll_paf_masked,
                         total_loss_ll_heat_masked, learning_rate, merged_summary_op],
                        feed_dict={q_inp: images_train, q_vect: vectmaps_train,
                                   q_heat: heatmaps_train, q_mask: mask_train})

                    batch_per_sec = (gs_num - initial_gs_num) / (time.time() - time_started)
                    logger.info(
                        'epoch=%.2f step=%d, %0.4f examples/sec lr=%f, loss=%g, loss_ll=%g, '
                        'loss_ll_paf=%g, loss_ll_heat=%g, loss_masked=%g, loss_ll_masked=%g, '
                        'loss_ll_paf_masked=%g, loss_ll_heat_masked=%g' % (
                            gs_num / step_per_epoch, gs_num, batch_per_sec * args.batchsize,
                            lr_val, train_loss, train_loss_ll, train_loss_ll_paf,
                            train_loss_ll_heat, train_loss_masked, train_loss_ll_masked,
                            train_loss_ll_paf_masked, train_loss_ll_heat_masked))

                    last_gs_num = gs_num
                    file_writer.add_summary(summary, gs_num)

                if gs_num - last_gs_num2 >= 1000:

                    saver.save(sess, args.checkpoint, global_step=global_step)

                    average_loss = average_loss_ll = average_loss_ll_paf = average_loss_ll_heat \
                        = average_loss_masked = average_loss_ll_masked = \
                        average_loss_ll_paf_masked = average_loss_ll_heat_masked = 0
                    total_cnt = 0

                    last_gs_num2 = gs_num

                if reset:
                    break
    #######################
    saver.save(sess, args.checkpoint, global_step=global_step)
    logger.info('optimization finished. %f' % (time.time() - time_started))
    exit(0)
