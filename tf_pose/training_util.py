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

from pose_dataset_personLevel_minDist import get_dataflow_batch
import numpy as np

def load_gm_params(path):
    if not ('.npy' in path):
        from scipy.io import loadmat
        gms = loadmat(path, squeeze_me=True, struct_as_record=False)
        gms = gms['gm']
        param_dict = {'camera_angle': gms.camera_angle,
                      'median_dist': gms.median_dist,
                      'std_dist': gms.var_dist,
                      'median_scale': gms.median_scale,
                      'std_scale': gms.var_scale,
                      'nr_people': gms.nr_people,
                      'visibility_ratio_total': gms.visibility_ratio_total,
                      'cropped_ratio_total': gms.cropped_ratio_total}
        return param_dict
    else:
        gm_params = np.load(path)
        gm_params = gm_params.item()
        return gm_params



def check_if_empty(df_train_synth, picked, useBins):
    if useBins:
        inBin = np.squeeze(df_train_synth.ds.ds.ds.ds.ds.ds.ds.ds.ds.ds.probs[:, picked])  # is a boolean index if in bin or not if useBins is true
        # check if something in bin (do before in train)
        validSampleCount = np.sum(inBin)
        print(validSampleCount)
    else:
        gm_data_point_probs = df_train_synth.ds.ds.ds.ds.ds.ds.ds.ds.ds.ds.probs
        mask = gm_data_point_probs[:, picked] > 0
        conditional_probs = np.log(gm_data_point_probs[:, picked], where=mask)
        conditional_probs[np.logical_not(mask)] = -np.inf
        validSampleCount = np.sum(np.exp(np.sum(conditional_probs, 1)))

    return validSampleCount



def set_up_new_dataflow(df_train_synth, synth_data_path, train, batchsize, synth_real_split, img_path, dataSet, hyperparams, construction_dict,
                        picked_bins, gm_params, dataflow_queuelenth, person_level, use_bins, params, synthTrainThreads, mixed_data, volumetricMask=False, args=None):
    construction_dict = df_train_synth.ds.ds.ds.ds.ds.ds.ds.ds.ds.ds.get_dictionary()
    for proc in df_train_synth.ds.ds.procs:
        proc.terminate()
    for proc in df_train_synth.procs:
        proc.terminate()
    del df_train_synth.ds.ds.ds.ds.ds.ds.ds.ds.ds.ds
    df_train_synth.ds.ds.ds.ds.ds.ds.ds.ds.ds.ds = None
    del df_train_synth.ds.ds.ds.ds.ds.ds.ds.ds.ds
    df_train_synth.ds.ds.ds.ds.ds.ds.ds.ds.ds = None
    del df_train_synth.ds.ds.ds.ds.ds.ds.ds.ds
    df_train_synth.ds.ds.ds.ds.ds.ds.ds.ds = None
    del df_train_synth.ds.ds.ds.ds.ds.ds.ds
    df_train_synth.ds.ds.ds.ds.ds.ds.ds = None
    del df_train_synth.ds.ds.ds.ds.ds.ds
    df_train_synth.ds.ds.ds.ds.ds.ds = None
    del df_train_synth.ds.ds.ds.ds.ds
    df_train_synth.ds.ds.ds.ds.ds = None
    del df_train_synth.ds.ds.ds.ds
    df_train_synth.ds.ds.ds.ds = None
    del df_train_synth.ds.ds.ds
    df_train_synth.ds.ds.ds = None
    del df_train_synth.ds.ds
    df_train_synth.ds.ds = None
    del df_train_synth.ds

    df_train_synth.ds = None
    del df_train_synth

    df_train_synth = get_dataflow_batch(args.synth_data_path, train, batchsize / synth_real_split,
                                        img_path=img_path, dataSet=dataSet, hyperparams=hyperparams, construction_dict=construction_dict,
                                        picked_bins=picked_bins, gm_params=gm_params, dataflow_queuelenth=dataflow_queuelenth, person_level=person_level, use_bins=use_bins,
                                        params=params, synthTrain=synthTrainThreads, mixedData=mixed_data, volumetricMask=volumetricMask)
    return df_train_synth



def get_picked(output_size, nbins_all, bin_names, picked_bins,
               probabilities=None, nbins=None, useBins=True):
    picked = np.zeros((output_size,), dtype=bool)
    nbins_last = 0
    nbins_sum = 0
    i = 0

    if probabilities is None:
        print('first one is random')
        picked[np.random.choice(np.arange(0, output_size))] = True
    else:
        if np.random.randint(0,10) == 0:
            print('lets roll some dice!')
            picked[np.random.choice(np.arange(0, output_size))] = True #to
            # ensure some random exploration with probability of 10%
        else:
            print('sampling with teacher probs')
            picked[np.random.choice(np.arange(0, output_size), p=probabilities)] = True
        picked = np.array(picked, dtype=bool)

    i = None

    return picked
