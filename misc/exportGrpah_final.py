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

import os, argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

dir = os.path.dirname(os.path.realpath(__file__))


def get_n_models(model_folder,output_nodes='y_hat',
                 output_filename='frozen-graph.pb',
                 rename_outputs=None,n=1):
    checkpoint_path = os.path.join(model_folder, 'checkpoint')

    list = os.listdir(model_folder)
    modelNrs = []
    for item in list:
        if 'meta' in item:
            it = item.split('.')[0]
            it = it.split('-')[1]
            modelNrs.append(int(it))
    max_it = np.max(modelNrs)

    modelSamples = max_it - np.arange(0,n) * 5000

    closestModels = []
    for sample in modelSamples:
        minidx=np.argmin(np.abs(modelNrs - sample))
        closestModels.append(modelNrs[minidx])

    os.rename(os.path.join(model_folder, 'checkpoint'), os.path.join(model_folder, 'checkpoint_temp'))

    for i in range(n):
        with open(os.path.join(model_folder,'checkpoint'),'w+')as outfile:
            outfile.write('model_checkpoint_path: "'+model_folder+'/-'+str(closestModels[i])+'"'+  '\n')
            outfile.write('all_model_checkpoint_paths: "' + model_folder + '/-' + str(closestModels[i]) + '"' + '\n')
        out_name = output_filename + str(closestModels[i]) +'.pb'
        freeze_graph(model_folder, output_nodes, out_name, rename_outputs)
        os.remove(os.path.join(model_folder,'checkpoint'))
    os.rename(os.path.join(model_folder, 'checkpoint_temp'), os.path.join(model_folder, 'checkpoint'))


def freeze_graph(model_folder, output_nodes='y_hat',
                 output_filename='frozen-graph.pb',
                 rename_outputs=None):


    # Load checkpoint
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    output_graph = output_filename

    # Devices should be cleared to allow Tensorflow to control placement of
    # graph when loading on different machines
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=True)

    graph = tf.get_default_graph()

    onames = output_nodes.split(',')

    # https://stackoverflow.com/a/34399966/4190475
    if rename_outputs is not None:
        nnames = rename_outputs.split(',')
        with graph.as_default():
            for o, n in zip(onames, nnames):
                _out = tf.identity(graph.get_tensor_by_name(o + ':0'), name=n)
            onames = nnames

    input_graph_def = graph.as_graph_def()

    # fix batch norm nodes
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, input_checkpoint)

        # In production, graph weights no longer need to be updated
        # graph_util provides utility to change all variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, input_graph_def,
            onames  # unrelated nodes will be discarded
        )

        # Serialize and write to file
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


def freeze_list(checkpoint_path, output_nodes, output_graph, rename_outputs, nrOfModels, nr):

    list_of_model_folders = [
        #todo: add names of models to export here
    ]


    if nr is None:
        for model in list_of_model_folders:
            print(model)
            model_name = checkpoint_path.split('/')[0]
            output_graph_complete = os.path.join(output_graph,model)
            model_name_complete = os.path.join(checkpoint_path,model)
            get_n_models(model_name_complete, output_nodes, output_graph_complete, rename_outputs, nrOfModels)
    else:
        model = list_of_model_folders[nr]
        print(model)
        model_name = checkpoint_path.split('/')[0]
        output_graph_complete = os.path.join(output_graph, model)
        model_name_complete = os.path.join(checkpoint_path, model)
        get_n_models(model_name_complete, output_nodes, output_graph_complete, rename_outputs, nrOfModels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prune and freeze weights from checkpoints into production models')
    parser.add_argument("--checkpoint_path",
                        default='ckpt',
                        type=str, help="Path to checkpoint files")
    parser.add_argument("--output_nodes",
                        default='y_hat',
                        type=str, help="Names of output node, comma seperated")
    parser.add_argument("--output_graph",
                        default='frozen-graph.pb',
                        type=str, help="Output graph filename")
    parser.add_argument("--rename_outputs",
                        default=None,
                        type=str, help="Rename output nodes for better \
                        readability in production graph, to be specified in \
                        the same order as output_nodes")
    parser.add_argument("--nrOfModels",
                        default=0,
                        type=int,
                        help='if not 1. N models with approx distance of 5000 steps will be created')
    parser.add_argument("--parms", type=int, default=None)
    parser.add_argument('--freeze_list', default=1, type=int)
    args = parser.parse_args()

    if args.freeze_list ==1:
        freeze_list(args.checkpoint_path, args.output_nodes, args.output_graph, args.rename_outputs, args.nrOfModels, args.parms)

    elif args.nrOfModels ==1:
        freeze_graph(args.checkpoint_path, args.output_nodes, args.output_graph, args.rename_outputs)
    else:
        get_n_models(args.checkpoint_path, args.output_nodes, args.output_graph, args.rename_outputs, args.nrOfModels)

