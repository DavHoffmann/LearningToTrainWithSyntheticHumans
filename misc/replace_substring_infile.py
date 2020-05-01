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

import os

def replaceAll(infilepath, outfilepath, toReplace, replaceBy):
    with open(infilepath) as infile,  open(outfilepath, 'w') as outfile:
        for line in infile:
            outfile.write(line.replace(toReplace, replaceBy))


basepath = '../purelySynthetic/' #change for mixed and stylized
pathToSynthetic = '/ps/project/surreal_multi_person/ltsh/keypoints_smallCol_final/' #please add your path here
files = os.listdir(basepath)

for file in files:
    infile = os.path.join(basepath, file)
    outfile = infile.replace('.json', '_.json')

    replaceAll(infile, outfile, '/dummypath/', pathToSynthetic)
