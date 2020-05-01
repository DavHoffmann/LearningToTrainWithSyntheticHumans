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

from distutils.core import setup, Extension
import numpy
import os

# os.environ['CC'] = 'g++';
setup(name='pafprocess_ext', version='1.0',
    ext_modules=[
        Extension('_pafprocess_mpi', ['pafprocess_mpi.cpp', 'pafprocess_mpi.i'],
                  swig_opts=['-c++'],
                  depends=["pafprocess_mpi.h"],
                  include_dirs=[numpy.get_include(), '.'])
    ],
    py_modules=[
        "pafprocess_mpi"
    ]
)
