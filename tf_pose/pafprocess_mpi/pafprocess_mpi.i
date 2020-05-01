// Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
// holder of all proprietary rights on this computer program.
// You can only use this computer program if you have closed
// a license agreement with MPG or you get the right to use the computer
// program from someone who is authorized to grant you that right.
// Any use of the computer program without a valid license is prohibited and
// liable to prosecution.
//
// Copyright©2019 Max-Planck-Gesellschaft zur Förderung
// der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
// for Intelligent Systems and the Max Planck Institute for Biological
// Cybernetics. All rights reserved.
//
// Contact: ps-license@tuebingen.mpg.de
// This file is originally from https://github.com/ildoonet/tf-pose-estimation/ and was modified

%module pafprocess_mpi
%{
  #define SWIG_FILE_WITH_INIT
  #include "pafprocess_mpi.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

//%apply (int DIM1, int DIM2, int* IN_ARRAY2) {(int p1, int p2, int *peak_idxs)}
//%apply (int DIM1, int DIM2, int DIM3, float* IN_ARRAY3) {(int h1, int h2, int h3, float *heatmap), (int f1, int f2, int f3, float *pafmap)};
%apply (int DIM1, int DIM2, int DIM3, float* IN_ARRAY3) {(int p1, int p2, int p3, float *peaks), (int h1, int h2, int h3, float *heatmap), (int f1, int f2, int f3, float *pafmap)};
%include "pafprocess_mpi.h"
