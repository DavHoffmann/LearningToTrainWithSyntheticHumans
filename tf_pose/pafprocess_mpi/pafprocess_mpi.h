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

#include <vector>

#ifndef PAFPROCESS_MPI
#define PAFPROCESS_MPI

const float THRESH_HEAT = 0.05; //param.thre1
const float THRESH_VECTOR_SCORE = 0.01;//param.thre2
const int THRESH_VECTOR_CNT1 = 7; //line 141 suc_ratio but not devidied by 10
const int THRESH_PART_CNT = 3; //param.thre3
const float THRESH_HUMAN_SCORE = 0.2;//param.thre4 on 288 minival
const int NUM_PART = 15;

const int STEP_PAF = 10;

const int COCOPAIRS_SIZE = 14;
const int COCOPAIRS_NET[COCOPAIRS_SIZE][2] = {

{0,  1},  {2,  3},  {4,  5},  {6,  7},  {8,  9}, {10, 11}, {12, 13}, {14, 15}, {16,
       17}, {18, 19}, {20, 21}, {22, 23}, {24, 25}, {26,27}
       };


const int COCOPAIRS[COCOPAIRS_SIZE][2] = {
     {0,     1},
     {1,     2},
     {2,     3},
     {3,     4},
     {1,     5},
     {5,     6},
     {6,     7},
     {1,    14},
     {14,     8},
     {8,     9},
     {9,    10},
     {14,    11},
     {11,    12},
     {12,    13}
};

struct Peak {
    int x;
    int y;
    float score;
    int id;
};

struct VectorXY {
    float x;
    float y;
};

struct ConnectionCandidate {
    int idx1;
    int idx2;
    float score;
    float etc;
};

struct Connection {
    int cid1;
    int cid2;
    float score;
    int peak_id1;
    int peak_id2;
};

int process_paf(int p1, int p2, int p3, float *peaks, int h1, int h2, int h3, float *heatmap, int f1, int f2, int f3, float *pafmap);
int get_num_humans();
int get_part_cid(int human_id, int part_id);
float get_score(int human_id);
int get_part_x(int cid);
int get_part_y(int cid);
float get_part_score(int cid);

#endif
