//
// Created by Xx220xX on 05/05/2020.
//

#ifndef GABRIELAGPUPADA_H
#define GABRIELAGPUPADA_H

#ifdef __cplusdcplus
extern "C"{
#endif


#include <stdio.h>
#include "MatrixGPUPADA.h"

#ifdef GPU_CL
#include "config_gpu_access.h"
#define FUNC_ID_TANH 0
#define FUNC_ID_DFTANH 1
#define FUNC_ID_RELU 2
#define FUNC_ID_DFRELU 3
#define FUNC_ID_SIGMOID 8
#define FUNC_ID_DFSIGMOID 9
#define FUNC_ID_ALAN 16
#define FUNC_ID_DFALAN 17
#define FLAG_DIF 1
#else

#include <math.h>
#include "config_cpu_access.h"
#include "Optime_GPU.cpp"

#endif
typedef struct {
    int L; // last layer
    int arq_0; // neuronios on first layer
    int arq_o;// neuronios on output layer
    Mat *a; // after activate
    Mat *z; // sum(Wij*Ali)
    Mat *w, *b; //  weight and bias
} GAB;
typedef struct {
    int len, *p;
} Vector;
static int activation_type = FUNC_ID_TANH;

int setFuncActivation(int id);

int getFuncActivation();

int call(GAB *g, double *input);

int aprende(GAB *g, double *output, double hitLearn);

#ifdef __cplusdcplus
};
#endif
#endif //GABRIELAGPUPADA_H
