/*
 -----------------------------------------------------------------
 ** Top contributors:
 **   Shiqi Wang and Suman Jana
 ** This file is part of the ReluVal project.
 ** Copyright (c) 2018-2019 by the authors listed in the file LICENSE
 ** and their institutional affiliations.
 ** All rights reserved.
 -----------------------------------------------------------------
 */


#include "matrix.h"
#include <string.h>
#include "interval.h"

#ifndef NNET_H
#define NNET_H

/* which property to test */
extern int PROPERTY;
extern float perturb;
extern int MNIST_3PIX;

/* log file */
extern char *LOG_FILE;
extern FILE *fp;

typedef int bool;
enum { false, true };


/*
 * Network instance modified from Reluplex
 * malloc all the memory needed for network
 */
struct NNet 
{
    int symmetric;     
    int numLayers;     
    int inputSize;     
    int outputSize;    
    int maxLayerSize;  
    int *layerSizes;   

    float *mins;      
    float *maxes;     
    float *means; 
    float *ranges;
    /*
     * first dimension: the layer (k)
     * second dimension: is bias (0 = no, 1 = yes)
     * third dimension: neuron in layer (k)
     * fourth dimension: source neuron in layer (k - 1)
     */
    float ****matrix;
                       
    struct Matrix* weights;
    struct Matrix* posWeights;
    struct Matrix* negWeights;
    struct Matrix* bias;

    int target;
    int *feature_range;
    int feature_range_length;
    int split_feature;

    float epsilon;
    float perturb;

    float *equation_upper;
    float *equation_lower;
    float *new_equation_upper;
    float *new_equation_lower;
};


/* load the network from file */
struct NNet *load_network(const char *filename, int target);

/*
 * Subtracts the weights of nnet2 from nnet1.
 */
void compute_network_delta(struct NNet *nnet1, struct NNet *nnet2);

/*
 * Allocates the arrays for positive and negative weights, and
 * fills these weight matrices.
 */
void load_positive_and_negative_weights(struct NNet *nnet);

/* free all the memory for the network */
void destroy_network(struct NNet *network);


/* load the input range of the property */
void load_inputs(int PROPERTY, int inputSize, float *u, float *l, struct NNet *nnet);


/* denormalize input */
void denormalize_input(struct NNet *nnet, struct Matrix *input);


/* denormalize input range */
void denormalize_input_interval(struct NNet *nnet, struct Interval *input);


/* normalize input */
void normalize_input(struct NNet *nnet, struct Matrix *input);


/* normalize input range */
void normalize_input_interval(struct NNet *nnet, struct Interval *input);


int evaluate(struct NNet *network, struct Matrix *input, struct Matrix *output);

int forward_prop(struct NNet *network, struct Matrix *input, struct Matrix *output);

void backward_prop(struct NNet *nnet, struct Interval *grad, int R[][nnet->maxLayerSize], int subtract);


/*
 * Computes the difference in output between in nnet1 and nnet2
 * and stores it in the given output matrix.
 * (i.e. output = nnet2 - nnet1)
 */
int forward_prop_delta(struct NNet *nnet1, struct NNet *nnet2, struct Matrix *input, struct Matrix *output);

// Brandon's naive delta computation with concrete bounds and un-optimized loops
int forward_prop_delta_concrete(struct NNet *network1, struct NNet *network2,\
            struct Interval *input, struct Interval *output,\
            struct Interval *grad, struct Interval *outputDelta);

int forward_prop_delta_symbolic(struct NNet *network1, struct NNet *network2, struct NNet *deltas,\
            struct Interval *input, struct Interval *output,\
            struct Interval *grad, struct Interval *outputDelta);

#endif
