/*
 ------------------------------------------------------------------
 ** Top contributors:
 **   Shiqi Wang and Suman Jana
 ** This file is part of the ReluVal project.
 ** Copyright (c) 2018-2019 by the authors listed in the file LICENSE
 ** and their institutional affiliations.
 ** All rights reserved.
 -----------------------------------------------------------------
 */


#include "nnet.h"
#include <math.h>
#include <fenv.h>
#include "mnist_tests.h"
#include "HAR_tests.h"

// Not used anymore, but needed to compile
#define OUTWARD_ROUND 0.00005

int PROPERTY = 5;
float perturb = 0.0;
int MNIST_3PIX = 0;

void *memset(void *ptr, int value, size_t numBytes);
void *memcpy(void *to, const void *from, size_t numBytes);
void computeAllBounds(float *eqLow, float *eqUp,
                      struct Interval *input, int inputSize,
                      float *low, float *lowsUp, float *upsLow, float *up);


/*
 * Load_network is a function modified from Reluplex
 * It takes in a nnet filename with path and load the
 * network from the file
 * Outputs the NNet instance of loaded network.
 */
struct NNet *load_network(const char* filename, int target)
{

    FILE *fstream = fopen(filename,"r");

    if (fstream == NULL) {
        printf("Wrong network!\n");
        exit(1);
    }

    /* Largest line is ~39K bytes (for the CIFAR networks),
     * but allocate a bit extra just in case... */
    int bufferSize = 50240;
    char *buffer = (char*)malloc(sizeof(char)*bufferSize);
    char *record, *line;
    int i=0, layer=0, row=0, j=0, param=0;

    struct NNet *nnet = (struct NNet*)malloc(sizeof(struct NNet));

    nnet->target = target;

    line=fgets(buffer,bufferSize,fstream);

    /* Skip comments in the beginning */
    while (strstr(line, "//") != NULL) {
        line = fgets(buffer,bufferSize,fstream);
    }

    /* First four inputs are number of layers, number of inputs,
     * number of outputs, and maximum layer size. */
    record = strtok(line,",\n");
    nnet->numLayers = atoi(record);
    nnet->inputSize = atoi(strtok(NULL,",\n"));
    nnet->outputSize = atoi(strtok(NULL,",\n"));
    nnet->maxLayerSize = atoi(strtok(NULL,",\n"));

    /* Read array sizes */
    nnet->layerSizes = (int*)malloc(sizeof(int)*(nnet->numLayers+1));
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0;i<((nnet->numLayers)+1);i++) {
        nnet->layerSizes[i] = atoi(record);
        record = strtok(NULL,",\n");
    }

    /* Deprecated property... */
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    nnet->symmetric = atoi(record);

    /* Mins, maxes, means, and ranges used for normalizing inputs...*/
    /* Read mins */
    nnet->mins = (float*)malloc(sizeof(float)*nnet->inputSize);
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0;i<(nnet->inputSize);i++) {
        nnet->mins[i] = (float)atof(record);
        record = strtok(NULL,",\n");
    }

    /* Read maxes */
    nnet->maxes = (float*)malloc(sizeof(float)*nnet->inputSize);
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0;i<(nnet->inputSize);i++) {
        nnet->maxes[i] = (float)atof(record);
        record = strtok(NULL,",\n");
    }

    /* Read means */
    nnet->means = (float*)malloc(sizeof(float)*(nnet->inputSize+1));
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0;i<((nnet->inputSize)+1);i++) {
        nnet->means[i] = (float)atof(record);
        record = strtok(NULL,",\n");
    }

    /* Read ranges */
    nnet->ranges = (float*)malloc(sizeof(float)*(nnet->inputSize+1));
    line = fgets(buffer,bufferSize,fstream);
    record = strtok(line,",\n");
    for (i = 0;i<((nnet->inputSize)+1);i++) {
        nnet->ranges[i] = (float)atof(record);
        record = strtok(NULL,",\n");
    }

    /* Alloc memory for reading in weights and biases */
    nnet->matrix = (float****)malloc(sizeof(float *)*nnet->numLayers);
    for (layer = 0;layer<(nnet->numLayers);layer++) {
        nnet->matrix[layer] =\
                (float***)malloc(sizeof(float *)*2);
        nnet->matrix[layer][0] =\
                (float**)malloc(sizeof(float *)*nnet->layerSizes[layer+1]);
        nnet->matrix[layer][1] =\
                (float**)malloc(sizeof(float *)*nnet->layerSizes[layer+1]);

        for (row = 0;row<nnet->layerSizes[layer+1];row++) {
            nnet->matrix[layer][0][row] =\
                    (float*)malloc(sizeof(float)*nnet->layerSizes[layer]);
            nnet->matrix[layer][1][row] = (float*)malloc(sizeof(float));
        }

    }

    layer = 0;
    param = 0;
    i=0;
    j=0;

    char *tmpptr=NULL;

    float w = 0.0;
    /* Read weights and biases */
    while ((line = fgets(buffer,bufferSize,fstream)) != NULL) {

        if (i >= nnet->layerSizes[layer+1]) {

            if (param==0) {
                param = 1;
            }
            else {
                param = 0;
                layer++;
            }

            i=0;
            j=0;
        }

        record = strtok_r(line,",\n", &tmpptr);

        while (record != NULL) {
            w = (float)atof(record);
            nnet->matrix[layer][param][i][j] = w;
            j++;
            record = strtok_r(NULL, ",\n", &tmpptr);
        }

        tmpptr=NULL;
        j=0;
        i++;
    }


    /* Copy weights and biases into Matrix structs */
    struct Matrix *weights=malloc(nnet->numLayers*sizeof(struct Matrix));
    struct Matrix *bias = malloc(nnet->numLayers*sizeof(struct Matrix));
    for (int layer=0;layer<nnet->numLayers;layer++) {
        weights[layer].row = nnet->layerSizes[layer];
        weights[layer].col = nnet->layerSizes[layer+1];
        weights[layer].data = (float*)malloc(sizeof(float)\
                    * weights[layer].row * weights[layer].col);

        int n=0;


        for (int i=0;i<weights[layer].col;i++) {

            for (int j=0;j<weights[layer].row;j++) {
                weights[layer].data[n] = nnet->matrix[layer][0][i][j];
                n++;
            }

        }

        bias[layer].col = nnet->layerSizes[layer+1];
        bias[layer].row = (float)1;
        bias[layer].data = (float*)malloc(sizeof(float)*bias[layer].col);

        for (int i=0;i<bias[layer].col;i++) {
            bias[layer].data[i] = nnet->matrix[layer][1][i][0];
        }



    }

    nnet->weights = weights;
    nnet->bias = bias;

    free(buffer);
    fclose(fstream);

    return nnet;

}


/*
 * Subtracts the weights of nnet2 from the weights of nnet1.
 */
void compute_network_delta(struct NNet *nnet1, struct NNet *nnet2) {
    for (int k = 0; k < nnet1->numLayers; k++) {
        struct Matrix new = nnet1->weights[k];
        struct Matrix orig = nnet2->weights[k];
        for (int i = 0; i < new.col*new.row; i++) {
            new.data[i] -= orig.data[i];
        }

        new = nnet1->bias[k];
        orig = nnet2->bias[k];
        for (int i = 0; i < new.col*new.row; i++) {
            new.data[i] -= orig.data[i];
        }
    }
}

/*
 * Allocates memory for and loads the positive/negative weights.
 */
void load_positive_and_negative_weights(struct NNet *nnet) {
    struct Matrix *posWeights = malloc(nnet->numLayers*sizeof(struct Matrix));
    struct Matrix *negWeights = malloc(nnet->numLayers*sizeof(struct Matrix));
    struct Matrix weights;
    for (int layer=0;layer<nnet->numLayers;layer++) {
        weights = nnet->weights[layer];

        posWeights[layer].row = weights.row;
        posWeights[layer].col = weights.col;
        negWeights[layer].row = weights.row;
        negWeights[layer].col = weights.col;
        posWeights[layer].data = (float *) malloc(sizeof(float) * weights.row * weights.col);
        negWeights[layer].data = (float *) malloc(sizeof(float) * weights.row * weights.col);
        memset(posWeights[layer].data, 0, sizeof(float) * weights.row * weights.col);
        memset(negWeights[layer].data, 0, sizeof(float) * weights.row * weights.col);

        for(int i=0; i < weights.row * weights.col; i++) {
            if (weights.data[i] >= 0) {
                posWeights[layer].data[i] = weights.data[i];
            } else {
                negWeights[layer].data[i] = weights.data[i];
            }
        }
    }
    nnet->posWeights = posWeights;
    nnet->negWeights = negWeights;
}

/*
 * destroy_network is a function modified from Reluplex
 * It release all the memory allocated to the network instance
 * It takes in the instance of nnet
 */
void destroy_network(struct NNet *nnet)
{

    int i=0, row=0;
    if (nnet != NULL) {

        for (i=0;i<(nnet->numLayers);i++) {

            for (row=0;row<nnet->layerSizes[i+1];row++) {
                free(nnet->matrix[i][0][row]);
                free(nnet->matrix[i][1][row]);
            }

            free(nnet->matrix[i][0]);
            free(nnet->matrix[i][1]);
            free(nnet->weights[i].data);
            free(nnet->posWeights[i].data);
            free(nnet->negWeights[i].data);
            free(nnet->bias[i].data);
            free(nnet->matrix[i]);
        }

        free(nnet->weights);
        free(nnet->posWeights);
        free(nnet->negWeights);
        free(nnet->bias);
        free(nnet->layerSizes);
        free(nnet->mins);
        free(nnet->maxes);
        free(nnet->means);
        free(nnet->ranges);
        free(nnet->matrix);
        free(nnet);
    }

}


/*
 * Load the inputs of all the predefined properties
 * It takes in the property and input pointers
 */
void load_inputs(int PROPERTY, int inputSize, float *u, float *l, struct NNet *nnet)
{
    // HAR properties
    if (PROPERTY >= 1000 && PROPERTY < 1099) {
        int testNum = PROPERTY - 1000;
        for (int i = 0; i < 561; i++) {
            u[i] = fmin((HAR_test[testNum][i] + perturb), 1.0);
            l[i] = fmax((HAR_test[testNum][i] - perturb), -1.0);
        }
        nnet->target = HAR_correct_class[testNum];
    }

    if (!MNIST_3PIX) {
	    // MNIST properties global exp
	    if (PROPERTY >= 400 && PROPERTY <= 499) {
		    int testNum = PROPERTY - 400;
		    for (int i = 0; i < 784; i++) {
			    u[i] = fmin((mnist_test[testNum][i] + perturb)/255.0, 1.0);
			    l[i] = fmax((mnist_test[testNum][i] - perturb)/255.0, 0.0);
		    }
		    nnet->target = correct_class[testNum];
		    return;
	    } 
    } else {
	    // MNIST properties pixel exp
	    if (PROPERTY >= 400 && PROPERTY <= 499) {
		    // pixel experiment
		    int num_pixels = 3;
		    int testNum = PROPERTY - 400;
		    for (int i = 0; i < 784; i++) {
			    u[i] = mnist_test[testNum][i]/255.0;
			    l[i] = mnist_test[testNum][i]/255.0;
		    }
		    for (int i = 0; i < num_pixels; i++) {
			    u[random_pixels[testNum][i]] = 1.0;
			    l[random_pixels[testNum][i]] = 0.0;
		    }
		    nnet->target = correct_class[testNum];
		    return;
	    }
    }

    if (PROPERTY == 301) {
        float upper[] = {1.5, 0.5};
        float lower[] = {0.5, -0.5};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 0;
        return;
    }

    if (PROPERTY == 300 || PROPERTY == 302) {
        float upper[] = {6.0, 5.0};
        float lower[] = {4.0, 1.0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 0;
        return;
    }

    if (PROPERTY == 303) {
        float upper[] = {1.0, 1.0};
        float lower[] = {-1.0, 0.0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 0;
        return;
    }

    if (PROPERTY == 304) {
        float upper[] = {1.0, 2.0};
        float lower[] = {0.0, 1.0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 0;
        return;
    }


    if (PROPERTY == 1) {
        float upper[] = {60760,3.141592,3.141592,1200,60};
        float lower[] = {55947.691,-3.141592,-3.141592,1145,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 0;
    }

    if (PROPERTY == 2) {
        float upper[] = {60760,3.141592,3.141592, 1200, 60};
        float lower[] = {55947.691,-3.141592,-3.141592,1145,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 0;
    }

    if (PROPERTY == 3) {
        float upper[] = {1800,0.06,3.141592,1200,1200};
        float lower[] = {1500,-0.06,3.10,980,960};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 0;
    }

    if (PROPERTY == 4) {
        float upper[] = {1800,0.06,0,1200,800};
        float lower[] = {1500,-0.06,0,1000,700};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 0;
    }

    if (PROPERTY == 5) {
        float upper[] = {400,0.4,-3.1415926+0.005,400,400};
        float lower[] = {250,0.2,-3.1415926,100,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 4;
    }

    if (PROPERTY==16) {
        float upper[] = {62000,-0.7,-3.141592+0.005,200,1200};
        float lower[] = {12000,-3.141592,-3.141592,100,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 0;

    }

    if (PROPERTY==26) {
        float upper[] = {62000,3.141592,-3.141592+0.005,200,1200};
        float lower[] = {12000,0.7,-3.141592,100,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 0;
    }

    if (PROPERTY==7) {
        float upper[] = {60760,3.141592,3.141592,1200,1200};
        float lower[] = {0,-3.141592,-3.141592,100,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 4;
    }

    if (PROPERTY==8) {
        float upper[] = {60760,-3.141592*0.75,0.1,1200,1200};
        float lower[] = {0,-3.141592,-0.1,600,600};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 1;
    }

    if (PROPERTY==9) {
        float upper[] = {7000,-0.14,-3.141592+0.01,150,150};
        float lower[] = {2000,-0.4,-3.141592,100,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 3;
    }

    if (PROPERTY==10) {
        float upper[] = {60760,3.141592,-3.141592+0.01,1200,1200};
        float lower[] = {36000,0.7,-3.141592,900,600};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 0;
    }

    if (PROPERTY == 11) {
        float upper[] = {400,0.4,-3.1415926+0.005,400,400};
        float lower[] = {250,0.2,-3.1415926,100,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 4;
    }

    if (PROPERTY == 12) {
        float upper[] = {60760,3.141592,3.141592, 1200, 60};
        float lower[] = {55947.691,-3.141592,-3.141592,1145,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 0;
    }

    if (PROPERTY == 13) {
        float upper[] = {60760,3.141592,3.141592, 360, 360};
        float lower[] = {60000,-3.141592,-3.141592,0,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 0;
    }

    if (PROPERTY == 14) {
        float upper[] = {400,0.4,-3.1415926+0.005,400,400};
        float lower[] = {250,0.2,-3.1415926,100,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 4;
    }

    if (PROPERTY == 15) {
        float upper[] = {400,-0.2,-3.1415926+0.005,400,400};
        float lower[] = {250,-0.4,-3.1415926,100,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
        nnet->target = 3;
    }

    if (PROPERTY == 100) {
        float upper[] = {400,0,-3.1415926+0.025,250,200};
        float lower[] = {250,0,-3.1415926+0.025,250,200};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 101) {
        float upper[] = {400,0.4,-3.1415926+0.025,250,200};
        float lower[] = {250,0.2,-3.1415926+0.025,250,200};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 102) {
        float upper[] = {400,0.2,-3.1415926+0.05,0,0};
        float lower[] = {250,0.2,-3.1415926+0.05,0,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 110) {
        float upper[] = {10000,3.141592,-3.141592+0.01,1200,1200};
        float lower[] = {1000,3.141592,-3.141592+0.01,1200,1200};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 111) {
        float upper[] = {1000,3.141592,-3.141592+0.01,1200,1200};
        float lower[] = {1000,3.141592,-3.141592+0.01,0,1200};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }

    if (PROPERTY == 112) {
        float upper[] = {1000,3.141592,-3.141592+0.01,1200,1200};
        float lower[] = {1000,3.141592,-3.141592+0.01,1200,0};
        memcpy(u, upper, sizeof(float)*inputSize);
        memcpy(l, lower, sizeof(float)*inputSize);
    }



}



/*
 * Following functions denomalize and normalize the concrete inputs
 * and input intervals.
 * They take in concrete inputs or input intervals.
 * Output normalized or denormalized concrete inputs or input intervals.
 */
void denormalize_input(struct NNet *nnet, struct Matrix *input)
{

    for (int i=0; i<nnet->inputSize;i++) {
        input->data[i] = input->data[i]*(nnet->ranges[i]) + nnet->means[i];
    }

}


void denormalize_input_interval(struct NNet *nnet, struct Interval *input)
{

    denormalize_input(nnet, &input->upper_matrix);
    denormalize_input(nnet, &input->lower_matrix);

}


void normalize_input(struct NNet *nnet, struct Matrix *input)
{

    for (int i=0;i<nnet->inputSize;i++) {

        if (input->data[i] > nnet->maxes[i]) {
            input->data[i] = (nnet->maxes[i]-nnet->means[i])/(nnet->ranges[i]);
        }
        else if (input->data[i] < nnet->mins[i]) {
            input->data[i] = (nnet->mins[i]-nnet->means[i])/(nnet->ranges[i]);
        }
        else {
            input->data[i] = (input->data[i]-nnet->means[i])/(nnet->ranges[i]);
        }

    }

}


void normalize_input_interval(struct NNet *nnet, struct Interval *input)
{

    normalize_input(nnet, &input->upper_matrix);
    normalize_input(nnet, &input->lower_matrix);

}


/*
 * Concrete forward propagation with openblas
 * It takes in network and concrete input matrix.
 * Outputs the concrete outputs.
 */
int forward_prop(struct NNet *network, struct Matrix *input, struct Matrix *output)
{

    int layer;

    struct NNet* nnet = network;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;


    float z[nnet->maxLayerSize];
    float a[nnet->maxLayerSize];
    struct Matrix Z = {z, 1, inputSize};
    struct Matrix A = {a, 1, inputSize};

    memcpy(Z.data, input->data, nnet->inputSize*sizeof(float));

    for(layer=0;layer<numLayers;layer++){
        A.row = nnet->bias[layer].row;
        A.col = nnet->bias[layer].col;
        memcpy(A.data, nnet->bias[layer].data, A.row*A.col*sizeof(float));

        matmul_with_bias(&Z, &nnet->weights[layer], &A);
        if(layer<numLayers-1){
            relu(&A);
        }
        memcpy(Z.data, A.data, A.row*A.col*sizeof(float));
        Z.row = A.row;
        Z.col = A.col;

    }

    memcpy(output->data, A.data, A.row*A.col*sizeof(float));
    output->row = A.row;
    output->col = A.col;

    return 1;
}


int forward_prop_delta(struct NNet *nnet1, struct NNet *nnet2, struct Matrix *input, struct Matrix *output) {
    float o1[nnet1->outputSize];
    float o2[nnet1->outputSize];
    struct Matrix output1 = {o1, nnet1->outputSize, 1};
    struct Matrix output2 = {o2, nnet1->outputSize, 1};

    forward_prop(nnet1, input, &output1);
    forward_prop(nnet2, input, &output2);

    for (int i = 0; i < nnet1->outputSize; i++) {
        output->data[i] = output2.data[i] - output1.data[i];
    }
    return 1;
}


void backward_prop(struct NNet *nnet, struct Interval *grad,
                       int R[][nnet->maxLayerSize], int subtract) {
    int i, j, layer;
    int numLayers = nnet->numLayers;
    int inputSize = nnet->inputSize;
    int maxLayerSize = nnet->maxLayerSize;

    float gradLower[maxLayerSize];
    float gradUpper[maxLayerSize];
    float newGradLower[maxLayerSize];
    float newGradUpper[maxLayerSize];

    /* Initialize gradient with weights of last (output) layer */
    memcpy(gradLower, nnet->matrix[numLayers - 1][0][nnet->target],\
             sizeof(float) * nnet->layerSizes[numLayers-1]);
    memcpy(gradUpper, nnet->matrix[numLayers - 1][0][nnet->target],\
             sizeof(float) * nnet->layerSizes[numLayers-1]);

    if (subtract) {
        for (i = 0; i < nnet->layerSizes[numLayers-1]; i++) {
            gradLower[i] = -gradLower[i];
            gradUpper[i] = -gradUpper[i];
        }
    }

    /* Process second to last layer to first layer */
    for (layer = numLayers - 2; layer >= 0; layer--) {
        float **weights = nnet->matrix[layer][0];
        memset(newGradUpper, 0, sizeof(float) * maxLayerSize);
        memset(newGradLower, 0, sizeof(float) * maxLayerSize);

        if (layer != 0) { // general case
            /* For each neuron in the _current_ layer
             * (layerSizes includes the input layer, hence the layer + 1).
             * In the nnet->weights matrix, layer is the _current_ layer.
             * In the nnet->layerSizes, layer is the _previous_ layer. */
            for (j = 0; j < nnet->layerSizes[layer+1]; j++) {
                /* In backward propagation, we first perform ReLU, then do matrix mult */
                /* Perform ReLU */
                if (R[layer][j] == 0) {
                    /* both slopes where 0 */
                    gradUpper[j] = gradLower[j] = 0;
                } else if (R[layer][j] == 1) {
                    /* This neuron could be either linear (the original equations) or 0.
                     * We have to ensure 0 is in the gradient. */
                    /* min(grad_lower, 0) */
                    gradLower[j] = (gradLower[j] < 0) ? gradLower[j] : 0;
                    /* max(grad_upper, 0) */
                    gradUpper[j] = (gradUpper[j] > 0) ? gradUpper[j] : 0;
                } else {
                    /* Both upper and lower slopes are linear, no need to do anything. */
                }

                /* Perform matrix multiplication */
                /* For each neuron in the _previous_ layer,
                 * add the _current_ neuron's gradient to it.
                 * layer is the _previous_ layer in layerSizes */
                for (i = 0; i < nnet->layerSizes[layer]; i++) {
                    if (weights[j][i] >= 0) {
                        /* Weight is positive
                         * Lower to lower, upper to upper */
                        newGradLower[i] += weights[j][i] * gradLower[j];
                        newGradUpper[i] += weights[j][i] * gradUpper[j];
                    } else {
                        /* Else flip */
                        newGradLower[i] += weights[j][i] * gradUpper[j];
                        newGradUpper[i] += weights[j][i] * gradLower[j];
                    }
                }
            }

        } else { /* Input layer */
            /* For each neuron in second layer */
            for (j = 0; j < nnet->layerSizes[layer + 1]; j++) {
                /* Perform ReLU */
                if (R[layer][j] == 0) {
                    gradUpper[j] = gradLower[j] = 0;
                } else if (R[layer][j] == 1) {
                    /* min(gradLower[j], 0) */
                    gradLower[j] = (gradLower[j] < 0) ? gradLower[j] : 0;
                    /* max(gradUpper[j], 0) */
                    gradUpper[j] = (gradUpper[j] > 0) ? gradUpper[j] : 0;
                } else {
                    /* Neuron was linear, no need to do anything */
                }

                /* For each input neuron's gradient, add the current neuron's
                 * gradient to it. */
                for (i = 0; i < inputSize; i++) {
                    if (weights[j][i] >= 0) { /* Weight is positive */
                        newGradLower[i] += weights[j][i] * gradLower[j];
                        newGradUpper[i] += weights[j][i] * gradUpper[j];
                    } else {
                        newGradLower[i] += weights[j][i] * gradUpper[j];
                        newGradUpper[i] += weights[j][i] * gradLower[j];
                    }
                }
            }
        }


        if (layer != 0) {
            /* layer is the _previous_ layer in the layerSizes array */
            memcpy(gradLower, newGradLower, sizeof(float) * nnet->layerSizes[layer]);
            memcpy(gradUpper, newGradUpper, sizeof(float) * nnet->layerSizes[layer]);
        } else {
            memcpy(grad->lower_matrix.data, newGradLower, sizeof(float) * inputSize);
            memcpy(grad->upper_matrix.data, newGradUpper, sizeof(float) * inputSize);
        }
    }
}


void computeBounds(float *new_equation_lower, float *new_equation_upper,
                   struct Interval *input, int inputSize, int i, float *bounds) {
    float tempVal_lower = 0.0, tempVal_upper = 0.0;
    for(int k=0;k<inputSize;k++){
        if(new_equation_lower[k+i*(inputSize+1)]>=0){
            tempVal_lower += new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k]-OUTWARD_ROUND;
        }
        else{
            tempVal_lower += new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k]-OUTWARD_ROUND;
        }
        if(new_equation_upper[k+i*(inputSize+1)]>=0){
            tempVal_upper += new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k]+OUTWARD_ROUND;
        }
        else{
            tempVal_upper += new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k]+OUTWARD_ROUND;
        }
    }
    tempVal_lower += new_equation_lower[inputSize+i*(inputSize+1)];
    tempVal_upper += new_equation_upper[inputSize+i*(inputSize+1)];
    bounds[0] = tempVal_lower;
    bounds[1] = tempVal_upper;
}

/* Not used currently.
 * TODO: implement outward rounding.
 */
int forward_prop_delta_concrete(struct NNet *network1, struct NNet *network2,\
            struct Interval *input, struct Interval *output,\
            struct Interval *grad, struct Interval *outputDelta)
{
    int i,k,layer;

    struct NNet* nnet = network1;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;

    int maxLayerSize   = nnet->maxLayerSize;

    int R[numLayers][maxLayerSize];
    memset(R, 0, sizeof(float)*numLayers*maxLayerSize);

    // equation is the temp equation for each layer
    // Each arrays is divided into sections of size inputSize+1.
    // The coefficient of input k for neuron i is stored at
    // index k + i*(inputSize + 1), and the constant value is stored
    // at k = inputSize.
    float equation_upper[(inputSize+1)*maxLayerSize];
    float equation_lower[(inputSize+1)*maxLayerSize];
    float new_equation_upper[(inputSize+1)*maxLayerSize];
    float new_equation_lower[(inputSize+1)*maxLayerSize];

    // the upper and lower concrete bounds on the deltas
    float delta_equation_upper[maxLayerSize];
    float delta_equation_lower[maxLayerSize];
    float new_delta_equation_upper[maxLayerSize];
    float new_delta_equation_lower[maxLayerSize];
    float n_hat[2];

    memset(equation_upper,0,sizeof(float)*(inputSize+1)*maxLayerSize);
    memset(equation_lower,0,sizeof(float)*(inputSize+1)*maxLayerSize);

    memset(delta_equation_upper,0,sizeof(float)*maxLayerSize);
    memset(delta_equation_lower,0,sizeof(float)*maxLayerSize);

    struct Interval equation_inteval = {
                (struct Matrix){(float*)equation_lower, inputSize+1, inputSize},
                (struct Matrix){(float*)equation_upper, inputSize+1, inputSize}
            };
    struct Interval new_equation_inteval = {
                (struct Matrix){(float*)new_equation_lower, inputSize+1, maxLayerSize},
                (struct Matrix){(float*)new_equation_upper, inputSize+1, maxLayerSize}
            };

    float tempVal_upper=0.0, tempVal_lower=0.0;
    float upper_s_lower=0.0, lower_s_upper=0.0;

    float tempDelta_upper = 0.0, tempDelta_lower = 0.0;

    for (i=0; i < nnet->inputSize; i++)
    {
        equation_lower[i*(inputSize+1)+i] = 1;
        equation_upper[i*(inputSize+1)+i] = 1;
    }

    for (layer = 0; layer<(numLayers); layer++)
    {

        memset(new_equation_upper, 0, sizeof(float)*(inputSize+1)*maxLayerSize);
        memset(new_equation_lower, 0, sizeof(float)*(inputSize+1)*maxLayerSize);

        struct Matrix weights = nnet->weights[layer];
        struct Matrix bias = nnet->bias[layer];
        float p[weights.col*weights.row];
        float n[weights.col*weights.row];
        memset(p, 0, sizeof(float)*weights.col*weights.row);
        memset(n, 0, sizeof(float)*weights.col*weights.row);
        struct Matrix pos_weights = {p, weights.row, weights.col};
        struct Matrix neg_weights = {n, weights.row, weights.col};
        for(i=0;i<weights.row*weights.col;i++){
            if(weights.data[i]>=0){
                p[i] = weights.data[i];
            }
            else{
                n[i] = weights.data[i];
            }
        }

        matmul(&equation_inteval.upper_matrix, &pos_weights, &new_equation_inteval.upper_matrix);
        matmul_with_bias(&equation_inteval.lower_matrix, &neg_weights, &new_equation_inteval.upper_matrix);

        matmul(&equation_inteval.lower_matrix, &pos_weights, &new_equation_inteval.lower_matrix);
        matmul_with_bias(&equation_inteval.upper_matrix, &neg_weights, &new_equation_inteval.lower_matrix);


        for (i=0; i < nnet->layerSizes[layer+1]; i++)
        {
            tempVal_upper = tempVal_lower = 0.0;
            lower_s_upper = upper_s_lower = 0.0;

            // compute bound with outward rounding
            for(k=0;k<inputSize;k++){
                if(new_equation_lower[k+i*(inputSize+1)]>=0){
                    tempVal_lower += new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k]-OUTWARD_ROUND;
                    lower_s_upper += new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k]-OUTWARD_ROUND;
                }
                else{
                    tempVal_lower += new_equation_lower[k+i*(inputSize+1)] * input->upper_matrix.data[k]-OUTWARD_ROUND;
                    lower_s_upper += new_equation_lower[k+i*(inputSize+1)] * input->lower_matrix.data[k]-OUTWARD_ROUND;
                }
                if(new_equation_upper[k+i*(inputSize+1)]>=0){
                    tempVal_upper += new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k]+OUTWARD_ROUND;
                    upper_s_lower += new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k]+OUTWARD_ROUND;
                }
                else{
                    tempVal_upper += new_equation_upper[k+i*(inputSize+1)] * input->lower_matrix.data[k]+OUTWARD_ROUND;
                    upper_s_lower += new_equation_upper[k+i*(inputSize+1)] * input->upper_matrix.data[k]+OUTWARD_ROUND;
                }
            }

            new_equation_lower[inputSize+i*(inputSize+1)] += bias.data[i];
            new_equation_upper[inputSize+i*(inputSize+1)] += bias.data[i];
            tempVal_lower += new_equation_lower[inputSize+i*(inputSize+1)];
            lower_s_upper += new_equation_lower[inputSize+i*(inputSize+1)];
            tempVal_upper += new_equation_upper[inputSize+i*(inputSize+1)];
            upper_s_lower += new_equation_upper[inputSize+i*(inputSize+1)];

            tempDelta_lower = tempDelta_upper = 0.0;
            for (int prevNeuron = 0; prevNeuron < nnet->layerSizes[layer]; prevNeuron++) {
                computeBounds(equation_lower, equation_upper, input, inputSize, prevNeuron, n_hat);
                float w_prime = network2->matrix[layer][0][i][prevNeuron];
                float w_delta = w_prime -\
                                network1->matrix[layer][0][i][prevNeuron];
                // n_delta_hat is at delta_equation_upper[prevNeuron] and delta_equation_lower[prevNeuron]
                if (w_delta == 0) {
                    // No change.  No need to add in the outward rounding error
                } else if (w_delta > 0) {
                    // upper goes to upper, lower goes to lower
                    tempDelta_upper += w_delta * n_hat[1] + OUTWARD_ROUND;
                    tempDelta_lower += w_delta * n_hat[0] - OUTWARD_ROUND;
                } else {
                    // upper and lower are flipped
                    tempDelta_upper += w_delta * n_hat[0] + OUTWARD_ROUND;
                    tempDelta_lower += w_delta * n_hat[1] - OUTWARD_ROUND;
                }

                if (w_prime >= 0) {
                    // upper goes to upper, lower goes to lower
                    tempDelta_upper += w_prime * delta_equation_upper[prevNeuron] + OUTWARD_ROUND;
                    tempDelta_lower += w_prime * delta_equation_lower[prevNeuron] - OUTWARD_ROUND;
                } else {
                    // upper and lower are flipped
                    tempDelta_lower += w_prime * delta_equation_upper[prevNeuron] - OUTWARD_ROUND;
                    tempDelta_upper += w_prime * delta_equation_lower[prevNeuron] + OUTWARD_ROUND;
                }
            }
            // add bias
            tempDelta_lower += (network2->bias[layer].data[i] - bias.data[i]);
            tempDelta_upper += (network2->bias[layer].data[i] - bias.data[i]);

            if(layer<(numLayers-1)) {
                //Perform ReLU for delta
                if (tempVal_lower >= 0 && tempVal_lower + tempDelta_lower >= 0) {
                    // always greater than 0
                    new_delta_equation_lower[i] = tempDelta_lower;
                    new_delta_equation_upper[i] = tempDelta_upper;
                } else if (tempVal_upper <= 0 && tempVal_upper + tempDelta_upper <= 0) {
                    // always less than 0
                    new_delta_equation_lower[i] = 0.0;
                    new_delta_equation_upper[i] = 0.0;
                } else {
                    // could be either... ensure 0 is in the interval
                    new_delta_equation_lower[i] = fminf(0.0, tempDelta_lower);
                    new_delta_equation_upper[i] = fmaxf(0.0, tempDelta_upper);
                }

                //Perform ReLU
                if (tempVal_upper<=0.0){
                    tempVal_upper = 0.0;
                    for(k=0;k<inputSize+1;k++){
                        new_equation_upper[k+i*(inputSize+1)] = 0;
                        new_equation_lower[k+i*(inputSize+1)] = 0;
                    }
                    R[layer][i] = 0;
                }
                else if(tempVal_lower>=0.0 ){
                    R[layer][i] = 2;
                }
                else{
                    if(lower_s_upper>0 || upper_s_lower<0){
                        //printf("%d,%d:%f, %f, %f, %f\n",layer, i, tempVal_lower, lower_s_upper, upper_s_lower, tempVal_upper );
                    }
                    //printf("wrong node: ");
                    if(upper_s_lower<0.0){
                        for(k=0;k<inputSize+1;k++){
                            new_equation_upper[k+i*(inputSize+1)] =\
                                                    new_equation_upper[k+i*(inputSize+1)]*\
                                                    tempVal_upper / (tempVal_upper-upper_s_lower);
                        }
                        new_equation_upper[inputSize+i*(inputSize+1)] -= tempVal_upper*upper_s_lower/\
                                                            (tempVal_upper-upper_s_lower);
                    }

                    if(lower_s_upper<0.0){
                        for(k=0;k<inputSize+1;k++){
                            new_equation_lower[k+i*(inputSize+1)] = 0;
                        }
                    }
                    else{
                        /*
                        if(lower_s_upper<-tempVal_lower){
                            for(k=0;k<inputSize+1;k++){
                                new_equation_lower[k+i*(inputSize+1)] = 0;
                            }
                        }
                        */
                        for(k=0;k<inputSize+1;k++){
                            new_equation_lower[k+i*(inputSize+1)] =\
                                                    new_equation_lower[k+i*(inputSize+1)]*\
                                                    lower_s_upper / (lower_s_upper- tempVal_lower);
                        }


                    }
                    R[layer][i] = 1;
                }
            }
            else{
                output->upper_matrix.data[i] = tempVal_upper;
                output->lower_matrix.data[i] = tempVal_lower;
                outputDelta->upper_matrix.data[i] = tempDelta_upper;
                outputDelta->lower_matrix.data[i] = tempDelta_lower;
            }
        }

        memcpy(equation_upper, new_equation_upper, sizeof(float)*(inputSize+1)*maxLayerSize);
        memcpy(equation_lower, new_equation_lower, sizeof(float)*(inputSize+1)*maxLayerSize);
        memcpy(delta_equation_upper, new_delta_equation_upper, sizeof(float)*maxLayerSize);
        memcpy(delta_equation_lower, new_delta_equation_lower, sizeof(float)*maxLayerSize);
        equation_inteval.lower_matrix.row = equation_inteval.upper_matrix.row =\
                                                         new_equation_inteval.lower_matrix.row;
        equation_inteval.lower_matrix.col = equation_inteval.upper_matrix.col =\
                                                         new_equation_inteval.lower_matrix.col;
    }

    backward_prop(nnet, grad, R, 0);

    return 1;
}


/* Computes the bounds for an equation with terms of input variables */
void computeAllBounds(float *eqLow, float *eqUp,
                      struct Interval *input, int inputSize,
                      float *low, float *lowsUp, float *upsLow, float *up) {
    float tempVal_lower = 0, tempVal_upper = 0, lower_s_upper = 0, upper_s_lower = 0;

    fesetround(FE_UPWARD); // Compute upper bounds
    for (int k = 0; k < inputSize; k++) {
        /* lower's upper bound */
        if (eqLow[k] >= 0) {
            /* If coefficient is positive, multiply it by the upper bound
             * of the input. */
            lower_s_upper += eqLow[k] * input->upper_matrix.data[k];
        } else {
            /* Otherwise, multiply by lower bound of the input. */
            lower_s_upper += eqLow[k] * input->lower_matrix.data[k];
        }
        /* upper bound */
        if (eqUp[k] >= 0) {
            tempVal_upper += eqUp[k] * input->upper_matrix.data[k];
        } else {
            tempVal_upper += eqUp[k] * input->lower_matrix.data[k];
        }
    }
    lower_s_upper += eqLow[inputSize];
    tempVal_upper += eqUp[inputSize];

    fesetround(FE_DOWNWARD); // Compute lower bounds
    for (int k = 0; k < inputSize; k++) {
        /* lower bound */
        if (eqLow[k] >= 0) {
            /* If coefficient is positive, multiply by lower bound
             * of the input. */
            tempVal_lower += eqLow[k] * input->lower_matrix.data[k];
        } else {
            /* Otherwise, multiply by upper bound. */
            tempVal_lower += eqLow[k] * input->upper_matrix.data[k];
        }
        /* upper's lower bound */
        if(eqUp[k] >= 0) {
            upper_s_lower += eqUp[k] * input->lower_matrix.data[k];
        } else {
            upper_s_lower += eqUp[k] * input->upper_matrix.data[k];
        }
    }
    tempVal_lower += eqLow[inputSize];
    upper_s_lower += eqUp[inputSize];

    *low = tempVal_lower;
    *lowsUp = lower_s_upper;
    *upsLow = upper_s_lower;
    *up = tempVal_upper;
}

void zero_interval(struct Interval *interval, int eqSize, int neuron) {
    for (int k = 0; k < eqSize; k++) {
        interval->lower_matrix.data[k + neuron*(eqSize)] = 0;
        interval->upper_matrix.data[k + neuron*(eqSize)] = 0;
    }
}

/*
 * Implements the ReLUVal style concretization of a neuron's interval equation.
 * In particular, the lower bound is always concretized to zero, but the upper
 * bound can be left alone if the lower bound of the upper bound's eqaution is
 * greater than 0.
 */
void concretizeNeuronEq(struct Interval *interval, int eqSize, int neuron, float upsLower, float up) {
    int eqOffset = neuron*eqSize;
    if (upsLower < 0.0) { // concretize
        zero_interval(interval, eqSize, neuron);
        interval->upper_matrix.data[eqSize - 1 + eqOffset] = up;
    } else { // upper bound is linear, so leave it alone
        for (int k = 0; k < eqSize; k++) {
            interval->lower_matrix.data[k + eqOffset] = 0.0;
        }
    }
}

/*
 * Performs an affine multiplication transform to the array of input intervals.
 */
void affineTransform(struct Interval *interval, struct Matrix *posMatrix, struct Matrix *negMatrix,
        struct Interval *outInterval, int overWrite) {
    if (overWrite) {
        /* source neuron --weight--> dest neuron */
        fesetround(FE_UPWARD); // compute upper bound
        /* For source neurons with a positive incoming weight, we multiply
         * by the source neuron's upper bound to maximize it. */
        matmul(          &interval->upper_matrix, posMatrix, &outInterval->upper_matrix);
        /* For source neurons with a negative incoming weight, we multiply
         * by the source neuron's lower bound to minimize it. */
        matmul_with_bias(&interval->lower_matrix, negMatrix, &outInterval->upper_matrix);

        fesetround(FE_DOWNWARD); // compute lower bound
        /* Use the lower bound when multiplying by a positive weight to minimize */
        matmul(          &interval->lower_matrix, posMatrix, &outInterval->lower_matrix);
        /* Use the upper bound when multiplying by negative weight to minimize */
        matmul_with_bias(&interval->upper_matrix, negMatrix, &outInterval->lower_matrix);
    } else {
        fesetround(FE_UPWARD);
        matmul_with_bias(&interval->upper_matrix, posMatrix, &outInterval->upper_matrix);
        matmul_with_bias(&interval->lower_matrix, negMatrix, &outInterval->upper_matrix);
        fesetround(FE_DOWNWARD);
        matmul_with_bias(&interval->lower_matrix, posMatrix, &outInterval->lower_matrix);
        matmul_with_bias(&interval->upper_matrix, negMatrix, &outInterval->lower_matrix);
    }
}


/*
 * Implements the ReLUDiff algorithm using ReLUVal to compute the neuron's absolute bounds.
 */
int forward_prop_delta_symbolic(struct NNet *network1, struct NNet *network2, struct NNet *deltas,\
            struct Interval *input, struct Interval *output,\
            struct Interval *grad, struct Interval *outputDelta)
{
    int i,k,layer,eqOffset,constantIndex;

    struct NNet* nnet = network1;
    struct NNet* nnetPrime = network2;
    int numLayers    = nnet->numLayers;
    int inputSize    = nnet->inputSize;

    int maxLayerSize = nnet->maxLayerSize;

    int R[numLayers][maxLayerSize];
    int RPrime[numLayers][maxLayerSize];
    memset(R, 0, sizeof(float)*numLayers*maxLayerSize);
    memset(RPrime, 0, sizeof(float)*numLayers*maxLayerSize);

    // equation is the temp equation for each layer
    // Each arrays is divided into sections of size inputSize+1.
    // The coefficient of input k for neuron i is stored at
    // index k + i*(inputSize + 1), and the constant value is stored
    // at k = inputSize.
        // Initialize equation matrices
        int equationSize = inputSize + 1; // number of variables in equation
        // number of bytes needed to store matrix of equations
        int maxEquationMatrixSize = equationSize * nnet->maxLayerSize * sizeof(float);
        float *eqLow = malloc(maxEquationMatrixSize);
        float *eqUp = malloc(maxEquationMatrixSize);
        float *newEqLow = malloc(maxEquationMatrixSize);
        float *newEqUp = malloc(maxEquationMatrixSize);

        float *eqPrimeLow = malloc(maxEquationMatrixSize);
        float *eqPrimeUp = malloc(maxEquationMatrixSize);
        float *newEqPrimeLow = malloc(maxEquationMatrixSize);
        float *newEqPrimeUp = malloc(maxEquationMatrixSize);

        float *eqDeltaLow = malloc(maxEquationMatrixSize);
        float *eqDeltaUp = malloc(maxEquationMatrixSize);
        float *newEqDeltaLow = malloc(maxEquationMatrixSize);
        float *newEqDeltaUp = malloc(maxEquationMatrixSize);

        memset(eqUp, 0, maxEquationMatrixSize);
        memset(eqLow, 0, maxEquationMatrixSize);
        memset(newEqUp, 0, maxEquationMatrixSize);
        memset(newEqLow, 0, maxEquationMatrixSize);

        memset(eqPrimeUp, 0, maxEquationMatrixSize);
        memset(eqPrimeLow, 0, maxEquationMatrixSize);
        memset(newEqPrimeUp, 0, maxEquationMatrixSize);
        memset(newEqPrimeLow, 0, maxEquationMatrixSize);

        memset(eqDeltaUp, 0, maxEquationMatrixSize);
        memset(eqDeltaLow, 0, maxEquationMatrixSize);
        memset(newEqDeltaUp, 0, maxEquationMatrixSize);
        memset(newEqDeltaLow, 0, maxEquationMatrixSize);


    struct Interval eqInterval = {
            (struct Matrix){(float*)eqLow, equationSize, inputSize},
            (struct Matrix){(float*)eqUp, equationSize, inputSize}
    };
    // the row/col initialization doesn't matter for the new equations
    // because it gets overwritten in the matmul functions
    struct Interval newEqInterval = {
            (struct Matrix){(float*)newEqLow, equationSize, maxLayerSize},
            (struct Matrix){(float*)newEqUp, equationSize, maxLayerSize}
    };
    struct Interval eqPrimeInterval = {
            (struct Matrix){(float*)eqPrimeLow, equationSize, inputSize},
            (struct Matrix){(float*)eqPrimeUp, equationSize, inputSize}
    };
    struct Interval newEqPrimeInterval = {
            (struct Matrix){(float*)newEqPrimeLow, equationSize, maxLayerSize},
            (struct Matrix){(float*)newEqPrimeUp, equationSize, maxLayerSize}
    };
    struct Interval eqDeltaInterval = {
            (struct Matrix){(float*)eqDeltaLow, equationSize, inputSize},
            (struct Matrix){(float*)eqDeltaUp, equationSize, inputSize}
    };
    struct Interval newEqDeltaInterval = {
            (struct Matrix){(float*)newEqDeltaLow, equationSize, maxLayerSize},
            (struct Matrix){(float*)newEqDeltaUp, equationSize, maxLayerSize}
    };

    float concUp = 0.0, concLow = 0.0;
    float concUpsLow = 0.0, concLowsUp = 0.0;

    float concPrimeUp = 0.0, concPrimeLow = 0.0;
    float concPrimeUpsLow = 0.0, concPrimeLowsUp = 0.0;

    float concDeltaLow = 0.0, concDeltaUp = 0.0;
    float tmp; // for unused parameters

    /* Initialize the equations for each input neuron.
     * Note that the delta equation starts as all 0's. */
    for (i=0; i < nnet->inputSize; i++)
    {
        eqLow[i*equationSize+i] = 1;
        eqUp[i*equationSize+i] = 1;
        eqPrimeLow[i*equationSize+i] = 1;
        eqPrimeUp[i*equationSize+i] = 1;
    }

    for (layer = 0; layer<(numLayers); layer++)
    {
        memset(newEqUp, 0, maxEquationMatrixSize);
        memset(newEqLow, 0, maxEquationMatrixSize);
        memset(newEqPrimeUp, 0, maxEquationMatrixSize);
        memset(newEqPrimeLow, 0, maxEquationMatrixSize);
        memset(newEqDeltaLow, 0, maxEquationMatrixSize);
        memset(newEqDeltaUp, 0, maxEquationMatrixSize);

        struct Matrix bias = nnet->bias[layer];
        struct Matrix biasPrime = nnetPrime->bias[layer];
        struct Matrix biasDelta = deltas->bias[layer];

        struct Matrix pWeights = nnet->posWeights[layer];
        struct Matrix nWeights = nnet->negWeights[layer];
        struct Matrix pWeightsPrime = nnetPrime->posWeights[layer];
        struct Matrix nWeightsPrime = nnetPrime->negWeights[layer];
        struct Matrix pWeightsDelta = deltas->posWeights[layer];
        struct Matrix nWeightsDelta = deltas->negWeights[layer];

        // Apply weights to symbolic intervals

        /* nnet */
        affineTransform(&eqInterval, &pWeights, &nWeights, &newEqInterval, 1);

        /* nnetPrime */
        affineTransform(&eqPrimeInterval, &pWeightsPrime, &nWeightsPrime, &newEqPrimeInterval, 1);

        /* nnetDelta */
        affineTransform(&eqInterval, &pWeightsDelta, &nWeightsDelta, &newEqDeltaInterval, 1);
        affineTransform(&eqDeltaInterval, &pWeightsPrime, &nWeightsPrime, &newEqDeltaInterval, 0);

                        // +1 because this array includes input size
        for (i = 0; i < nnet->layerSizes[layer+1]; i++) {
            concUp = concLow = 0.0;
            concLowsUp = concUpsLow = 0.0;

            concPrimeLow = concPrimeUp = 0.0;
            concPrimeLowsUp = concPrimeUpsLow = 0.0;

            concDeltaLow = concDeltaUp = 0.0;

            eqOffset = i*equationSize;
            constantIndex = eqOffset + equationSize-1;

            /* Add bias to the constant */
            fesetround(FE_DOWNWARD); // lower bounds
            newEqLow[constantIndex] += bias.data[i];
            newEqPrimeLow[constantIndex] += biasPrime.data[i];
            newEqDeltaLow[constantIndex] += biasDelta.data[i];
            fesetround(FE_UPWARD); // upper bounds
            newEqUp[constantIndex] += bias.data[i];
            newEqPrimeUp[constantIndex] += biasPrime.data[i];
            newEqDeltaUp[constantIndex] += biasDelta.data[i];

            computeAllBounds(newEqLow + eqOffset, newEqUp + eqOffset,
                    input, inputSize, &concLow, &concLowsUp, &concUpsLow, &concUp);
            computeAllBounds(newEqPrimeLow + eqOffset, newEqPrimeUp + eqOffset,
                    input, inputSize, &concPrimeLow, &concPrimeLowsUp, &concPrimeUpsLow, &concPrimeUp);
            computeAllBounds(newEqDeltaLow + eqOffset, newEqDeltaUp + eqOffset,
                    input, inputSize, &concDeltaLow, &tmp, &tmp, &concDeltaUp);


            if (layer < (numLayers-1)) { // ReLU Transform

                if (concUp <= 0.0) { // eq is 0
                    R[layer][i] = 0;
                    zero_interval(&newEqInterval, equationSize, i);

                    if (concPrimeUp <= 0.0) { // case 1, both are zero
                        RPrime[layer][i] = 0;

                        zero_interval(&newEqPrimeInterval, equationSize, i);

                        zero_interval(&newEqDeltaInterval, equationSize, i);

                    } else if (concPrimeLow >= 0.0) { // case 2, eqPrime is linear
                        RPrime[layer][i] = 2;
                        /* No need to change eqPrime */
                        /* newEqDelta = newEqPrime - 0 */
                        for (k = eqOffset; k < eqOffset + equationSize; k++) {
                            newEqDeltaLow[k] = newEqPrimeLow[k];
                            newEqDeltaUp [k] = newEqPrimeUp [k];
                        }
                        /* Brandon: Why are we assigning newEqDelta and then zeroing it? */
                        //zero_interval(&newEqDeltaInterval, equationSize, i);


                    } else { // case 3, eqPrime is non-linear
                        RPrime[layer][i] = 1;

                        concretizeNeuronEq(&newEqPrimeInterval, equationSize, i, concPrimeUpsLow, concPrimeUp);

                        /* lower bound is always 0 */
                        if (concPrimeUpsLow >= 0.0) {
                            /* upper bound is linear, so save the equation. */
                            for (k = eqOffset; k < eqOffset + equationSize; k++) {
                                newEqDeltaLow[k] = 0.0;
                                newEqDeltaUp[k] = newEqPrimeUp[k];
                            }
                        } else {
                            /* otherwise both a concrete */
                            zero_interval(&newEqDeltaInterval, equationSize, i);
                            newEqDeltaLow[constantIndex] = 0.0;
                            newEqDeltaUp[constantIndex] = concPrimeUp;
                        }
                    }

                } else if (concLow >= 0.0) { // eq is linear
                    R[layer][i] = 2;
                    /* no need to do anything to eq */

                    if (concPrimeUp <= 0.0) { // case 4, eqPrime is 0
                        RPrime[layer][i] = 0;

                        zero_interval(&newEqPrimeInterval, equationSize, i);

                        for (k = eqOffset; k < eqOffset + equationSize; k++) {
                            newEqDeltaLow[k] = -newEqUp[k];
                            newEqDeltaUp[k] = -newEqLow[k];
                        }

                    } else if (concPrimeLow >= 0.0) {// case 5, both are linear, we can do nothing
                        RPrime[layer][i] = 2;

                    } else {// case 6
                        RPrime[layer][i] = 1;
                        concretizeNeuronEq(&newEqPrimeInterval, equationSize, i, concPrimeUpsLow, concPrimeUp);

                        zero_interval(&newEqDeltaInterval, equationSize, i);
                        newEqDeltaLow[constantIndex] = fmax(-concUp, concDeltaLow);
                        newEqDeltaUp[constantIndex] = fmax(-concLow, concDeltaUp);
                    }
                }
                else { // eq is non-linear
                    R[layer][i] = 1;
                    concretizeNeuronEq(&newEqInterval, equationSize, i, concUpsLow, concUp);

                    if (concPrimeUp <= 0.0) { // case 7, eq prime is 0
                        RPrime[layer][i] = 0;

                        zero_interval(&newEqPrimeInterval, equationSize, i);

                        if (concUpsLow >= 0.0) {
                            for (k = eqOffset; k < eqOffset + equationSize; k++) {
                                newEqDeltaLow[k] = -newEqUp[k];
                                newEqDeltaUp[k] = 0.0;
                            }
                        } else {
                            zero_interval(&newEqDeltaInterval, equationSize, i);
                            newEqDeltaLow[constantIndex] = -concUp;
                            newEqDeltaUp[constantIndex] = 0;
                        }

                    } else if (concPrimeLow >= 0.0) { // case 8
                        RPrime[layer][i] = 2;
                        // eqprime is linear, no need to do anything to it

                        zero_interval(&newEqDeltaInterval, equationSize, i);
						newEqDeltaLow[constantIndex] = fmin(concPrimeLow, concDeltaLow);
						newEqDeltaUp[constantIndex] = fmin(concPrimeUp, concDeltaUp);

                    } else {// case 9
                        RPrime[layer][i] = 1;

                        concretizeNeuronEq(&newEqPrimeInterval, equationSize, i, concPrimeUpsLow, concPrimeUp);

                        zero_interval(&newEqDeltaInterval, equationSize, i);
                        if (concDeltaLow > 0)
                            newEqDeltaLow[constantIndex] = 0;
                        else
                            newEqDeltaLow[constantIndex] = fmax(concDeltaLow, -concUp);
                        newEqDeltaUp[constantIndex] = fmax(0, concDeltaUp);

                    }
                }

            } else {
                output->lower_matrix.data[i] = concLow;
                output->upper_matrix.data[i] = concUp;
                outputDelta->lower_matrix.data[i] = concDeltaLow;
                outputDelta->upper_matrix.data[i] = concDeltaUp;
            }

        }

        memcpy(eqLow, newEqLow, maxEquationMatrixSize);
        memcpy(eqUp, newEqUp, maxEquationMatrixSize);
        memcpy(eqPrimeLow, newEqPrimeLow, maxEquationMatrixSize);
        memcpy(eqPrimeUp, newEqPrimeUp, maxEquationMatrixSize);
        memcpy(eqDeltaLow, newEqDeltaLow, maxEquationMatrixSize);
        memcpy(eqDeltaUp, newEqDeltaUp, maxEquationMatrixSize);

        eqInterval.lower_matrix.row = eqInterval.upper_matrix.row =\
                                                         newEqInterval.lower_matrix.row;
        eqInterval.lower_matrix.col = eqInterval.upper_matrix.col =\
                                                         newEqInterval.lower_matrix.col;
        eqPrimeInterval.lower_matrix.row = eqPrimeInterval.upper_matrix.row =\
                                                         newEqPrimeInterval.lower_matrix.row;
        eqPrimeInterval.lower_matrix.col = eqPrimeInterval.upper_matrix.col =\
                                                         newEqPrimeInterval.lower_matrix.col;
        eqDeltaInterval.lower_matrix.row = eqDeltaInterval.upper_matrix.row =\
                                                         newEqDeltaInterval.lower_matrix.row;
        eqDeltaInterval.lower_matrix.col = eqDeltaInterval.upper_matrix.col =\
                                                         newEqDeltaInterval.lower_matrix.col;
    }

    backward_prop(nnet, grad, R, 0);
    float grad_lower[inputSize], grad_upper[inputSize];
    struct Interval grad_interval = {
            (struct Matrix){grad_lower, 1, inputSize},
            (struct Matrix){grad_upper, 1, inputSize}
    };
    backward_prop(nnetPrime, &grad_interval, RPrime, 1);
    for (k = 0; k < inputSize; k++) {
        grad->lower_matrix.data[k] += grad_interval.lower_matrix.data[k];
        grad->upper_matrix.data[k] += grad_interval.upper_matrix.data[k];
    }


    free(eqLow);
    free(eqUp);
    free(newEqLow);
    free(newEqUp);

    free(eqPrimeLow);
    free(eqPrimeUp);
    free(newEqPrimeLow);
    free(newEqPrimeUp);

    free(eqDeltaLow);
    free(eqDeltaUp);
    free(newEqDeltaLow);
    free(newEqDeltaUp);

    return 1;
}

