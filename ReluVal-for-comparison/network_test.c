/*
 ------------------------------------------------------------------
 ** Top contributors:
 **   Shiqi Wang and Suman Jana
 ** This file is part of the ReluVal project.
 ** Copyright (c) 2018-2019 by the authors listed in the file LICENSE
 ** and their institutional affiliations.
 ** All rights reserved.
 -----------------------------------------------------------------
 *
 * This is the main file of ReluVal, here is the usage:
 * ./network_test [property] [network] [target] 
 *      [need to print=0] [test for one run=0] [check mode=0]
 *
 * [property]: the saftety property want to verify
 *
 * [network]: the network want to test with
 *
 * [target]: Wanted label of the property
 *
 * [need to print]: whether need to print the detailed info of each split.
 * 0 is not and 1 is yes. Default value is 0.
 *
 * [test for one run]: whether need to estimate the output range without
 * split. 0 is no, 1 is yes. Default value is 0.
 *
 * [check mode]: normal split mode is 0. Check adv mode is 1.
 * Check adv mode will prevent further splits as long as the depth goes
 * upper than 20 so as to locate the concrete adversarial examples faster.
 * Default value is 0.
 * 
 */


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "split.h"
#include <float.h>

#ifdef DEBUG
#include <fenv.h>
#endif

//extern int thread_tot_cnt;

/* print the progress if getting SIGQUIT */
void sig_handler(int signo)
{

    if (signo == SIGINT) {
        fprintf(stderr, "progress: %d/1024\n", progress);
	fprintf(stderr, "numSplits: %lld\n", numSplits);
    }
	exit(0);
}

void printUsage(char *argv[]) {
    printf("USAGE\n\t%s PROPERTY SUBBED_NNET EPSILON ", argv[0]);
    printf("[OPTIONS]\n\n");
    printf("DESCRIPTION\n");
    printf("\tTries to verify that the output of SUBBED_NNET is within the\n");
    printf("\tinterval [-EPSILON, EPSILON] over the input region defined by\n");
    printf("\tPROPERTY.\n");
    printf("\n");
    printf("OPTIONS\n");
    printf("\t-p PERTURB\n\t\tSpecifies the maximum perturbation allowed when\n");
    printf("\t\tapplicable to the input property (specifically, MNIST and HAR).\n\n");
    printf("\t-t\n\t\tPerforms 3 pixel perturbation for MNIST instead of global\n");
    printf("\t\tperturbation\n\n");
    printf("\t-m DEPTH\n\t\tForces the analysis to make DEPTH splits, and then\n");
    printf("\t\tprints the region verified at each depth.\n\n");
}


int main( int argc, char *argv[])
{

#ifdef DEBUG
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW);
#endif
    signal(SIGINT, sig_handler);
    char *FULL_NET_PATH = NULL;

    int target = 0;
    float epsilon = 0;

    int opt;
    while ((opt = getopt(argc, argv, "p:tm:")) != -1) {
        switch (opt) {
            case 'p':
                perturb = atof(optarg);
                break;
            case 'm':
                RUN_TO_DEPTH = atoi(optarg);
                break;
            case 't':
                MNIST_3PIX = 1;
                break;
            default:
                printf("Invalid option: %c\n", opt);
                printUsage(argv);
                exit(1);
        }
    }

    if (argc - optind != 3) {
        printf("Only recieved %d positional parameters.\n", argc - optind);
        printUsage(argv);
        exit(1);
    }

    PROPERTY = atoi(argv[optind]);
    FULL_NET_PATH = argv[optind + 1];
    epsilon = atof(argv[optind + 2]);


    openblas_set_num_threads(1);

    //clock_t start, end;
    srand((unsigned)time(NULL));
    double time_spent;


    struct NNet* nnet = load_network(FULL_NET_PATH, target);
    nnet->epsilon = epsilon;

    int inputSize    = nnet->inputSize;
    int outputSize   = nnet->outputSize;

    if (RUN_TO_DEPTH >= 0) {
        verified_region_for_depth = malloc(sizeof(struct Interval)*(RUN_TO_DEPTH+1));
        for (int i = 0; i < RUN_TO_DEPTH+1; i++) {
            verified_region_for_depth[i].lower_matrix.data = malloc(sizeof(float)*outputSize);
            verified_region_for_depth[i].upper_matrix.data = malloc(sizeof(float)*outputSize);
            verified_region_for_depth[i].lower_matrix.row = 1;
            verified_region_for_depth[i].lower_matrix.col = outputSize;
            verified_region_for_depth[i].upper_matrix.row = 1;
            verified_region_for_depth[i].upper_matrix.col = outputSize;
            for (int j = 0; j < outputSize; j++) {
                verified_region_for_depth[i].lower_matrix.data[j] = FLT_MAX;
                verified_region_for_depth[i].upper_matrix.data[j] = -FLT_MAX;
            }
        }
    }

    float u[inputSize], l[inputSize];
    load_inputs(PROPERTY, inputSize, u, l, nnet);

    struct Matrix input_upper = {u,1,nnet->inputSize};
    struct Matrix input_lower = {l,1,nnet->inputSize};

    struct Interval input_interval = {input_lower, input_upper};

    float grad_upper[inputSize], grad_lower[inputSize];
    struct Interval grad_interval = {
                (struct Matrix){grad_upper, 1, inputSize},
                (struct Matrix){grad_lower, 1, inputSize}
            };

    if (PROPERTY < 300 || PROPERTY > 499) {
        normalize_input_interval(nnet, &input_interval);
    }


    float o_upper[nnet->outputSize], o_lower[nnet->outputSize];
    struct Interval output_interval = {
                (struct Matrix){o_lower, outputSize, 1},
                (struct Matrix){o_upper, outputSize, 1}
            };

    //if (signal(SIGQUIT, sig_handler) == SIG_ERR)
        //printf("\ncan't catch SIGQUIT\n");

    int n = 0;
    int feature_range_length = 0;
    int split_feature = -1;

    printf("running property %d with network %s with epsilon %.6f\n",\
                PROPERTY, FULL_NET_PATH, epsilon);
    printf("input ranges:\n");

    printMatrix(&input_upper);
    printMatrix(&input_lower);

    for (int i=0;i<inputSize;i++) {

        if (input_interval.upper_matrix.data[i] <
                input_interval.lower_matrix.data[i]) {
            printf("wrong input!\n");
            exit(0);
        }

        if(input_interval.upper_matrix.data[i] !=
                input_interval.lower_matrix.data[i]){
            n++;
        }

    }

    feature_range_length = n;
    int *feature_range = (int*)malloc(n*sizeof(int));

    for (int i=0, n=0;i<nnet->inputSize;i++) {
        if(input_interval.upper_matrix.data[i] !=\
                input_interval.lower_matrix.data[i]){
            feature_range[n] = i;
            n++;
        }
    }

    gettimeofday(&start, NULL);
    int isOverlap = 0;

    if (RUN_TO_DEPTH == 0) {
	forward_prop_interval_equation(nnet, &input_interval, &output_interval, &grad_interval);
    } else {
        isOverlap = direct_run_check(nnet,
                &input_interval, &output_interval,
                &grad_interval, 0, feature_range,
                feature_range_length, split_feature);
    }
   
    printMatrix(&output_interval.lower_matrix);
    printMatrix(&output_interval.upper_matrix);

    gettimeofday(&finish, NULL);
    time_spent = ((float)(finish.tv_sec - start.tv_sec) *\
            1000000 + (float)(finish.tv_usec - start.tv_usec)) /\
            1000000;

    if (isOverlap == 0 && adv_found == 0) {
        fprintf(stderr, "\nNo adv!\n");
    }

    for (int i = 0; i < RUN_TO_DEPTH+1; i++) {
        fprintf(stderr, "depth: %d\n", i);
        fprintMatrix(stderr, &verified_region_for_depth[i].lower_matrix);
        fprintMatrix(stderr, &verified_region_for_depth[i].upper_matrix);
        free(verified_region_for_depth[i].lower_matrix.data);
        free(verified_region_for_depth[i].upper_matrix.data);
    }
    free(verified_region_for_depth);


    fprintf(stderr, "time: %f \n\n\n", time_spent);
    fprintf(stderr, "numSplits: %lld\n", numSplits);

    destroy_network(nnet);
    free(feature_range);

}
