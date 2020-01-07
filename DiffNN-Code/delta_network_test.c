#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "split.h"
#include <float.h>

#ifdef DEBUG
    #include <fenv.h>
#endif

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
    printf("USAGE\n\t%s PROPERTY NNET1 NNET2 EPSILON ", argv[0]);
    printf("[OPTIONS]\n\n");
    printf("DESCRIPTION\n");
    printf("\tTries to verify that the outputs of NNET1 and NNET2\n");
    printf("\tcannot differ by more than EPSILON over the input region\n");
    printf("\tdefined by PROPERTY.\n");
    printf("\n");
    printf("OPTIONS\n");
    printf("\t-p PERTURB\n\t\tSpecifies the maximum perturbation allowed when applicable\n\n");
    printf("\t-t\n\t\tPerforms 3 pixel perturbation for MNIST instead of global perturbation\n\n");
    printf("\t-m DEPTH\n\t\tForces the analysis to make DEPTH splits, and then prints the\n");
    printf("\t\tregion verified at each depth.\n\n");
}

int main( int argc, char *argv[])
{

    signal(SIGINT, sig_handler);
    int target = -1;
    float epsilon = 0.0;

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

    if (argc - optind != 4) {
        printf("Only recieved %d positional parameters.\n", argc - optind);
        printUsage(argv);
        exit(1);
    }

    char *FULL_NET_PATH1 = NULL, *FULL_NET_PATH2 = NULL;
    PROPERTY = atoi(argv[optind]);
    FULL_NET_PATH1 = argv[optind + 1];
    FULL_NET_PATH2 = argv[optind + 2];
    epsilon = atof(argv[optind + 3]);


    openblas_set_num_threads(1);


    srand((unsigned)time(NULL));
    double time_spent;

    struct NNet* nnet1 = load_network(FULL_NET_PATH1, target);
    load_positive_and_negative_weights(nnet1);
    struct NNet* nnet2 = load_network(FULL_NET_PATH2, target);
    load_positive_and_negative_weights(nnet2);
    struct NNet* nnetDelta = load_network(FULL_NET_PATH2, target);
    compute_network_delta(nnetDelta, nnet1);
    load_positive_and_negative_weights(nnetDelta);
    nnet1->epsilon = epsilon;


    int inputSize    = nnet1->inputSize;
    int outputSize   = nnet1->outputSize;


    /* load the input intervals, and put them into matrices */
    float u[inputSize], l[inputSize];
    load_inputs(PROPERTY, inputSize, u, l, nnet1);
    nnet2->target = nnet1->target;
    struct Matrix input_upper = {u,1,nnet1->inputSize};
    struct Matrix input_lower = {l,1,nnet1->inputSize};
    struct Interval input_interval = {input_lower, input_upper};

    /* If RUN_TO_DEPTH is set (i.e it is >= 0), then we will perform a fixed number of splits
     * and exit. We record the bounds verified at each depth, so we allocate arrays to record
     * the bounds. */
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

    /* Allocate gradient for the forward interval */
    float grad_upper[inputSize], grad_lower[inputSize];
    struct Interval grad_interval = {
                (struct Matrix){grad_lower, 1, inputSize},
                (struct Matrix){grad_upper, 1, inputSize}
            };


    /* Certain properties should not have their inputs normalized.
     * 300 - 399 are toy properties for our motivating example.
     * 400 - 499 are MNIST properties, which are normalized in the
     * load_inputs function. */
    if (PROPERTY < 300 || PROPERTY > 499) {
        normalize_input_interval(nnet1, &input_interval);
    }


    float oDelta_upper[nnet1->outputSize], oDelta_lower[nnet1->outputSize];
    struct Interval output_delta_interval = {
            (struct Matrix){oDelta_lower, outputSize, 1},
            (struct Matrix){oDelta_upper, outputSize, 1}
    };
    float o_upper[nnet1->outputSize], o_lower[nnet1->outputSize];
    struct Interval output_interval = {
                (struct Matrix){o_lower, outputSize, 1},
                (struct Matrix){o_upper, outputSize, 1}
    };

    printf("running property %d with network %s with epsilon %.6f\n",\
                PROPERTY, FULL_NET_PATH2, epsilon);
    printf("input ranges:\n");
    printMatrix(&input_upper);
    printMatrix(&input_lower);

    int n = 0;
    int feature_range_length;
    int split_feature = -1;
    for (int i=0;i<inputSize;i++) {
        if (input_interval.upper_matrix.data[i] <\
                input_interval.lower_matrix.data[i]) {
            printf("wrong input!\n");
            exit(0);
        }
        if(input_interval.upper_matrix.data[i] !=\
                input_interval.lower_matrix.data[i]){
            n++;
        }

    }

    feature_range_length = n;
    int *feature_range = (int*)malloc(n*sizeof(int));
    for (int i=0, n=0;i<nnet1->inputSize;i++) {
        if(input_interval.upper_matrix.data[i] !=\
                input_interval.lower_matrix.data[i]){
            feature_range[n] = i;
            n++;
        }
    }


    gettimeofday(&start, NULL);
    int isOverlap = 0;

    if (RUN_TO_DEPTH == 0) {
        forward_prop_delta_symbolic(nnet1, nnet2, nnetDelta, &input_interval, &output_interval, &grad_interval, &output_delta_interval);
    } else {
        isOverlap = direct_run_check_delta(
                    nnet1, nnet2, nnetDelta, &input_interval, &output_interval,
                    &output_delta_interval, &grad_interval, 0, feature_range,
                    feature_range_length, split_feature);
    }


    gettimeofday(&finish, NULL);
    time_spent = ((float)(finish.tv_sec - start.tv_sec) *\
            1000000 + (float)(finish.tv_usec - start.tv_usec)) /\
            1000000;

    if (isOverlap == 0 && adv_found == 0) {
        fprintf(stderr, "\nNo adv!\n");
    }

    if (RUN_TO_DEPTH > 0) {
        for (int i = 0; i < RUN_TO_DEPTH+1; i++) {
            fprintf(stderr, "depth: %d\n", i);
            fprintMatrix(stderr, &verified_region_for_depth[i].lower_matrix);
            fprintMatrix(stderr, &verified_region_for_depth[i].upper_matrix);
            free(verified_region_for_depth[i].lower_matrix.data);
            free(verified_region_for_depth[i].upper_matrix.data);
        }
    } else if (RUN_TO_DEPTH == 0) {
        fprintf(stderr, "Output interval: ");
        fprintf(stderr, "[%2.4f, %2.4f]\n",
                output_interval.lower_matrix.data[nnet1->target],
                output_interval.upper_matrix.data[nnet1->target]);
        fprintf(stderr, "Output delta interval: ");
        fprintf(stderr, "[%2.4f, %2.4f]\n", 
                output_delta_interval.lower_matrix.data[nnet1->target],
                output_delta_interval.upper_matrix.data[nnet1->target]);
    }
    free(verified_region_for_depth);


    fprintf(stderr, "time: %f \n\n\n", time_spent);
    fprintf(stderr, "numSplits: %lld\n", numSplits);
    destroy_network(nnet1);
    destroy_network(nnet2);
    destroy_network(nnetDelta);
    free(feature_range);

}
