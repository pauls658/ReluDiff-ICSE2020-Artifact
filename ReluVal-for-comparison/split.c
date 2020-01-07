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

#include "split.h"

#define AVG_WINDOW 5
#ifndef MAX_THREAD
#define MAX_THREAD -1
#endif
#define MIN_DEPTH_PER_THREAD 5 

int NEED_PRINT = 0;
int RUN_TO_DEPTH = -1;
int input_depth = 0;
int adv_found = 0;
int count = 0;
int thread_tot_cnt  = 0;
int smear_cnt = 0;
long long numSplits = 0;
struct Interval *verified_region_for_depth;

int progress = 0;

int CHECK_ADV_MODE = 0;
int PARTIAL_MODE = 0;

float avg_depth = 50;
float total_avg_depth = 0;
int leaf_num = 0;
float max_depth = 0;

struct timeval start,finish,last_finish;


int check_between(float x, float lower, float upper) {
    if (x > lower && x < upper) {
        return 1;
    }
    return 0;
}

int check_epsilon_concrete(struct NNet *nnet, struct Matrix *output) {
    float epsilon = nnet->epsilon;
        if (!check_between(output->data[nnet->target], -epsilon, epsilon)) {
            // output is not within boundary
            return 1;
        }
    return 0;
}

int check_epsilon_interval(struct NNet *nnet, struct Interval *output) {
    float epsilon = nnet->epsilon;
        if (!check_between(output->lower_matrix.data[nnet->target], -epsilon, epsilon) ||
            !check_between(output->upper_matrix.data[nnet->target], -epsilon, epsilon)) {
            return 1;
        }
    return 0;
}

/*
 * multithread for direct_run_check function
*/
void *direct_run_check_thread(void *args)
{

    struct direct_run_check_args *actual_args = args;

    direct_run_check(actual_args->nnet,\
                    actual_args->input,\
                    actual_args->output,\
                    actual_args->grad,\
                    actual_args->depth,\
                    actual_args->feature_range,\
                    actual_args->feature_range_length,\
                    actual_args->split_feature);

    return NULL;

}


/*
 * Main property check function
 * It Supports multithreads
 * Takes in all input info and network info.
 * outputs the final split result.
 */
int direct_run_check(struct NNet *nnet,\
                    struct Interval *input,\
                    struct Interval *output,\
                    struct Interval *grad,\
                    int depth, int *feature_range,\
                    int feature_range_length,\
                    int split_feature)
{

    pthread_mutex_lock(&lock);

    if (adv_found) {
        pthread_mutex_unlock(&lock);

        return 0;
    }

    pthread_mutex_unlock(&lock);

    //forward_prop_interval_equation_linear2(nnet, input, output, grad);
    forward_prop_interval_equation(nnet, input, output, grad);

    if (depth == 0) {
        fprintf(stderr, "Initial output delta:\n");
        fprintMatrix(stderr, &output->lower_matrix);
        fprintMatrix(stderr, &output->upper_matrix);
    }   


    int isOverlap = check_epsilon_interval(nnet, output);

#ifdef DEPTH_VERIFIED
    pthread_mutex_lock(&lock);
    if (!isOverlap) {
    	fprintf(stderr, "%d\n", depth);
    }
    pthread_mutex_unlock(&lock);
#endif


    if (RUN_TO_DEPTH != -1) {
        pthread_mutex_lock(&lock);
        for (int i = 0; i < nnet->outputSize; i++) {
            verified_region_for_depth[depth].upper_matrix.data[i] = fmax(output->upper_matrix.data[i],
                                                                         verified_region_for_depth[depth].upper_matrix.data[i]);
            verified_region_for_depth[depth].lower_matrix.data[i] = fmin(output->lower_matrix.data[i],
                                                                         verified_region_for_depth[depth].lower_matrix.data[i]);
        }
        pthread_mutex_unlock(&lock);
    }


    if (depth == 10 && isOverlap == 0) {

        pthread_mutex_lock(&lock);

            progress++;

            if(!adv_found){
                //fprintf(stderr, "progress=%d/1024\r", progress);
                printf("progress: [");

                for (int pp=0;pp<50;pp++) {

                    if (pp <= ((float)progress/1024)*50) {
                        printf("=");
                    }
                    else {
                        printf(" ");
                    }
                }

                printf("] %0.2f%%\r", ((float)progress/1024)*100);
                fflush(stdout);
                //fprintf(stderr, "smear_cnt=%d\n", smear_cnt);
            }

            if (PARTIAL_MODE) {
                if (PROPERTY < 300 || PROPERTY > 399) {
                    denormalize_input_interval(nnet, input);
                }
                printMatrix(&input->upper_matrix);
                printMatrix(&input->lower_matrix);
                printf("\n");
            }

        pthread_mutex_unlock(&lock);

        //printf("progress,%d/1024\n", progress);
    }

    if (RUN_TO_DEPTH != -1) {
        if (depth < RUN_TO_DEPTH) {
            isOverlap = split_interval(nnet,
                    input, output,
                    grad, depth, feature_range,\
                    feature_range_length,\
                    split_feature);
        }
    } else if (isOverlap) {
       isOverlap = split_interval(nnet, input, output,\
                            grad, depth, feature_range,\
                            feature_range_length,\
                            split_feature);
    }
    else if (!isOverlap) {
    pthread_mutex_lock(&lock);

        avg_depth -= (avg_depth) / AVG_WINDOW;
        avg_depth += depth / AVG_WINDOW;

    pthread_mutex_unlock(&lock);

        //printf("%f, %f\n", total_avg_depth,avg_depth);

#ifdef DEBUG
    pthread_mutex_lock(&lock);

        total_avg_depth = (total_avg_depth*leaf_num+depth)/(leaf_num+1);
        leaf_num++;

        if (depth > max_depth) {
            max_depth = depth;
        }

    pthread_mutex_unlock(&lock);

        gettimeofday(&finish, NULL);

        float time_spent =\
                ((float)(finish.tv_sec - start.tv_sec) * 1000000 +\
                (float)(finish.tv_usec - start.tv_usec)) / 1000000;
        float time_spent_last =\
                ((float)(last_finish.tv_sec - start.tv_sec) * 1000000 +\
                (float)(last_finish.tv_usec - start.tv_usec)) / 1000000;

        if ((int)(time_spent)%20==0 && (time_spent - time_spent_last)>10) {
            printf("progress: %f%%  total_avg_depth: %f  "
                   "max_depth:%f avg_depth:%f\n", \
                time_spent/pow(2, total_avg_depth+1)/0.000025/4*100, \
                total_avg_depth, max_depth, avg_depth);
            gettimeofday(&last_finish, NULL);
        }

#endif

    }

    if(!adv_found && depth==0){
        printf("progress: [");

        for (int pp=0;pp<50;pp++) {
            printf("=");
        }

        printf("] %0.2f%%\n", 100.00);
    }

    return isOverlap;

}


/*
 * Check the existance of concrete adversarial examples
 * It takes in the network and input ranges.
 * If a concrete adversarial example is found,
 * global adv_found will be set to 1. 
 */
void check_adv(struct NNet* nnet, struct Interval *input)
{

    float a[nnet->inputSize];
    struct Matrix adv = {a, 1, nnet->inputSize};

    for (int i=0;i<nnet->inputSize;i++) {
        float upper = input->upper_matrix.data[i];
        float lower = input->lower_matrix.data[i];
        float middle = (lower+upper)/2;

        a[i] = middle;
    }

    float out[nnet->outputSize];
    struct Matrix output = {out, nnet->outputSize, 1};

    forward_prop(nnet, &adv, &output);

    int is_adv = 0;
    is_adv = check_epsilon_concrete(nnet, &output);

    //printMatrix(&adv);
    //printMatrix(&output);

    if (is_adv) {
        fprintf(stderr, "\nadv found:\n");
        if (PROPERTY < 300 || PROPERTY > 399) {
            denormalize_input(nnet, &adv);
        }
        fprintf(stderr, "adv is: ");
        fprintMatrix(stderr, &adv);
        fprintf(stderr, "it's output is: ");
        fprintMatrix(stderr, &output);
        pthread_mutex_lock(&lock);
        adv_found = 1;
        pthread_mutex_unlock(&lock);
    }

}


/*
 * Split input interval on most influencial input features 
 * and check for monotonocity.
 * 
 */
int split_interval(struct NNet *nnet, struct Interval *input,\
                struct Interval *output, struct Interval *grad,\
                int depth, int *feature_range,\
                int feature_range_length, int split_feature)
{

    int inputSize = nnet->inputSize;

    float input_upper1[nnet->inputSize];
    float input_lower1[nnet->inputSize]; 
    float input_upper2[nnet->inputSize];
    float input_lower2[nnet->inputSize];
    
    pthread_mutex_lock(&lock);
	numSplits++;

    if (adv_found && RUN_TO_DEPTH == -1) {

    pthread_mutex_unlock(&lock);

        return 0;
    }
    
    pthread_mutex_unlock(&lock);
   
    memcpy(input_upper1, input->upper_matrix.data,\
        sizeof(float)*inputSize);
    memcpy(input_upper2, input->upper_matrix.data,\
        sizeof(float)*inputSize);
    memcpy(input_lower1, input->lower_matrix.data,\
        sizeof(float)*inputSize);
    memcpy(input_lower2, input->lower_matrix.data,\
        sizeof(float)*inputSize);
    
    struct Interval input_interval1 = {
            (struct Matrix){input_lower1, 1, nnet->inputSize},
            (struct Matrix){input_upper1, 1, nnet->inputSize}
        };
    struct Interval input_interval2 = {
            (struct Matrix){input_lower2, 1, nnet->inputSize}, 
            (struct Matrix){input_upper2, 1, nnet->inputSize}
        };

    float o_upper1[nnet->outputSize], o_lower1[nnet->outputSize];
    struct Interval output_interval1 = {
            (struct Matrix){o_lower1, nnet->outputSize, 1},
            (struct Matrix){o_upper1, nnet->outputSize, 1}
        };

    float o_upper2[nnet->outputSize], o_lower2[nnet->outputSize];
    struct Interval output_interval2 = {
            (struct Matrix){o_lower2, nnet->outputSize, 1},
            (struct Matrix){o_upper2, nnet->outputSize, 1}
        };

    float grad_upper1[inputSize], grad_lower1[inputSize];
    struct Interval grad_interval1 = {
            (struct Matrix){grad_upper1, 1, nnet->inputSize},
            (struct Matrix){grad_lower1, 1, nnet->inputSize}
        };

    float grad_upper2[inputSize], grad_lower2[inputSize];
    struct Interval grad_interval2 = {
            (struct Matrix){grad_upper2, 1, nnet->inputSize},
            (struct Matrix){grad_lower2, 1, nnet->inputSize}
        };

    int feature_range1[feature_range_length];
    memcpy(feature_range1, feature_range,\
        sizeof(int)*feature_range_length);

    int feature_range2[feature_range_length];
    memcpy(feature_range2, feature_range,\
        sizeof(int)*feature_range_length);

    int feature_range_length1 = feature_range_length;
    int feature_range_length2 = feature_range_length;

    depth = depth + 1;
    
    int mono = 0;
    float smear = 0;
    float largest_smear = 0;

    for (int i=0;i<feature_range_length;i++) {

        if (grad->upper_matrix.data[feature_range[i]]<=0 ||\
            grad->lower_matrix.data[feature_range[i]]>=0) {
            mono = 1;

            smear = ((grad->upper_matrix.data[feature_range[i]]>\
                - grad->lower_matrix.data[feature_range[i]])?\
                grad->upper_matrix.data[feature_range[i]]:\
                - grad->lower_matrix.data[feature_range[i]])*\
                (input->upper_matrix.data[feature_range[i]]-\
                input->lower_matrix.data[feature_range[i]]);

            if (smear >= largest_smear) {
                largest_smear = smear;
                split_feature = i;
            }
            
        }

    }

    if (mono == 1) {
        feature_range_length1 = feature_range_length - 1;
        feature_range_length2 = feature_range_length - 1;

        for (int j=1;j<feature_range_length-split_feature;j++) {
            feature_range1[split_feature+j-1] = feature_range[split_feature+j];
            feature_range2[split_feature+j-1] = feature_range[split_feature+j];
        }

        input_lower1[feature_range[split_feature]] =\
                     input_upper1[feature_range[split_feature]] =\
                     input->upper_matrix.data[feature_range[split_feature]];
        input_lower2[feature_range[split_feature]] =\
                     input_upper2[feature_range[split_feature]] =\
                     input->lower_matrix.data[feature_range[split_feature]];

        if (feature_range_length1 == 0) {
            check_adv(nnet, &input_interval1);
            check_adv(nnet, &input_interval2);
            return 0;
        }

    }

    if (mono == 0) {
        float smear = 0;
        float largest_smear = 0;
        float smear_sum=0;
        float interval_range[feature_range_length];
        float e=0;

        for (int i=0;i<feature_range_length;i++) {
            interval_range[i] = input->upper_matrix.data[feature_range[i]]-\
                                input->lower_matrix.data[feature_range[i]];

            e = (grad->upper_matrix.data[feature_range[i]]>\
                -grad->lower_matrix.data[feature_range[i]])?\
                grad->upper_matrix.data[feature_range[i]]:\
                -grad->lower_matrix.data[feature_range[i]];

            smear = e*interval_range[i];
            smear_sum += smear;

            if (largest_smear< smear) {
                largest_smear = smear;
                split_feature = i;
            }

        }


        float upper = input->upper_matrix.data[feature_range[split_feature]];
        float lower = input->lower_matrix.data[feature_range[split_feature]];

        float middle;

        if (upper != lower) {
            middle = (upper + lower) / 2;
        }
        else {
            middle = upper;
        }
        
        
        //if (depth >= 40) {
            //printMatrix(&input->upper_matrix);
            //printMatrix(&input->lower_matrix);
        //}
        /*
        if (smear_sum <= 0.02) {
            //printf("tighten:  smear_sum: %f  depth:%d  output:", smear_sum, depth);

            if (tighten_still_overlap(nnet, input, smear_sum) == 0) {
                //printf("0\n");

                pthread_mutex_lock(&lock);

                    smear_cnt ++;

                pthread_mutex_unlock(&lock);

                return 0;
            }

            //printf("1\n");

        }
	*/
        if (CHECK_ADV_MODE) {

            if (depth >= 25 || upper-middle <= ADV_THRESHOLD) {
                check_adv(nnet, input);
                return 0;
            }

        }
        else {

            if ((depth >= 35 || upper-middle <= ADV_THRESHOLD) && RUN_TO_DEPTH == -1) {

#ifdef DEBUG
        if (depth >= 50) {
            printf("check for depth!\n"); 
        }
        else {
            printf("check for thershold\n");
        }
#endif
                check_adv(nnet, input);
            }

        }

        input_lower1[feature_range[split_feature]] = middle;
        input_upper2[feature_range[split_feature]] = middle;
    }

    
    pthread_mutex_lock(&lock);

    if ((depth <= avg_depth - MIN_DEPTH_PER_THREAD) &&\
            (count<=MAX_THREAD)) {

    pthread_mutex_unlock(&lock);

        pthread_t workers1, workers2;
        struct direct_run_check_args args1 = {
                                    nnet, &input_interval1,
                                    &output_interval1, &grad_interval1,
                                    depth, feature_range1,
                                    feature_range_length1, split_feature
                                };

        struct direct_run_check_args args2 = {
                                    nnet, &input_interval2,
                                    &output_interval2, &grad_interval2,
                                    depth, feature_range2,
                                    feature_range_length2, split_feature
                                };

        pthread_create(&workers1, NULL, direct_run_check_thread, &args1);

    pthread_mutex_lock(&lock);

        count++;
        thread_tot_cnt++;

    pthread_mutex_unlock(&lock);

        //printf ( "pid1: %ld start %d \n", syscall(SYS_gettid), count);

        pthread_create(&workers2, NULL, direct_run_check_thread, &args2);

    pthread_mutex_lock(&lock);

        count++;
        thread_tot_cnt++;

    pthread_mutex_unlock(&lock);

        //printf ( "pid2: %ld start %d \n", syscall(SYS_gettid), count);

        pthread_join(workers1, NULL);

    pthread_mutex_lock(&lock);

        count--;

    pthread_mutex_unlock(&lock);

        //printf ( "pid1: %ld done %d\n",syscall(SYS_gettid), count);

        pthread_join(workers2, NULL);
    
    pthread_mutex_lock(&lock);

        count--;

    pthread_mutex_unlock(&lock);

        //printf ( "pid2: %ld done %d\n", syscall(SYS_gettid), count);

        if (depth == 11) {

            pthread_mutex_lock(&lock);

                progress++;

                if(!adv_found){
                    //fprintf(stderr, "progress=%d/1024\r", progress);
                    printf("progress: [");

                    for (int pp=0;pp<50;pp++) {

                        if (pp <= ((float)progress/1024)*50) {
                            printf("=");
                        }
                        else {
                            printf(" ");
                        }

                    }

                    printf("] %0.2f%%\r", ((float)progress/1024)*100);
                    fflush(stdout);
                }
                
                //fprintf(stderr, "smear_cnt=%d\n", smear_cnt);

                if (PARTIAL_MODE) {
                    denormalize_input_interval(nnet, input);
                    printMatrix(&input->upper_matrix);
                    printMatrix(&input->lower_matrix);
                    printf("\n");
                }

            pthread_mutex_unlock(&lock);
        }

        return 0;
    }
    else {

    pthread_mutex_unlock(&lock);

        int isOverlap1 = direct_run_check(nnet, &input_interval1, 
                                     &output_interval1, &grad_interval1,
                                     depth, feature_range1, 
                                     feature_range_length1, split_feature);

        int isOverlap2 = direct_run_check(nnet, &input_interval2, 
                                     &output_interval2, &grad_interval2, 
                                     depth, feature_range2,
                                     feature_range_length2, split_feature);

        int result = isOverlap1 || isOverlap2;

        if (result == 0 && depth == 11) {

        pthread_mutex_lock(&lock);

            progress++;

            if(!adv_found){
                //fprintf(stderr, "progress=%d/1024\r", progress);
                printf("progress: [");

                for (int pp=0;pp<50;pp++) {

                    if (pp <= ((float)progress/1024)*50) {
                        printf("=");
                    }
                    else {
                        printf(" ");
                    }

                }

                printf("] %0.2f%%\r", ((float)progress/1024)*100);
                fflush(stdout);
            }

            //fprintf(stderr, "smear_cnt=%d\n", smear_cnt);

            if (PARTIAL_MODE) {
                denormalize_input_interval(nnet, input);
                printMatrix(&input->upper_matrix);
                printMatrix(&input->lower_matrix);
                printf("\n");
            }

        pthread_mutex_unlock(&lock);

        }

        return result;
    }
    
}
