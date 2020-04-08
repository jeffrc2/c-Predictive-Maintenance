#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "./include/k2c_include.h"
#include "bintest.h"

#define MTEST 93
#define MHEIGHT 50
#define MWIDTH  25

float maxabs(k2c_tensor *tensor1, k2c_tensor *tensor2) {

    float x = 0;

    float y = 0;

    for(size_t i=0; i<tensor1->numel; i++) {
		printf("Truth: %f Result: %f ", tensor1->array[i], tensor2->array[i]);
        y = fabs(tensor1->array[i]-tensor2->array[i]);
        if (y>x) {
            x=y;
        }
    }
    return x;
}


int main() {
	FILE * fp, tp;
	int i, j;
	float test_input_array[MTEST*MHEIGHT*MWIDTH];
	float truth_array[MTEST];
	
	if ((fp = fopen("seq_array_test_last.txt", "r")) == NULL) {
		printf("Input file not found");
		exit(1);
	}
	for (i = 0; i < MTEST*MHEIGHT*MWIDTH; i++) {
		if (fscanf(fp, "%f", &test_input_array[i]) != 1) {
			printf("Array index %u ", i);
			printf("Input not float");
			exit(1);
		} else if (i%MWIDTH == 0) {
			printf ("%f", test_input_array[i]);
			printf("\n");
		}
		
	}
	if ((fp = fopen("label_array_test_last.txt", "r")) == NULL) {
		printf("Truth file not found");
		exit(1);
	}
	for (i = 0; i < MTEST; i++) {
		if (fscanf(fp, "%f", &truth_array[i]) != 1) {
			printf("Test index %u ", i);
			printf("Input not float");
			exit(1);
		} else if (i%MWIDTH == 0) {
			printf ("%f", truth_array[i]);
			printf("\n");
		}
		
	}
	
	
	fclose(fp);
	
	k2c_tensor test_input[MTEST];
	k2c_tensor test_output[MTEST];
	k2c_tensor test_truth[MTEST];
	
	float output_array[MTEST] = {0,};
	
	for (i = 0; i < MTEST; i++) {
		test_input[i] = (k2c_tensor) {&test_input_array[i*MHEIGHT*MWIDTH],2,1250,{50,25,1,1,1}};
		test_output[i] = (k2c_tensor) {&output_array[i],1,1,{1,1,1,1,1}};
		test_truth[i] = (k2c_tensor) {&truth_array[i],1,1,{1,1,1,1,1}};
	}
	
	
	float errors[MTEST];
	size_t num_tests = 93;
	size_t num_outputs = 1;
	bintest_initialize();
	clock_t t0 = clock();
	for (int i =0; i < MTEST; i++)
		bintest(&test_input[i],&test_output[i]);
	
	clock_t t1 = clock();
	printf("Test time: %e s \n",
           (double)(t1-t0)/(double)CLOCKS_PER_SEC/(double)10);
	for (int i =0; i < MTEST; i++) {
		printf("Test: %u ", i);
		errors[i] = maxabs(&test_truth[i],&test_output[i]);
		printf("Error: %f \n", errors[i]);
	}
	
	float maxerror = errors[0];
    for(i=1; i< MTEST; i++) {
        if (errors[i] > maxerror) {
            maxerror = errors[i];
        }
    }
	
	printf("Max absolute error for 10 tests: %e \n", maxerror);
    bintest_terminate();
    if (maxerror > 1e-05) {
        return 1;
    }
	
	return 0;
}
	

	