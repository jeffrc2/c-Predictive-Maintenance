#include <math.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <assert.h>


#include <stdlib.h>

#define MTEST 93
#define MHEIGHT 50
#define MWIDTH  25
#define MAX_NDIM 2

typedef float value;

typedef struct tensor
{
    /** Pointer to array of tensor values. */
    value *array;

    /** Rank of the tensor (number of dimensions). */
    size_t ndim;

    /** Number of elements in the tensor. */
    size_t numel;

    /** Array, size of the tensor in each dimension. */
    size_t shape[MAX_NDIM];
} tensor;

 void writetofile(tensor* weights, char* filename){
	 FILE *filePtr;
	 filePtr = fopen(filename, "w");
	 for (int i = 0; i < weights->numel; i++) {
		 fprintf(filePtr, "%.18e", weights->array[i]);
		 if (i < weights->numel-1) {
			 fprintf(filePtr, "\n");
		 }
	 }
	 fclose(filePtr);
 }

void readfromfile(value* value_array, char* filename, int length)
{
	FILE * fp, tp;
	if ((fp = fopen(filename, "r")) == NULL) {
		printf("File not found: ");
		printf(filename);
		exit(1);
	}
	for (int i = 0; i < length; i++) {
		if (fscanf(fp, "%e", &value_array[i]) != 1) {
			printf(filename);
			printf("\nArray index %u ", i);
			printf("Input not valid");
			exit(1);
		}
	}
}

value hard_sigmoid_func(value x) {
    if (x <= -2.5f) {
        return 0.0f;
    }
    else if (x>=2.5f) {
        return 1.0f;
    }
    else {
        return 0.2f*x + 0.5f;
    }
}

value k2c_sigmoid_func(value x) {

    return 1/(1+exp(-x));
}


void dense(tensor* output, const tensor* input, const tensor* kernel,
               const tensor* bias, value fwork[]) {

	//printf("dense \n ");
    size_t outrows;
    outrows = 1;
    const size_t outcols = kernel->shape[1];
    const size_t innerdim = kernel->shape[0];
	
	memset(output->array, 0, outcols*sizeof(output->array[0]));
	
	for (size_t j = 0;  j < outcols; ++j) {//outcols = units
        for (size_t k = 0; k < innerdim; ++k) {//innerdim = inwidth/units
            output->array[j] += input->array[k] * kernel->array[k*outcols+j]; //W_i/f/c/o*input (2.5k times) or U_i/f/c/o*x_i_f_c_o (10k times) x4 overall for i f c o
        }
		output->array[j] = hard_sigmoid_func(output->array[j] + bias->array[j]);//+bias (50 times));
    }
}

void lstmcell(value state[], const value input[], const tensor* kernel,
                  const tensor* recurrent_kernel, const tensor* bias, value fwork[]) {

    const size_t units = recurrent_kernel->shape[1];
    const size_t in_width = kernel->shape[0]/4;

    value *h_prev = &state[0];  // previous memory state
    value *c_prev = &state[units];  // previous carry state
    const size_t outrows = 1;
    const value * const Wi = &kernel->array[0];
    const value * const Wf = &kernel->array[in_width*units];
    const value * const Wc = &kernel->array[2*in_width*units];
    const value * const Wo = &kernel->array[3*in_width*units];
    const value * const Ui = &recurrent_kernel->array[0];
    const value * const Uf = &recurrent_kernel->array[units*units];
    const value * const Uc = &recurrent_kernel->array[2*units*units];
    const value * const Uo = &recurrent_kernel->array[3*units*units];
    const value * const bi = &bias->array[0];
    const value * const bf = &bias->array[units];
    const value * const bc = &bias->array[2*units];
    value *xi = &fwork[0];
    const value * const bo = &bias->array[3*units];                                                                                                             
    value *xf = &fwork[units];
    value *xc = &fwork[2*units];
    value *xo = &fwork[3*units];
    value *yi = &fwork[4*units];
    value *yf = &fwork[5*units];
    value *yc = &fwork[6*units];
    value *yo = &fwork[7*units];
	
	//printf("lstm_cell \n ");
	//printf("units: %zd\n ", units); //100 for lstm1, 50 for lstm2
	//printf("in_width: %zd\n ", in_width); //25 for lstm1, 100 for lstm2
	memset(xi, 0, units*8*sizeof(xi[0]));
	
    for (size_t j = 0;  j < units; ++j) {//outcols = units
        for (size_t k = 0; k < in_width; ++k) {//innerdim = inwidth
			//W_i/f/c/o*input (2.5k times) or U_i/f/c/o*x_i_f_c_o (10k times) x4 overall for i f c o
            xi[j] += input[k] * Wi[k*units+j]; 
			xf[j] += input[k] * Wf[k*units+j]; 
			xc[j] += input[k] * Wc[k*units+j]; 
			xo[j] += input[k] * Wo[k*units+j]; 		
        }
        for (size_t k = 0; k < units; ++k) {//innerdim = units
			yi[j] += h_prev[k] * Ui[k*units+j]; 
			yf[j] += h_prev[k] * Uf[k*units+j];
			yc[j] += h_prev[k] * Uc[k*units+j];
			yo[j] += h_prev[k] * Uo[k*units+j];
		}
    }
	
	for (size_t j = 0;  j < units; ++j) {
		yo[j] = hard_sigmoid_func(yo[j] + xo[j]+ bo[j]);
		yc[j] = hard_sigmoid_func(yf[j] + xf[j]+ bf[j])*c_prev[j] + hard_sigmoid_func(yi[j] + xi[j]+ bi[j])*hard_sigmoid_func(yc[j] + xc[j]+ bc[j]);
		state[units+j] = yc[j];
		state[j] = yo[j]*hard_sigmoid_func(yc[j]);
	}   
}

void lstm(tensor* output, const tensor* input, value state[],
              const tensor* kernel, const tensor* recurrent_kernel,
              const tensor* bias, value fwork[],
              const int return_sequences) {
    const size_t in_height = input->shape[0];
    const size_t in_width = input->shape[1];
    const size_t units = recurrent_kernel->shape[1];
	//printf("lstm \n ");
	//printf("in_height: %zd\n ", in_height);//50 timesteps
	//printf("in_width: %zd\n ", in_width);//25, 100
	//printf("units: %zd\n ", units);//100, 50 
	//printf("return_sequences: %u\n ", return_sequences);//0, 1
	//state = 800 for lstm1, 400 for lstm2
	//fwork = 200 for lstm1, 100 for lstm2
    for (size_t i=0; i < in_height; ++i) {
        lstmcell(state, &input->array[i*in_width], kernel, recurrent_kernel,
            bias, fwork);
        if (return_sequences) {
			//printf("Returning sequence \n ");
            for (size_t j=0; j<units; ++j) {
				//printf("%zd ", j);
                output->array[i*units+j] = state[j];
            }
        }
    }
    if (!return_sequences) {
        for (size_t i=0; i < units; ++i) {
            output->array[i] = state[i];
        }
    }
}

//activationType * hard_sigmoid = hard_sigmoid_func;

void predictive_maintenance(tensor* lstm_1_input_input, tensor* dense_1_output) {


    value lstm_1_kernel_array[10000];
	value lstm_1_recurrent_kernel_array[40000];
	value lstm_1_bias_array[400];
    value lstm_2_kernel_array[20000];
    value lstm_2_recurrent_kernel_array[10000];
    value lstm_2_bias_array[200];
    value dense_1_kernel_array[50];

	
	readfromfile(&lstm_1_kernel_array[0], "lstm_1_kernel.txt", 10000);
	readfromfile(&lstm_1_recurrent_kernel_array[0], "lstm_1_recurrent_kernel.txt", 40000);
	readfromfile(&lstm_1_bias_array[0], "lstm_1_bias.txt", 400);
	readfromfile(&lstm_2_kernel_array[0], "lstm_2_kernel.txt", 20000);
	readfromfile(&lstm_2_recurrent_kernel_array[0], "lstm_2_recurrent_kernel.txt", 10000);
	readfromfile(&lstm_2_bias_array[0], "lstm_2_bias.txt", 200);
	readfromfile(&dense_1_kernel_array[0], "dense_1_kernel.txt", 50);
	
    value lstm_1_output_array[5000] = {0};
    tensor lstm_1_output = {&lstm_1_output_array[0],2,5000,{ 50,100}};
    value lstm_1_fwork[800] = {0};
    int lstm_1_return_sequences = 1;
    value lstm_1_state[200] = {0};

    tensor lstm_1_kernel = {&lstm_1_kernel_array[0],2,10000,{100,100}};
    tensor lstm_1_recurrent_kernel = {&lstm_1_recurrent_kernel_array[0],2,40000,{400,100}};
    tensor lstm_1_bias = {&lstm_1_bias_array[0],1,400,{400,  1}};
	
    value lstm_2_output_array[50] = {0};
    tensor lstm_2_output = {&lstm_2_output_array[0],1,50,{50, 1}};
    value lstm_2_fwork[400] = {0};
    int lstm_2_return_sequences = 0;
    value lstm_2_state[100] = {0};

    tensor lstm_2_kernel = {&lstm_2_kernel_array[0],2,20000,{400, 50}};
    tensor lstm_2_recurrent_kernel = {&lstm_2_recurrent_kernel_array[0],2,10000,{200, 50}};
    tensor lstm_2_bias = {&lstm_2_bias_array[0],1,200,{200,  1}};
	
    tensor dense_1_kernel = {&dense_1_kernel_array[0],2,50,{50, 1}};
    value dense_1_bias_array[1] = {
        -4.99580391e-02,
    };
    tensor dense_1_bias = {&dense_1_bias_array[0],1,1,{1,1}};
    value dense_1_fwork[100] = {0};

	//writetofile(&lstm_1_kernel, "lstm_1_kernel.txt");
	//writetofile(&lstm_1_recurrent_kernel, "lstm_1_recurrent_kernel.txt");
	//writetofile(&lstm_1_bias, "lstm_1_bias.txt");
	//writetofile(&lstm_2_kernel, "lstm_2_kernel.txt");
	//writetofile(&lstm_2_recurrent_kernel, "lstm_2_recurrent_kernel.txt");
	//writetofile(&lstm_2_bias, "lstm_2_bias.txt");
	//writetofile(&dense_1_kernel, "dense_1_kernel.txt");
	
    lstm(&lstm_1_output,lstm_1_input_input,lstm_1_state,&lstm_1_kernel,
             &lstm_1_recurrent_kernel,&lstm_1_bias,lstm_1_fwork,
             lstm_1_return_sequences);
    lstm(&lstm_2_output,&lstm_1_output,lstm_2_state,&lstm_2_kernel,
             &lstm_2_recurrent_kernel,&lstm_2_bias,lstm_2_fwork,
             lstm_2_return_sequences);
    dense(dense_1_output,&lstm_2_output,&dense_1_kernel,
              &dense_1_bias,dense_1_fwork);

}

value roundabs(tensor *tensor1, tensor *tensor2);
struct timeval GetTimeStamp();

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
		}
		
	}
	fclose(fp);
	tensor test_input[MTEST];
	tensor test_output[MTEST];
	tensor test_truth[MTEST];
	
	float test_output_array[MTEST] = {0,};
	
	for (i = 0; i < MTEST; i++) {
		test_input[i] = (tensor) {&test_input_array[i*MHEIGHT*MWIDTH],2,1250,{50,25}};
		test_output[i] = (tensor) {&test_output_array[i],1,1,{1,1}};
		test_truth[i] = (tensor) {&truth_array[i],1,1,{1,1}};
	}
	
	float errors[MTEST];
	size_t num_tests = 93;
	size_t num_outputs = 1;
	
	clock_t t0 = clock();
	for (int i =0; i < MTEST; i++)
		predictive_maintenance(&test_input[i],&test_output[i]);
	clock_t t1 = clock();
	printf("Test time: %e s \n",
           (double)(t1-t0)/(double)CLOCKS_PER_SEC/(double)10);
	for (int i =0; i < MTEST; i++) {
		printf("Test: %u ", i);
		errors[i] = roundabs(&test_truth[i],&test_output[i]);
		printf("Error: %f \n", errors[i]);
		
	}
	float maxerror = errors[0];
    for(i=1; i< MTEST; i++) {
        maxerror += errors[i];
	}
	printf("Number of tests failed: %f\n", maxerror);
	maxerror = maxerror / MTEST;
	printf("Max absolute error for 93 tests: %e \n", maxerror);
	
}

value roundabs(tensor *tensor1, tensor *tensor2) {

    value x = 0;
    value y = 0;

    for(size_t i=0; i<tensor1->numel; i++) {
		printf("Truth: %f Result: %f ", tensor1->array[i], tensor2->array[i]);
        y = fabs(round(tensor1->array[i])-round(tensor2->array[i]));
        if (y>x) {
            x=y;
        }
    }
    return x;
}



