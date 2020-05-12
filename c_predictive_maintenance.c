#include <math.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <stdlib.h>

#define MTEST 93
#define MHEIGHT 50
#define MWIDTH  25
#define MAX_NDIM 2

#define LSTM_1_UNITS 50
#define LSTM_2_UNITS 100

#define LSTM_WEIGHT_FACTOR 4

#define QUANT_BITS 8 //Set to 8,16,32

#define QUANTIZE 1 //Running quantization

#define SIGNED 1 //Signed integer or not

#define CONVERT_INPUT 1 //If you want to convert the input from float to value
#define SAVE_DATA 0 //If you want to save the data
#define F_RANGE 7.125f

float float_range;
float float_max;

bool data_loaded = false;
bool weights_saved = false;

#if QUANTIZE
	#if QUANT_BITS == 32
		#if SIGNED
			typedef int32_t value;
			typedef int64_t multiplier;
		#else
			typedef uint32_t value;
			typedef uint64_t multiplier;
		#endif
	#elif QUANT_BITS == 16
		#if SIGNED
			typedef int16_t value;
			typedef int32_t multiplier;
		#else
			typedef uint16_t value;
			typedef uint32_t multiplier;
		#endif
	#elif QUANT_BITS == 8
		#if SIGNED
			typedef int8_t value;
			typedef int16_t multiplier;
		#else
			typedef uint8_t value;
			typedef uint16_t multiplier;
		#endif
	// #elif QUANT_BITS == 4
		// #if SIGNED
			// typedef int4_t value;
			// typedef int8_t multiplier;
		// #else
			// typedef uint4_t value;
			// typedef uint8_t multiplier;
		// #endif
	#else
		typedef float value;
		typedef float multiplier;
	#endif
#else
	typedef float value;
	typedef float multiplier;
#endif


value invscale;
value zeropoint;

//const char value_format = "%e";
//%d for signed integer
//%e for float
//%SCNd8, SCNd16, SCNd32 and SCNd64 for bit-defined integers

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

 void writetofile(value* value_array, char* filename, size_t length){
	printf("Writing values to file  %s\n", filename);
	FILE *filePtr;
	filePtr = fopen(filename, "w");
	for (size_t i = 0; i < length; i++) {
#if QUANTIZE
	#if QUANT_BITS == 32
		#if SIGNED
			fprintf(filePtr, "%" PRId32, value_array[i]);
		#else
			fprintf(filePtr, "%" PRIo32, value_array[i]);
		#endif
	#elif QUANT_BITS == 16
		#if SIGNED
			fprintf(filePtr, "%" PRId16, value_array[i]);
		#else
			fprintf(filePtr, "%" PRIo16, value_array[i]);
		#endif
	#elif QUANT_BITS == 8
		#if SIGNED
			fprintf(filePtr, "%" PRId8, value_array[i]);
		#else
			fprintf(filePtr, "%" PRIo8, value_array[i]);
		#endif
	#elif QUANT_BITS == 4
		printf("Invalid Fixed Bit Size");
		exit(1);
	#else
		printf("Invalid Fixed Bit Size");
		exit(1);	
	#endif
#else
		fprintf(filePtr, "%.18e", value_array[i]);
#endif
		if (i < length-1) {
			fprintf(filePtr, "\n");
		}
	}
	fclose(filePtr);
 }

void readvaluefromfile(value* value_array, char* filename, size_t length)
{
	printf("Reading values from file %s\n", filename);
	FILE * fp;
	fp = fopen(filename, "r");
	if (fp == NULL) {
		printf("File not found: %s", filename);
		exit(1);
	}
	
	for (size_t i = 0; i < length; i++) {
		bool status;
#if QUANTIZE
	#if QUANT_BITS == 32
		#if SIGNED
			status = fscanf(fp, "%" SCNd32, &value_array[i])!= 1;
		#else
			status = fscanf(fp, "%" SCNo32, &value_array[i])!= 1;
		#endif
	#elif QUANT_BITS == 16
		#if SIGNED
			status = fscanf(fp, "%" SCNd16, &value_array[i])!= 1; 
		#else
			status = fscanf(fp, "%" SCNo16, &value_array[i])!= 1;
		#endif
	#elif QUANT_BITS == 8
		#if SIGNED
			status = fscanf(fp, "%"SCNd8, &value_array[i])!= 1; 
		#else
			status = fscanf(fp, "%" SCNo8, &value_array[i])!= 1;
		#endif
	#elif QUANT_BITS == 4
		printf("Invalid Fixed Bit Size");
		exit(1);
	#else
		printf("Invalid Fixed Bit Size");
		exit(1);	
	#endif
#endif
		if (status) {
			printf("Access error in %s array index %zu : input not valid.", filename, i);
			exit(1);
		} 
	}
	fclose(fp);
}

void readfloatfromfile(float* float_array, char* filename, size_t length)
{
	printf("Reading floats from file %s \n", filename);
	FILE * fp;
	fp = fopen(filename, "r");
	if (fp == NULL) {
		printf("File not found: %s", filename);
		exit(1);
	}
	for (size_t i = 0; i < length; i++) {
		if (fscanf(fp, "%e", &float_array[i]) != 1) {
			printf("Access error in %s array index %zu : input not valid.", filename, i);
			exit(1);
		}
		// else 
		// {
			// printf("%e ", float_array[i]);
		// }
	}
	fclose(fp);
}

#if QUANTIZE
float get_qrange()
{	
	float qrange;
	if (QUANT_BITS == 32) qrange = UINT32_MAX;
	else if (QUANT_BITS == 16) qrange = UINT16_MAX;
	else if (QUANT_BITS == 8) qrange = UINT8_MAX;
	else if (QUANT_BITS == 4) qrange = 15;
	else 
	{
		printf("Invalid Fixed Bit Size");
		exit(1);
	}
	printf("Float range: %f\n", float_range);
	printf("Quantized range: %f\n", qrange);
	return qrange;
}

value get_invscale()
{
	float q_range = (float) get_qrange();
	value qinvscale = (value) (q_range/float_range);
	printf("Quantized Inverse Scale: %d\n", qinvscale);
	return qinvscale;
}

value get_zeropoint(value qinvscale)
{
	float qmax = (float) float_max;
	if (QUANT_BITS == 32) qmax = (SIGNED) ? INT32_MAX : UINT32_MAX;
	else if (QUANT_BITS == 16) qmax = (SIGNED) ? INT16_MAX : UINT16_MAX;
	else if (QUANT_BITS == 8) qmax = (SIGNED) ? INT8_MAX : UINT8_MAX;
	else if (QUANT_BITS == 4) qmax = (SIGNED) ? 7 : 15;
	else 
	{
		printf("Invalid Fixed Bit Size");
		exit(1);
	}
	printf("Float max: %f\n", float_max);
	printf("Quantized max: %f\n", qmax);
	float subtrahend = (float)(float_max*qinvscale);
	printf("Quantized Zeropoint Subtrahend: %f\n", subtrahend);
	float qzeropoint =  qmax - subtrahend;
	printf("Quantized Zeropoint: %f\n", qzeropoint);
	value vzeropoint = (value) qzeropoint;
	printf("Value Zeropoint: %"PRIo32"\n", vzeropoint);
	return vzeropoint;
}
int test = 0;
value quantize(float val)
{
	//if (test < 5) printf("qq sigmoidf %f \n", val);
	float scaled = invscale * val;
	//if (test < 5) printf("qqq sigmoidf %f \n", scaled);
	value quantized = (value) (scaled + zeropoint);
	
	return quantized;
}

float dequantize(value val)
{
	float adjusted = (float)(val - zeropoint);
	float dequantized = (float) adjusted / invscale;
	return dequantized;
}


void quantize_arr(float* ff_array, value* val_array, size_t len)
{
	printf("Quantizing Array...\n");
	for (size_t i = 0; i < len; i++)
	{	
		//if (len == 10000 && i == 0 && test < 5) printf("q sigmoidf %f \n", (ff_array[i]));
		val_array[i] = quantize(ff_array[i]);
		//if (len == 10000 && i == 0 && test < 5) printf("sigmoid %"PRId8" \n", (val_array[i]));
		//printf("%d", val_array[i]);
	}
}

value quantized_mult(value x, value y)
{
	multiplier xterm = (multiplier) (x - zeropoint);
	multiplier yterm = (multiplier) (y - zeropoint);
	multiplier scaled = xterm * yterm / invscale;
	value product = (value) scaled + zeropoint;
	return product;
}

value quantized_add(value x, value y)
{
	//multxy = 2*max(xscale,yscale)
	//multx = 2
	multiplier xterm = (multiplier) (x - zeropoint);
	xterm = (xterm) / 2;
	multiplier yterm = (multiplier) (y - zeropoint); 
	yterm = (yterm) / 2;
	multiplier scaled = (xterm + yterm) * 2;
	value sum = (value) scaled + zeropoint;
	return sum;
}

#endif


value relu_func(value x) {
	value zero = quantize(0.0f);
	if (x > zero) return x;
	else return zero;
}


value hard_sigmoid_func(value x) {
	value sigmoid;
#if QUANTIZE
	value upperlimit = quantize(2.5f);
	value lowerlimit = quantize(-2.5f);
	if (x <= lowerlimit) sigmoid = quantize(0.0f);
	else if (x >= upperlimit) sigmoid = quantize(1.0f);
	else
	{
		value offset = quantize(0.5f);
		value coeff = quantize(0.2f);
		sigmoid = quantized_add(offset, quantized_mult(coeff, x));
	}
		//if (test < 1) printf("sigmoid %f \n", dequantize(sigmoid));
#else 
	if (x <= -2.5f) {
		sigmoid = 0.0f;
	}
	else if (x >= 2.5f) {
		sigmoid = 1.0f;
	}
	else {
		sigmoid = 0.2f*x + 0.5f;
	}
	if (test < 1) printf("sigmoid %f \n", (sigmoid));
#endif
	return sigmoid;
}

value sigmoid_func(value x) {
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
#if QUANTIZE
			output->array[j] = quantized_add(output->array[j], quantized_mult(input->array[k], kernel->array[k*outcols+j]));
		}
		output->array[j] = hard_sigmoid_func(quantized_add(output->array[j], bias->array[j]));
#else
			output->array[j] += input->array[k] * kernel->array[k*outcols+j]; //W_i/f/c/o*input (2.5k times) or U_i/f/c/o*x_i_f_c_o (10k times) x4 overall for i f c o
		}
		output->array[j] = hard_sigmoid_func(output->array[j] + bias->array[j]);//+bias (50 times));
#endif
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
    const value * const Wf = &kernel->array[1*in_width*units];
    const value * const Wc = &kernel->array[2*in_width*units];
    const value * const Wo = &kernel->array[3*in_width*units];
    const value * const Ui = &recurrent_kernel->array[0];
    const value * const Uf = &recurrent_kernel->array[1*units*units];
    const value * const Uc = &recurrent_kernel->array[2*units*units];
    const value * const Uo = &recurrent_kernel->array[3*units*units];
    const value * const bi = &bias->array[0];
    const value * const bf = &bias->array[1*units];
    const value * const bc = &bias->array[2*units];
	const value * const bo = &bias->array[3*units];   
    value *xi = &fwork[0];                                                                              
    value *xf = &fwork[1*units];
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
#if QUANTIZE
		for (size_t j = 0;  j < units; ++j) {//outcols = units
			for (size_t k = 0; k < in_width; ++k) {//innerdim = inwidth
			//W_i/f/c/o*input (2.5k times) or U_i/f/c/o*x_i_f_c_o (10k times) x4 overall for i f c o
				if (test < 5) printf("sigmoid %"PRId32" \n", (input[k]));
				if (test < 5) printf("sigmoidf %f \n", dequantize(input[k]));
				if (test < 5) printf("sigmoid %"PRId32" \n", ( Wi[k*units+j]));
				if (test < 5) printf("sigmoidf %f \n", dequantize(Wi[k*units+j]));
				if (test < 5) printf("sigmoid %"PRId32" \n", (xi[j]));
				if (test < 5) printf("sigmoidf %f \n", dequantize(xi[j]));
				if (test < 5) printf("sigmoid %"PRId32" \n", quantized_mult(input[k], Wi[k*units+j]));
				if (test < 5) printf("sigmoidf %f \n", dequantize(quantized_mult(input[k], Wi[k*units+j])));
				
				xi[j] = quantized_add(xi[j], quantized_mult(input[k], Wi[k*units+j]));
				xf[j] = quantized_add(xf[j], quantized_mult(input[k], Wf[k*units+j]));
				xc[j] = quantized_add(xc[j], quantized_mult(input[k], Wc[k*units+j]));
				xo[j] = quantized_add(xo[j], quantized_mult(input[k], Wo[k*units+j]));
				
				if (test < 5) printf("sigmoid %"PRId32" \n", (xi[j]));
				if (test < 5) printf("sigmoidf %f \n\n", dequantize(xi[j]));
				test +=1;
			}
			for (size_t k = 0; k < units; ++k) {//innerdim = units
				yi[j] = quantized_add(yi[j], quantized_mult(h_prev[k], Ui[k*units+j]));
				yf[j] = quantized_add(yf[j], quantized_mult(h_prev[k], Uf[k*units+j]));
				yc[j] = quantized_add(yc[j], quantized_mult(h_prev[k], Uc[k*units+j]));
				yo[j] = quantized_add(yo[j], quantized_mult(h_prev[k], Uo[k*units+j]));
			}
		}
		for (size_t j = 0;  j < units; ++j) {
			value qo = quantized_add(quantized_add(yo[j], xo[j]), bo[j]);
			value qf = quantized_add(quantized_add(yf[j], xf[j]), bf[j]);
			value qi = quantized_add(quantized_add(yi[j], xi[j]), bi[j]);
			value qc = quantized_add(quantized_add(yc[j], xc[j]), bc[j]);
			qo = hard_sigmoid_func(qo);
			yo[j] = qo;
			qf = hard_sigmoid_func(qf);
			qi = hard_sigmoid_func(qi);
			qc = hard_sigmoid_func(qc);
			yc[j] = quantized_add(quantized_mult(qf, c_prev[j]), quantized_mult(qi, qc));
			state[units+j] = yc[j];
			state[j] = quantized_mult(qo,hard_sigmoid_func(yc[j]));
		}
		for (size_t i = 0; i < 100 && test < 1; i++)
		{
			float weight = (dequantize(state[i]));
			printf("%zu : %f \n", i, weight);
		}
		test += 1;
#else 
		for (size_t j = 0;  j < units; ++j) {//outcols = units
			for (size_t k = 0; k < in_width; ++k) {//innerdim = inwidth
			//W_i/f/c/o*input (2.5k times) or U_i/f/c/o*x_i_f_c_o (10k times) x4 overall for i f c o
				if (test < 5) printf("sigmoid %f \n", (input[k]));
				if (test < 5) printf("sigmoid %f \n", ( Wi[k*units+j]));
				if (test < 5) printf("sigmoid %f \n", (xi[j]));
				if (test < 5) printf("sigmoid %f \n", ( input[k] * Wi[k*units+j]));
				xi[j] += input[k] * Wi[k*units+j]; 
				xf[j] += input[k] * Wf[k*units+j]; 
				xc[j] += input[k] * Wc[k*units+j]; 
				xo[j] += input[k] * Wo[k*units+j]; 
				
				if (test < 5) printf("sigmoid %f \n\n", (xi[j]));
				test +=1;
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
		for (size_t i = 0; i < 100 && test < 1; i++)
		{
			float weight = (state[i]);
			printf("%zu : %f \n", i, weight);
		}
		test += 1;
#endif
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

value lstm_1_kernel_array[10000];
value lstm_1_recurrent_kernel_array[40000];
value lstm_1_bias_array[400];
value lstm_2_kernel_array[20000];
value lstm_2_recurrent_kernel_array[10000];
value lstm_2_bias_array[200];
value dense_1_kernel_array[50];
value dense_1_bias_array[1] = {
    -4.99580391e-02,
};

void predictive_maintenance(tensor* lstm_1_input, tensor* dense_1_output) {
	if (!data_loaded) {
#if CONVERT_INPUT
		printf("Converting Float Weights...\n");
		
		float ff_lstm_1_kernel_array[10000];
		float ff_lstm_1_recurrent_kernel_array[40000];
		float ff_lstm_1_bias_array[400];
		float ff_lstm_2_kernel_array[20000];
		float ff_lstm_2_recurrent_kernel_array[10000];
		float ff_lstm_2_bias_array[200];
		float ff_dense_1_kernel_array[50];
		float ff_dense_1_bias_array[1] = {
			-4.99580391e-02,
		};
		
		readfloatfromfile(&ff_lstm_1_kernel_array[0], "lstm_1_kernel.txt", 10000);
		quantize_arr(&ff_lstm_1_kernel_array[0], &lstm_1_kernel_array[0], 10000);
		
		readfloatfromfile(&ff_lstm_1_recurrent_kernel_array[0], "lstm_1_recurrent_kernel.txt", 40000);
		quantize_arr(&ff_lstm_1_recurrent_kernel_array[0], &lstm_1_recurrent_kernel_array[0], 40000);
		
		readfloatfromfile(&ff_lstm_1_bias_array[0], "lstm_1_bias.txt", 400);
		quantize_arr(&ff_lstm_1_bias_array[0], &lstm_1_bias_array[0], 400);
		
		readfloatfromfile(&ff_lstm_2_kernel_array[0], "lstm_2_kernel.txt", 20000);
		quantize_arr(&ff_lstm_2_kernel_array[0], &lstm_2_kernel_array[0], 20000);
		
		readfloatfromfile(&ff_lstm_2_recurrent_kernel_array[0], "lstm_2_recurrent_kernel.txt", 10000);
		quantize_arr(&ff_lstm_2_recurrent_kernel_array[0], &lstm_2_recurrent_kernel_array[0], 10000);
		
		readfloatfromfile(&ff_lstm_2_bias_array[0], "lstm_2_bias.txt", 200);
		quantize_arr(&ff_lstm_2_bias_array[0], &lstm_2_bias_array[0], 200);
		
		readfloatfromfile(&ff_dense_1_kernel_array[0], "dense_1_kernel.txt", 50);
		quantize_arr(&ff_dense_1_kernel_array[0], &dense_1_kernel_array[0], 50);
		
		quantize_arr(&ff_dense_1_bias_array[0], &dense_1_bias_array[0], 1);
#else
		printf("Reading Weights...\n");
		readfloatfromfile(&lstm_1_kernel_array[0], "lstm_1_kernel.txt", 10000);
		readfloatfromfile(&lstm_1_recurrent_kernel_array[0], "lstm_1_recurrent_kernel.txt", 40000);
		readfloatfromfile(&lstm_1_bias_array[0], "lstm_1_bias.txt", 400);
		readfloatfromfile(&lstm_2_kernel_array[0], "lstm_2_kernel.txt", 20000);
		readfloatfromfile(&lstm_2_recurrent_kernel_array[0], "lstm_2_recurrent_kernel.txt", 10000);
		readfloatfromfile(&lstm_2_bias_array[0], "lstm_2_bias.txt", 200);
		readfloatfromfile(&dense_1_kernel_array[0], "dense_1_kernel.txt", 50);
#endif
		data_loaded = true;
	}
	
    value lstm_1_output_array[LSTM_1_UNITS*100] = {0};
    tensor lstm_1_output = {&lstm_1_output_array[0],2,LSTM_1_UNITS*100,{LSTM_1_UNITS,100}};
    value lstm_1_fwork[800] = {0};
    int lstm_1_return_sequences = 1;
    value lstm_1_state[200] = {0};

    tensor lstm_1_kernel = {&lstm_1_kernel_array[0],2,LSTM_2_UNITS*100,{LSTM_2_UNITS,100}};
    tensor lstm_1_recurrent_kernel = {&lstm_1_recurrent_kernel_array[0],2,400*100,{400,100}};
    tensor lstm_1_bias = {&lstm_1_bias_array[0],1,400,{400,  1}};
	
    value lstm_2_output_array[50] = {0};
    tensor lstm_2_output = {&lstm_2_output_array[0],1,50,{50, 1}};
    value lstm_2_fwork[400] = {0};
    int lstm_2_return_sequences = 0;
    value lstm_2_state[100] = {0};

    tensor lstm_2_kernel = {&lstm_2_kernel_array[0],2,200*50,{400, 50}};
    tensor lstm_2_recurrent_kernel = {&lstm_2_recurrent_kernel_array[0],2,200*50,{200, 50}};
    tensor lstm_2_bias = {&lstm_2_bias_array[0],1,200,{200,  1}};
	
    tensor dense_1_kernel = {&dense_1_kernel_array[0],2,50,{50, 1}};
    tensor dense_1_bias = {&dense_1_bias_array[0],1,1,{1,1}};
    value dense_1_fwork[100] = {0};

#if SAVE_DATA
	if (!weights_saved) {
		
	#if QUANTIZE
		writetofile(&lstm_1_kernel_array[0], "q_lstm_1_kernel.txt", 10000);
		writetofile(&lstm_1_recurrent_kernel_array[0], "q_lstm_1_recurrent_kernel.txt", 40000);
		writetofile(&lstm_1_bias_array[0], "q_lstm_1_bias.txt", 400);
		writetofile(&lstm_2_kernel_array[0], "q_lstm_2_kernel.txt", 20000);
		writetofile(&lstm_2_recurrent_kernel_array[0], "q_lstm_2_recurrent_kernel.txt", 10000);
		writetofile(&lstm_2_bias_array[0], "q_lstm_2_bias.txt", 200);
		writetofile(&dense_1_kernel_array[0], "q_dense_1_kernel.txt", 50);
		writetofile(&dense_1_bias_array[0], "q_dense_1_bias.txt", 1);
	#else
		writetofile(&lstm_1_kernel_array[0], "ff_lstm_1_kernel.txt", 10000);
		writetofile(&lstm_1_recurrent_kernel_array[0], "ff_lstm_1_recurrent_kernel.txt", 40000);
		writetofile(&lstm_1_bias_array[0], "ff_lstm_1_bias.txt", 400);
		writetofile(&lstm_2_kernel_array[0], "ff_lstm_2_kernel.txt", 20000);
		writetofile(&lstm_2_recurrent_kernel_array[0], "ff_lstm_2_recurrent_kernel.txt", 10000);
		writetofile(&lstm_2_bias_array[0], "ff_lstm_2_bias.txt", 200);
		writetofile(&dense_1_kernel_array[0], "ff_dense_1_kernel.txt", 50);
		writetofile(&dense_1_bias_array[0], "ff_dense_1_bias.txt", 1);
	#endif
		weights_saved = true;
	}
#endif
	
    lstm(&lstm_1_output,lstm_1_input,lstm_1_state,&lstm_1_kernel,
             &lstm_1_recurrent_kernel,&lstm_1_bias,lstm_1_fwork,
             lstm_1_return_sequences);
    lstm(&lstm_2_output,&lstm_1_output,lstm_2_state,&lstm_2_kernel,
             &lstm_2_recurrent_kernel,&lstm_2_bias,lstm_2_fwork,
             lstm_2_return_sequences);
    dense(dense_1_output,&lstm_2_output,&dense_1_kernel,
              &dense_1_bias,dense_1_fwork);

}

struct timeval GetTimeStamp();

int main() {
	
	float_range = F_RANGE;
	float_max = float_range/2.0f;
	
	printf("Starting C-Predictive-Maintenance...\n");
	value test_input_array[MTEST*MHEIGHT*MWIDTH];
	float test_truth_array[MTEST];
	value test_output_array[MTEST] = {0,};
	
#if QUANTIZE
	invscale = get_invscale();
	zeropoint = get_zeropoint(invscale);
	
	printf("Scale Inverse: %"PRId32"\n", invscale);
	printf("Zero Point: %"PRId32"\n", zeropoint);
	printf("Sigmoid Upper Limit: %"PRId32"\n", quantize(2.5f));
	printf("Sigmoid Lower Limit: %"PRId32"\n", quantize(-2.5f));
#endif
	
	
	printf("Loading Input...\n");

#if CONVERT_INPUT
	printf("Converting Float Input...\n");
	float ff_input_array[MTEST*MHEIGHT*MWIDTH];
	float ff_truth_array[MTEST];
	
	readfloatfromfile(&ff_input_array[0], "seq_array_test_last.txt", MTEST*MHEIGHT*MWIDTH);
	quantize_arr(&ff_input_array[0], &test_input_array[0], MTEST*MHEIGHT*MWIDTH);
	
	readfloatfromfile(&test_truth_array[0], "label_array_test_last.txt", MTEST);
#else
	printf("Reading Input...\n");
	readfloatfromfile(&test_input_array[0], "seq_array_test_last.txt", MTEST*MHEIGHT*MWIDTH);
	readfloatfromfile(&test_truth_array[0], "label_array_test_last.txt", MTEST);
#endif
	
	tensor test_input[MTEST];
	tensor test_output[MTEST];
	
	for (size_t i = 0; i < MTEST; i++) {
		test_input[i] = (tensor) {&test_input_array[i*MHEIGHT*MWIDTH],2,25*50,{50,25}};
		test_output[i] = (tensor) {&test_output_array[i],1,1,{1,1}};
	}
	
#if SAVE_DATA
	printf("Quantizing Float Input...\n");
	#if QUANTIZE
		writetofile(&test_input_array[0], "q_seq_array_test_last.txt", MTEST*MHEIGHT*MWIDTH);
		writetofile(&test_truth_array[0], "q_label_array_test_last.txt", MTEST);
	#else
		writetofile(&test_input_array[0], "ff_seq_array_test_last.txt", MTEST*MHEIGHT*MWIDTH);
		writetofile(&test_truth_array[0], "ff_label_array_test_last.txt", MTEST);
	#endif
#endif
	
	float errors[MTEST];
	size_t num_tests = 93;
	size_t num_outputs = 1;
	
	clock_t t0 = clock();
	for (size_t i=0; i < MTEST; i++) {
		printf("Running Test No. %zu...\n", i);
		predictive_maintenance(&test_input[i],&test_output[i]);
	}
	clock_t t1 = clock();
	printf("Test time: %e s \n",
           (double)(t1-t0)/(double)CLOCKS_PER_SEC/(double)10);
	for (size_t i =0; i < MTEST; i++) {
		printf("Test: %zu ", i);
		float result;
		value output = test_output[i].array[0];
		// printf("Output: %f ", output);
#if QUANTIZE
		result = dequantize(output);
#else
		result = test_output[i].array[0];
#endif
		printf("Result: %f ", result);
		printf("Truth: %f ", test_truth_array[i]); 
		errors[i] = round(fabs(round(result) - round(test_truth_array[i])));
		printf("Error: %f \n", errors[i]);
	}
	float maxerror = errors[0];
    for(size_t i=1; i< MTEST; i++) {
        maxerror += errors[i];
	}
	printf("Number of tests failed: %f\n", maxerror);
	maxerror = maxerror / MTEST;
	printf("Error rate for 93 tests: %.3f%% \n", maxerror);
}





