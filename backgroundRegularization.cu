/*
* Original source from nvidia cuda SDK 2.0
* Modified by S. James Lee (sjames@evl.uic.edi)
* 2008.12.05
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "backgroundRegularization.h"
#include "Point.h"
#include "GPU_macros.h"
#include <stdio.h>
#include <math.h>


__constant__ float d_Kernel[KERNEL_W];



void setCudaFFT(int Nrow, int Ncol, cufftHandle *fft, cufftHandle *ifft) {

	// cufftPlan2d(fft, Ncol, Nrow, CUFFT_R2C);
	// cufftPlan2d(ifft, Ncol, Nrow, CUFFT_C2R);
	cufftResult result;

    // Create FFT plan (Real to Complex)
    result = cufftPlan2d(fft, Ncol, Nrow, CUFFT_R2C);
    if (result != CUFFT_SUCCESS) {
        printf("CUFFT Plan Creation Failed for FFT (R2C) with error code %d\n", result);
        return;
    }

    // Create Inverse FFT plan (Complex to Real)
    result = cufftPlan2d(ifft, Ncol, Nrow, CUFFT_C2R);
    if (result != CUFFT_SUCCESS) {
        printf("CUFFT Plan Creation Failed for IFFT (C2R) with error code %d\n", result);
        return;
    }

    printf("CUFFT Plans created successfully: R2C and C2R\n");

}



void run_FFT(cufftHandle  * plan, cufftReal *in_data, cufftComplex *out_data) {
	cufftResult flag = 	cufftExecR2C(*plan, in_data, out_data);
	// if (CUFFT_SUCCESS != flag) 
	// 	printf("2D: cufftExecR2C fails\n");
	if (flag != CUFFT_SUCCESS) {
        printf("2D: cufftExecR2C fails with error code %d\n", flag);
        switch (flag) {
            case CUFFT_INVALID_PLAN:
                printf("Error: Invalid FFT plan\n");
                break;
            case CUFFT_ALLOC_FAILED:
                printf("Error: Memory allocation failed\n");
                break;
            case CUFFT_INVALID_TYPE:
                printf("Error: Invalid input/output data type\n");
                break;
            case CUFFT_EXEC_FAILED:
                printf("Error: Execution failed\n");
                break;
            default:
                printf("Error: Unknown error\n");
                break;
        }
    }
}


void run_IFFT(cufftHandle * plan, cufftComplex *in_data, cufftReal *out_data) {
	cufftResult flag = cufftExecC2R(*plan, in_data, out_data);
	if (CUFFT_SUCCESS != flag) {
		printf("2D: cufftExecR2C fails\n");
	}
}

void setKernel(float *h_kernel) {
	gpuErrchk(cudaMemcpyToSymbol(d_Kernel, h_kernel, KERNEL_W * sizeof(float)));
}


__global__ void exp_kernel (float *source_background, float *dest_background, const int Nrow,const int Ncol) {


	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;
	
	dest_background[linear_idx] = exp(source_background[linear_idx]);
}

__global__ void exp_MSkernel(float *source_background, float *dest_background, const int Nrow, const int Ncol, const int L) {


	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	for (int l=0; l<L; l++)
		dest_background[linear_idx + l*Nrow*Ncol] = exp(source_background[linear_idx + l * Nrow*Ncol]);
}

__global__ void clamp_value(float *in_background, const float max_bkg, const int Nrow, const int Ncol) {


	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	if (in_background[linear_idx] > max_bkg)
		in_background[linear_idx] = max_bkg;


	if (in_background[linear_idx] < 0.01)
		in_background[linear_idx] = 0.01;

}

__global__ void fft_kernel(cufftComplex *in_background, cufftReal *filter, const float sigma, const int Nrow, const int Ncol) {


	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);
	
	if (x >= Nrow || y >= ((Ncol / 2) + 1)) //finish out-of-scope threads
		return;

	float weight = (Nrow*Ncol * (1. + sigma * filter[linear_idx]));
	in_background[linear_idx].x /= weight;

	in_background[linear_idx].y /= weight;

	//printf("weight:%f - bkg: (%f,%f) \n", weight, in_background[idx].x, in_background[idx].y);

}


__global__ void extract_real_symmetric_fft(cufftComplex *in_filter, cufftReal *filter, const int Nrow, const int Ncol) {


	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	//printf("(%d,%d) value: %f \n", x, y, in_filter[idx].x);
	filter[linear_idx] = in_filter[linear_idx].x;

}


// gradient step bkg

__global__ void gradient_step_bkg(float *in_background, float *reg_background, float *out_background, const float mu, const float lambda, const int Nrow, const int Ncol) {


	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	float step_size = 1. / (lambda*mu + 1.);

	out_background[linear_idx] -=  step_size*( (out_background[linear_idx] -in_background[linear_idx]) + mu*lambda *(out_background[linear_idx]-reg_background[linear_idx]) );

}

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(
	float *d_Result,
	float *d_Data,
	int dataW,
	int dataH
)
{
	// Data cache: threadIdx.x , threadIdx.y
	__shared__ float data[TILE_H * (TILE_W + KERNEL_RADIUS * 2)];

	// global mem address of this thread
	// const int gLoc = threadIdx.x +
	// 	IMUL(blockIdx.x, blockDim.x) +
	// 	IMUL(threadIdx.y, dataW) +
	// 	IMUL(blockIdx.y, blockDim.y) * dataW;
	const int gLoc = threadIdx.x +
		(blockIdx.x * blockDim.x) +
		(threadIdx.y * dataW) +
		(blockIdx.y * blockDim.y) * dataW;


	// load cache (32x16 shared memory, 16x16 threads blocks)
	// each threads loads two values from global memory into shared mem
	// if in image area, get value in global mem, else 0
	int x;		// image based coordinate

				// original image based coordinate
	// const int x0 = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
	const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
	const int shift = threadIdx.y * (TILE_W + KERNEL_RADIUS * 2);
	// const int y = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	// finish out-of-scope threads
	if (y >= dataH)
		return;

	// case1: left
	x = x0 - KERNEL_RADIUS;
	if (x < 0)
		data[threadIdx.x + shift] = d_Data[gLoc - KERNEL_RADIUS + dataW];
	else
		data[threadIdx.x + shift] = d_Data[gLoc - KERNEL_RADIUS];

	// case2: right
	x = x0 + KERNEL_RADIUS;
	if (x >= dataW)
		data[threadIdx.x + blockDim.x + shift] = d_Data[gLoc + KERNEL_RADIUS - dataW];
	else
		data[threadIdx.x + blockDim.x + shift] = d_Data[gLoc + KERNEL_RADIUS];

	__syncthreads();

	if (x0>=dataW)
		return;

	// convolution
	float sum = 0;
	x = KERNEL_RADIUS + threadIdx.x;
	for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
		sum += data[x + i + shift] * d_Kernel[KERNEL_RADIUS + i];

	d_Result[gLoc] = sum;
}


////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColGPU(
	float *d_Result,
	float *d_Data,
	int dataW,
	int dataH
)
{
	// Data cache: threadIdx.x , threadIdx.y
	__shared__ float data[TILE_W * (TILE_H + KERNEL_RADIUS * 2)];

	// global mem address of this thread
	const int gLoc = threadIdx.x +
		blockIdx.x * blockDim.x +
		threadIdx.y * dataW +
		blockIdx.y * blockDim.y * dataW;

	// load cache (32x16 shared memory, 16x16 threads blocks)
	// each threads loads two values from global memory into shared mem
	// if in image area, get value in global mem, else 0
	int y;		// image based coordinate

				// original image based coordinate
	const int y0 = threadIdx.y + blockIdx.y * blockDim.y;
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int shift = threadIdx.y * (TILE_W);

	// finish out-of-scope threads
	if (x>= dataW)
		return;
	
	// case1: upper
	y = y0 - KERNEL_RADIUS;
	if (y < 0)
		data[threadIdx.x + shift] = d_Data[gLoc - dataW * KERNEL_RADIUS-dataH];
	else
		data[threadIdx.x + shift] = d_Data[gLoc - dataW * KERNEL_RADIUS];

	// case2: lower
	y = y0 + KERNEL_RADIUS;
	const int shift1 = shift + blockDim.y * TILE_W;
	if (y >= dataH )
		data[threadIdx.x + shift1] = d_Data[gLoc + dataW * KERNEL_RADIUS-dataH];
	else
		data[threadIdx.x + shift1] = d_Data[gLoc + dataW * KERNEL_RADIUS];
	
	__syncthreads();

	if (y0 >= dataH)
		return;

	// convolution
	float sum = 0;
	for (int i = 0; i <= KERNEL_RADIUS * 2; i++)
		sum += data[threadIdx.x + (threadIdx.y + i) * TILE_W] * d_Kernel[i];

	d_Result[gLoc] = sum;

}



