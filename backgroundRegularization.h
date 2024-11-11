#pragma once 

#include "cufft.h"

////////////////////////////////////////////////////////////////////////////////
// Kernel configuration
////////////////////////////////////////////////////////////////////////////////
#define KERNEL_RADIUS 8 // DONT CHANGE THIS (OTHERWISE IT WILL GIVE ERRORS)!!
#define KERNEL_W (2 * KERNEL_RADIUS + 1)

#define TILE_W 16		// active cell width
#define TILE_H 16		// active cell height
#define TILE_SIZE (TILE_W + KERNEL_RADIUS * 2) * (TILE_W + KERNEL_RADIUS * 2)

#define IMUL(a,b) __mul24(a,b)

__global__ void convolutionRowGPU(float *d_Result, float *d_Data, int dataW, int dataH);

__global__ void convolutionColGPU(float *d_Result, float *d_Data, int dataW, int dataH);
void setKernel(float *h_kernel);


__global__ void gradient_step_bkg(float *in_background, float *reg_background, float *out_background, const float mu, const float lambda, const int Nrow, const int Ncol);


__global__ void exp_kernel(float *src_background,float *dest_background,const int Nrow, const int Ncol);

__global__ void exp_MSkernel(float *source_background, float *dest_background, const int Nrow, const int Ncol, const int L);

__global__ void clamp_value(float *in_background, const float max_bkg, const int Nrow, const int Ncol);

__global__ void fft_kernel(cufftComplex *in_background, cufftReal *filter, const float sigma, const int Nrow, const int Ncol);

__global__ void extract_real_symmetric_fft(cufftComplex *in_filter, cufftReal *filter, const int Nrow, const int Ncol);

void run_IFFT(cufftHandle *plan, cufftComplex *in_data, cufftReal *out_data);

void run_FFT(cufftHandle *plan, cufftReal *in_data, cufftComplex *out_data);


void setCudaFFT(int Nrow, int Ncol, cufftHandle *fft, cufftHandle *ifft);
