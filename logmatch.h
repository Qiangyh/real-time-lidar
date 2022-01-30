#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void logMatchedFilterKernel(float *points, float * reflect, int *points_per_pix, float * background, const float * log_impulse, const float *d_integrated_impulse,
	const int *bins_counts, const int *pix_counts, const int *pix_counts_idx, const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const int subsampling,
	const int segments, const float max_bkg, const int many_irf, const float *d_gain);

__global__ void denseLogMatchedFilterKernel(float *points, float * reflect, int *points_per_pix, float *background, const float * log_impulse, const float *d_integrated_impulse,
	const int *data, const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const int downsample,
	const int segments, const float max_bkg, const int many_irf, const float *d_gain, const float SBR);

__global__ void denseLogMatchedFilterPeaks(float *points, float * reflect, int *points_per_pix, float *background, const float * log_impulse, const float *d_integrated_impulse,
	const int *data, const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const int downsample,
	const int segments, const float max_bkg, const int many_irf, const float *d_gain, const float SBR, const float min_dist);


__global__ void denseReInitLogMatchedFilterKernel(float *in_points, float * in_reflect, int *in_points_per_pix, const float *in_background, 
	const float * log_impulse, const float *d_integrated_impulse,	const int *data, const int impulse_len, const int T, const int Nrow,
	const int Ncol, const int upsampling, const int downsample,	const int segments, const float max_bkg, const int many_irf,
	const float *d_gain, const float min_dist);


// this function doesn't support per pixel IRFs
__global__ void logMatchedMSFilterKernel(float *points, float * reflect, int *points_per_pix, float *background, const float * log_impulse, const float *d_integrated_impulse,
	const int *bins_counts, const int *pix_counts, const int *pix_counts_idx, const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling,
	const int downsample, const int segments, const float max_bkg, const float *d_gain, const int * coded_aperture, const int L);


/*__global__ void logMatchedMSFilterKernelSMem(float *points, float * reflect, int *points_per_pix, float *background, const float * log_impulse, const float *d_integrated_impulse,
	const int *bins_counts, const int *pix_counts, const int *pix_counts_idx, const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling,
	const int downsample, const int max_points, const float max_bkg, const float *d_gain, const int * coded_aperture, const int L); */


__global__ void denseLogMatchedFilterOMP(float *points, float * reflect, int *points_per_pix, float *background, const float * in_impulse, const float *d_integrated_impulse,
	const int *data, const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const int downsample,
	const int segments, const float max_bkg, const int many_irf, const float *d_gain, const float *d_irf_norm, const float SBR);

__global__ void ReInitLogMatchedFilterKernel(float *in_points, float * in_reflect, int *in_points_per_pix, float *in_background, const float * log_impulse, const float *d_integrated_impulse,
	const int *bins_counts, const int *pix_counts, const int *pix_counts_idx, const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const int downsample,
	const int segments, const int many_irf, const float *d_gain, float min_dist);


__global__ void half_sample_mode(float *points, float * reflect, int *points_per_pix, float *background,
	const int *bins_counts, const int *pix_counts, const int *pix_counts_idx, const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling,
 const float max_bkg, const float *d_gain, const int L, int attack);

__global__ void circularMean(float *points, float * reflect, int *points_per_pix, const float * sk_impulse, const float * cm_corr,
	const float *data, const int m, const int T, const int Nrow, const int Ncol, const int upsampling,
	const int many_irf, const float *d_gain);


__global__ void sketchedMultiPeakInit(float *points, float * reflect, int *points_per_pix, const float * sk_impulse, const int segments,
	const float *data, const int m, const int T, const int Nrow, const int Ncol, const int upsampling, const int partitions, 
	const int many_irf, const float *d_gain, const float spectral_norm);


__global__ void sketchedMultiPeakOMPInit(float *points, float * reflect, int *points_per_pix, const float * sk_impulse, const int segments,
	const float *data, const int m, const int T, const int Nrow, const int Ncol, const int upsampling, const int partitions,
	const int many_irf, const float *d_gain, const float spectral_norm);