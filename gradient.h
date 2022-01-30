#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



// LIKELIHOOD FUNCTIONS 
__global__ void likelihood_kernel(float * likelihood, float *in_points, float * in_reflect, int *in_points_per_pix, float * in_background, float *reg_background,
	const int *bins_counts, const int *pix_counts, const int * pix_counts_idx, const float * impulse, const float *d_integrated_impulse,
	const int impulse_len,	const int T, const int Nrow, const int Ncol, const int upsampling, const int many_irf, const float *d_gain);

__global__ void likelihood_MS_kernel(float * likelihood, float *in_points, float * in_reflect, int *in_points_per_pix, float * in_background, float *reg_background,
	const int *bins_counts, const int *pix_counts, const int * pix_counts_idx, const float * impulse, const float *d_integrated_impulse,
	const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const int many_irf, const float *d_gain, const int *coded_aperture, const int L);

__global__ void dense_likelihood_kernel(float * likelihood, float *in_points, float * in_reflect, int *in_points_per_pix, float * in_background, float *reg_background,
	const int *data, const float * impulse, const float *d_integrated_impulse, const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling,
	const int many_irf, const float *d_gain);

__global__ void sketch_likelihood_kernel(float * likelihood, float *in_points, float * in_reflect, int *in_points_per_pix,
	const float *data, const float * sketched_irf, const int m, const int T, const int Nrow, const int Ncol, const int upsampling,
	const int many_irf, const float *d_gain);

// BACKGROUND FUNCTIONS
__global__ void prox_bkg_kernel(float *in_points, float * in_reflect, int *in_points_per_pix, float * in_background,
	const int *bins_counts, const int *pix_counts, const int * pix_counts_idx, const float * impulse, const float *d_integrated_impulse,
	const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const float step_size_bkg,
	const float max_bkg, const int many_irf, const float *d_gain);

__global__ void dense_prox_bkg_kernel(float *in_points, float * in_reflect, int *in_points_per_pix, float * in_background,
	const int *data, const float * impulse, const float *d_integrated_impulse, const int impulse_len, const int T,
	const int Nrow, const int Ncol, const int upsampling, const float step_size_bkg, const float max_bkg, const int many_irf, const float *d_gain);

__global__ void prox_MSbkg_kernel(float *in_points, float * in_reflect, int *in_points_per_pix, float * in_background,
	const int *bins_counts, const int *pix_counts, const int * pix_counts_idx, const float * impulse, const float *d_integrated_impulse,
	const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const float step_size_bkg,
	const float max_bkg, const int many_irf, const float *d_gain, const int * coded_aperture, const int L);

// OTHER USEFUL
void global_max_reduce(float *d_idata, float * d_odata, size_t data_size, int smemSize);
void global_reduce(float *d_idata, float * d_odata, size_t data_size, int smemSize);


// Point Cloud gradient functions 
__global__ void point_cloud_gradient_kernel(float *in_points, float * in_reflect, int *in_points_per_pix, float * in_background,
	const int *bins_counts, const int *pix_counts, const int * pix_counts_idx, const float * impulse, const float * der_impulse, const float *d_integrated_impulse,
	const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const float step_size_depth,
	const float step_size_reflec, const float max_refl, const int many_irf, const float *d_gain);

__global__ void point_cloud_ms_gradient_kernel(float *in_points, float * in_reflect, int *in_points_per_pix, float * in_background,
	const int *bins_counts, const int *pix_counts, const int * pix_counts_idx, const float * impulse, const float * der_impulse, const float *d_integrated_impulse,
	const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const float step_size_depth,
	const float step_size_reflec, const float max_refl, const int many_irf, const float *d_gain, const int *coded_aperture, const int L);

__global__ void simple_dense_point_cloud_gradient_kernel(float *in_points, float * in_reflect, int *in_points_per_pix, float * in_background,
	const int *data, const float * impulse, const float * der_impulse, const float *d_integrated_impulse, const int impulse_len, const int T,
	const int Nrow, const int Ncol, const int upsampling, const float step_size_depth, const float step_size_reflec, const float max_refl, const int many_irf, const float *d_gain);

__global__ void sketched_gradient_kernel (float * in_points, float * in_reflect, int * in_points_per_pix,
	const float * d_sketched, const float * d_sketched_irf, const int m, const int T, const int Nrow, const int Ncol, 
	const int upsampling, const float step_size_depth, const float step_size_reflec, const float max_refl, const int many_irf, const float *d_gain);