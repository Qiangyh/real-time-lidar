#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



// APSS functions
__global__ void APSS_with_normals_resample(float *in_points, float * in_normals, float * in_reflect, int *in_points_per_pix,
	float *out_points, float *out_normals, float * out_reflect, int *out_points_per_pix,
	const int T, const int Nrow, const int Ncol, const int pix_h, const float splat_proportion, const float scale_ratio, const int impulse_len, const int L);

__global__ void APSS_with_normals_resample_bilateral(float *in_points, float * in_normals, float * in_reflect, int *in_points_per_pix,
	float *out_points, float *out_normals, float * out_reflect, int *out_points_per_pix,
	const int T, const int Nrow, const int Ncol, const int pix_h, const float splat_proportion, const float scale_ratio, const int impulse_len, const int L);

__global__ void SPSS_without_normals_resample(const float *in_points,const float * in_reflect,const int *in_points_per_pix,
	float *out_points, float * out_reflect, int *out_points_per_pix, float * out_normals,
	const int T, const int Nrow, const int Ncol, const int pixhr, const float scale_ratio, const int impulse_len, const float proportion, const int L);

// others
__global__ void scale_normals(int * points_per_pix, float * in_normals, const float factor, const int Nrow, const int Ncol);


__global__ void upsample_pointcloud(float *in_points, float * in_reflect, int *in_points_per_pix, float * in_normals,
	float *out_points, float * out_reflect, int *out_points_per_pix, float * out_normals,
	const int Nrow, const int Ncol, const int upsampling);

__global__ void reset_normals(float * in_normals, int *in_points_per_pix, const int Nrow, const int Ncol);

__global__ void merge_points(float *in_points, float * in_normals, float *in_reflect, int *in_points_per_pix, const int Nrow, const int Ncol, const float min_dist);

__global__ void threshold_points(float *in_points, float * in_normals, float *in_reflect, int *in_points_per_pix, const int Nrow, const int Ncol, const float thres, const int L);


__global__ void shift_depth_kernel(float * in_points, const  int * in_points_per_pix, const int Nrow, const int Ncol, const int attack);

/* find maximum reflectivity in scene pixel */
__global__ void max_reflect_kernel(const float * in_reflect, float *out_reflect, const  int * in_points_per_pix, const int Nrow, const int Ncol, const int L);