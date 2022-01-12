#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "read_lidar.h"
#include "backgroundRegularization.cuh"

class LidarReconstruction {

public:

	typedef enum {RT3D, RT3D_bilateral, XCORR, XCORR_THRES, GRAD_DESC} algorithm;
	LidarReconstruction(void) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		results_available = false;
	};

	~LidarReconstruction(void);

	void AllocateFrame(LidarData &data, int frame=0, bool print_info= true);

	void allocateGPUMemory(LidarData &data, bool print_info = true);

	bool loadParameters(LidarData &data, bool manual = true, int algo=0);

	bool results(std::vector<float> & h_points, std::vector<float> & h_normals, std::vector<float> & h_reflect,
		std::vector<int> & h_points_per_pix, std::vector<float> & h_background, std::vector<float> &h_likelihood);

	void change_params(void);
	void run(bool print = true);

	std::vector<float> & getLikelihood(void) { return h_likelihood; };

	int getCloudHeight(void) { return height_cloud; };
	int getCloudWidth(void) { return width_cloud; };
	float getExecTime(void) { return exec_time; };
	float getScaleRatio(void) { return scale_ratio; };
	void run_frame(int frame, bool print);
	void saveHyperParameters(std::string filename);
	void loadHyperParameters(std::string filename);
	float thres, step_size_bkg, step_size_reflec, step_size_depth, reg_dep, proportion, lambda_reg_bkg, weight_coeff;
	float step_size_reflec_sketched, step_size_depth_sketched;

	std::string getAlgoName(void);
	void defaultHyperParameters(float mean_ppp, float SBR);
	void defaultHyperParameters(float mean_ppp);
private:
	std::string getHyperParametersFilename(std::string filename);
	void flip_pointer(void **p1, void **p2);
	void setFrame(int fr);
	void setBackgroundFilter(void);
	void SaveEntry(std::string & file, std::string name, float value);
	void SaveEntry(std::string & file, std::string name, int value);
	bool LoadEntryf(std::string & file, std::string name, float & value);
	bool LoadEntry(std::string & file, std::string name, int & value);

	void run_standard(int frame, bool print);
	void run_standard_thres(int frame, bool print);
	void run_palm_frame(int frame, bool print, bool likelihood);
	void run_grad_desc(int frame, bool print, bool likelihood);

	void run_palm_bilateral_frame(int frame, bool print, bool likelihood);
	// main subfunctions 
	void APSS_resampling(bool print_info = true);
	void APSS_resampling_bilateral(bool print_info);
	void APSSWithoutNormals(bool print_info = true);
	void initMatchFilter(bool print_info = true);
	void mergePoints(bool print_info = true);
	void thresholdPoints(bool print_info = true);
	void FilterBackground(bool print_info = true);
	void resetNormals(bool print_info = true);
	void computeLikelihood(bool print_info = true);
	void SPSSWithoutNormals(bool print_info = true);
	void upsamplePointCloud(bool print_info);
	void proxBackground(bool print_info = true);
	void drunkInitMatchFilter(bool print_info = true);
	void ReInitMatchFilter(bool print_info  = true);
	void gradientPointCloud(bool print_info);
	// Algorithm params
	int pix_h;
	type data_type;
	float mean_ppp, mean_gain;
	int splath;
	int subsampling, subsampling_sketched;
	int segments;
	float SBR, h_sigma;
	int algo_iter, frames;
	int shared_memory_size;
	int ndevices;
	algorithm algo;
	cudaEvent_t start, stop;
	dim3 block_cloud, grid_cloud, block_lidar, grid_lidar;
	int Nrow, Ncol, m, width_cloud, height_cloud, attack, impulse_len, T, sumH, upsampling, spectral_norm;
	float scale_ratio,mean_signal, sigmar2;
	float max_bkg, max_refl;
	bool results_available;
	int many_irf;
	// GPU Memory pointers
	int  *d_points_per_pix,  *d_points_per_pix2; // point cloud to estimate
	int  *out_points_per_pix,  *in_points_per_pix; // point cloud to estimate
	float *d_points2, *d_points, *in_points, *out_points, *d_normals, *d_normals2, *in_normals, *out_normals;
	float *d_background;
	float *in_background, *reg_background2, *reg_background;
	std::vector<float> h_likelihood;
	float *d_reflect, *d_reflect2;
	float *in_reflect, *out_reflect;
	float *d_likelihood;

	//enum dataset_type {
	//	SW_D,
	//	SW_S,
	//	MW_S
	//};

	cufftComplex *complex_fft_data;
	cufftReal *bkg_filter;
	cufftHandle fft_plan, ifft_plan;
	int *d_bins_counts, *d_pix_counts, *d_pix_counts_idx, *d_dense, *d_coded_aperture; // Lidar data
	float *d_sketched;
	std::vector<int *> d_frames_bins_counts, d_frames_pix_counts, d_frames_pix_counts_idx, d_frames_dense; // Container of frame pointers
	std::vector<float *> d_frames_sketched;
	float *d_impulse, *d_der_impulse, *d_integrated_impulse, *d_gain, *d_irf_norm, *d_cmean_irf, *d_log_impulse, *d_sketched_irf; // impulse response
	bool coaxial_lidar;
	int wavelengths;
	float exec_time;

};