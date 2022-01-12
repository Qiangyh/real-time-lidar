#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include "algorithm.h"
#include <iostream>
#include "GPU_macros.h"
#include "Point.h"
#include "logmatch.cuh"
#include "apss.cuh"
#include "misc.h"
#include "gradient.cuh"
#include <chrono>  // for high_resolution_clock
#include <boost/filesystem.hpp>

LidarReconstruction::~LidarReconstruction(void) {
	/************* Free Allocated Memory *************/
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// free point clouds
	cudaFree(d_points); cudaFree(d_reflect); cudaFree(d_points_per_pix); cudaFree(d_normals);
	cudaFree(d_points2); cudaFree(d_reflect2); cudaFree(d_points_per_pix2); cudaFree(d_normals2);


	// free background
	cudaFree(d_background); cudaFree(reg_background); cudaFree(reg_background2);


	// free data
	if (data_type == DENSE) {
		for (int fr = 0; fr < frames; fr++) {
			cudaFree(d_frames_dense[fr]);
		}
	} 
	else if (data_type == SPARSE) {
		for (int fr = 0; fr < frames; fr++) {
			cudaFree(d_frames_bins_counts[fr]); cudaFree(d_frames_pix_counts[fr]); cudaFree(d_frames_pix_counts_idx[fr]);
		}
	}
	else { // sketched
		for (int fr = 0; fr < frames; fr++) {
			cudaFree(d_frames_sketched[fr]);
		}
	}

	// free aperture
	cudaFree(d_coded_aperture);

	// free impulse response
	if (data_type == SKETCHED) {
		cudaFree(d_sketched_irf);
	} 
	else {
		cudaFree(d_impulse); cudaFree(d_der_impulse); cudaFree(d_log_impulse); cudaFree(d_integrated_impulse);
	}

	cudaFree(d_likelihood);
	
	// fft related stuff
	cufftDestroy(fft_plan);	cufftDestroy(ifft_plan);cudaFree(bkg_filter);cudaFree(complex_fft_data);
	
	std::cout << "Finished destroying algorithm" << std::endl;
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	//gpuErrchk(cudaDeviceReset());
};


bool LidarReconstruction::results(std::vector<float> & h_points, std::vector<float> & h_normals, std::vector<float> & h_reflect, std::vector<int> & h_points_per_pix, std::vector<float> & h_background, std::vector<float> & likelihood) {

	
	if (results_available) {
		int cloud_size = height_cloud * width_cloud;
		int lidar_size = Nrow * Ncol;
		// allocate memory for CPU transfer
		/************* Bring results from GPU Memory *************/


		shift_depth_kernel << < grid_cloud, block_cloud >> > (in_points, in_points_per_pix, height_cloud, width_cloud, attack);
		h_points.resize(cloud_size*MAX_POINTS_PER_PIX);
		gpuErrchk(cudaMemcpy(&h_points[0], in_points, cloud_size * MAX_POINTS_PER_PIX * sizeof(float), cudaMemcpyDeviceToHost));
		h_reflect.resize(cloud_size* MAX_POINTS_PER_PIX*wavelengths);
		gpuErrchk(cudaMemcpy(&h_reflect[0], in_reflect, cloud_size* MAX_POINTS_PER_PIX * sizeof(float) * wavelengths, cudaMemcpyDeviceToHost));
		h_points_per_pix.resize(cloud_size);
		gpuErrchk(cudaMemcpy(&h_points_per_pix[0], in_points_per_pix, cloud_size * sizeof(int), cudaMemcpyDeviceToHost));

		h_normals.resize(3*cloud_size*MAX_POINTS_PER_PIX);

		scale_normals <<< grid_lidar, block_lidar >>> (in_points_per_pix, in_normals, scale_ratio*weight_coeff, Nrow, Ncol);
		gpuErrchk(cudaMemcpy(&h_normals[0], in_normals, 3 * cloud_size * MAX_POINTS_PER_PIX * sizeof(float), cudaMemcpyDeviceToHost));

		if (wavelengths==1)
			exp_kernel << < grid_lidar, block_lidar >> > (in_background, reg_background2, Nrow, Ncol);
		else
			exp_MSkernel << < grid_lidar, block_lidar >> > (in_background, reg_background2, Nrow, Ncol, wavelengths);

		h_background.resize(lidar_size*wavelengths);
		gpuErrchk(cudaMemcpy(&h_background[0], reg_background2, wavelengths*lidar_size * sizeof(float), cudaMemcpyDeviceToHost));


		likelihood = h_likelihood;

		return true;

	}
	else {
		std::cout << "ERROR: There are no results available" << std::endl;
		return false;
	}

}


std::string LidarReconstruction::getAlgoName(void) {

	std::string out;
	if (data_type == SKETCHED) {
		switch (algo) {
		case RT3D:
			out = "SRT3D";
			break;
		case RT3D_bilateral:
			out = "SRT3Db";
			break;
		case XCORR:
			out = "init";
			break;
		case XCORR_THRES:
			out = "init_thres";
			break;
		case GRAD_DESC:
			out = "SMLE";
			break;
		};
		out += "_m";
		out += std::to_string(m);
		
	}
	else {
		switch (algo) {
		case RT3D:
			out = "RT3D";
			break;
		case RT3D_bilateral:
			out = "RT3Db";
			break;
		case XCORR:
			out = "XCORR";
			break;
		case XCORR_THRES:
			out = "XCORR_thres";
			break;
		case GRAD_DESC:
			out = "GRAD_DESC";
			break;
		};
	}

	out += '_';
	return out;
};



bool LidarReconstruction::loadParameters(LidarData &data, bool manual, int alg) {

	// get GPU
	cudaGetDeviceCount(&ndevices);

	if (ndevices == 0) {
		std::cout << "Sorry: A CUDA enabled GPU is needed to run this program" << std::endl;
		return false;
	}

	for (int d = 0; d < ndevices; d++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, d);
		std::cout << "Device " << d << ": " << prop.name << std::endl;
	}

	int dev = 0;

	if (ndevices > 1) {
		dev = ask_for_param("Choose GPU device: ", 0, ndevices, 0);
	}

	// set device
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, dev);
	shared_memory_size = prop.sharedMemPerBlock;
	gpuErrchk(cudaSetDevice(dev));

	// save basic data params
	data_type = data.getDataType();
	Nrow = data.getNrow(); 
	Ncol = data.getNcol();
	scale_ratio = data.getScaleRatio();
	impulse_len = data.getImpulseLen();
	T = data.getHistLen();
	m = data.getm();
	sumH = data.getSumImpulse();
	spectral_norm = data.getSpectralNorm();
	mean_ppp = data.getMeanPPP();
	mean_gain = data.getMeanGain();
	h_sigma = data.getSigma();
	frames = data.getFrameNumber();
	many_irf = data.MultipleIrf();
	data_type = data.getDataType();
	wavelengths = data.getL();
	attack = data.getAttack();

	// Main algorithm inputs
	//std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	if (manual) {
		std::cout << "SET ALGORITHM PARAMETERS:" << std::endl;
		std::cout << "If no input is selected, then a default value will be automatically asigned." << std::endl;
		std::cout << "All hyperparameter values are automatically saved in hyperparams/" + data.getFilename() + " and re-loaded the next time you run the same dataset." << std::endl;
	}

	std::string hyper_filename = getHyperParametersFilename("hyperparams/" + data.getFilename());
	if (boost::filesystem::exists(hyper_filename))
		loadHyperParameters(hyper_filename);
	else {
		if (manual)
			SBR = ask_for_paramf("Input an approximate signal-to-background ratio (SBR) ", 0., 100000., 1.); // 2 outdoor
		else
			SBR = 5.;

		defaultHyperParameters(mean_ppp);
	}
	

	int def_algo = 1;
	if (wavelengths > 1)
		def_algo = 2;

	if (alg == 0)
		if (data_type == SKETCHED)
			algo = algorithm(ask_for_param("Choose algorithm: \n1. SRT3D\n2. SRT3D bilateral \n3. Init.\n4. Init. with thres.\n5. SMLE\n", 1, 5, def_algo) - 1);
		else
			algo = algorithm(ask_for_param("Choose algorithm: \n1. RT3D\n2. RT3D bilateral \n3. Cross-correlation\n4. Cross-correlation with thresholding\n5. Gradient descent\n", 1, 5, def_algo) - 1);
	else
		algo = algorithm(alg - 1);

	if (manual)
		if (askYesNo("Do you want to set the hyperparameters manually? This might be necessary for datasets not provided by the authors.")) {

			scale_ratio = ask_for_paramf("Input the scale ratio: ", 0., 10000., scale_ratio);
			segments = ask_for_param("Input the number of points per pixel to init with: ", 0, MAX_POINTS_PER_PIX, segments);
			
			subsampling = ask_for_param("Input the subsampling for the log-matched filter init (integer value, 1 means no subsampling): ", 1, T / 10, subsampling);
	
			upsampling = ask_for_param("Input the upsampling rate (integer value, 1 means no upsampling): ", 1, MAX_UPSAMPLING, upsampling);


			if (algo == RT3D || algo == XCORR_THRES || algo == RT3D_bilateral) {
				thres = ask_for_paramf("Input the intensity threshold [photons]: ", 0, 10 * mean_ppp, thres);
			} 

			if (algo == RT3D) {
				proportion = ask_for_paramf("Input the proportion of dilated intensity in the denoising: ", 0., 1., proportion);// was 0.05 in outdoor step_size_reflec
			}
			if (algo == RT3D_bilateral) {
				reg_dep = ask_for_paramf("Input the amount of intensity regularization: ", 0.0001, 100000., reg_dep);// was 0.05 in outdoor step_size_reflec
			}

			if (algo == RT3D || algo == RT3D_bilateral || algo == GRAD_DESC) {
				algo_iter = ask_for_param("Input the number of iterations: ", 0, 100000, algo_iter);

				if (data_type == SKETCHED) {
					step_size_reflec_sketched = ask_for_paramf("Input the (sketched) gradient reflectivity step size: ", 0., 10000., step_size_reflec_sketched);
					step_size_depth_sketched = ask_for_paramf("Input the (sketched) gradient depth step size: ", 0., 10000., step_size_depth_sketched); //*10 in outdoor camouflage video 
				}
				else {
					step_size_reflec = ask_for_paramf("Input the gradient reflectivity step size: ", 0., 10000., step_size_reflec);
					step_size_depth = ask_for_paramf("Input the gradient depth step size: ", 0., 10000., step_size_depth); //*10 in outdoor camouflage video 
				}
			}

			if (algo == RT3D || algo == RT3D_bilateral) {
				pix_h = ask_for_param("Input the neighborhood size [pixels]: ", 0, 2, pix_h);
				weight_coeff = ask_for_paramf("Input the APSS kernel size in depth (dt): ", 1., 100., weight_coeff); // 1.5 in outdoor scene
				lambda_reg_bkg = ask_for_paramf("Input the regularization parameter for the background: ", 0., 10000., lambda_reg_bkg); // it was 100 outdoors
				max_refl = ask_for_paramf("Input the maximum reflectivity: ", 0.0001, 10000., max_refl);
			}

		}

	max_bkg = 1e5;
	 
	// TODO: Fix FFT when Ncol>Nrow !
	if (Nrow<Ncol)
		coaxial_lidar = false;
	else
		coaxial_lidar = true;


	saveHyperParameters(hyper_filename);


	mean_signal = mean_ppp / mean_gain * (SBR / (1 + SBR)) / float(upsampling*upsampling);
	height_cloud = Nrow * upsampling;
	width_cloud = Ncol * upsampling;
	sigmar2 = (reg_dep*step_size_reflec*wavelengths*wavelengths*mean_signal*mean_signal*mean_signal) / 10;
	allocateGPUMemory(data);
	// Set Laplacian Filter
	setBackgroundFilter();


	return true;
};


void LidarReconstruction::defaultHyperParameters(float mean_ppp) {

	algo_iter = 25;
	upsampling = int(std::ceil(90 / float(Ncol)));
	mean_signal = mean_ppp/mean_gain * (SBR / (1 + SBR)) / float(upsampling*upsampling);
	pix_h = 1;
	weight_coeff = 6; // 1.5;
	reg_dep = 1;
	scale_ratio /= float(upsampling);
	proportion = 0.1; 
	thres = 0.01 * mean_signal;
	segments = 1; // 2 surfaces
	lambda_reg_bkg = 10;
	max_refl =  500*mean_signal;
	step_size_reflec = 1 / (max_refl*sumH);
	step_size_depth = h_sigma * h_sigma / mean_signal / float(upsampling*upsampling);// /2
	step_size_reflec_sketched = 0.005; // / spectral_norm;
	step_size_depth_sketched = 0.01 / mean_signal / float(upsampling*upsampling);
	subsampling = 1; //std::ceil(T / 500.);
	subsampling_sketched = 1; // std::ceil(T / 500.);
}

void LidarReconstruction::defaultHyperParameters(float mean_ppp, float SBR) {

	algo_iter = 25;
	upsampling = int(std::ceil(90 / float(Ncol)));
	mean_signal = mean_ppp / mean_gain * (SBR / (1 + SBR)) / float(upsampling*upsampling);
	pix_h = 1;
	weight_coeff = 6; // 1.5;
	reg_dep = 1;
	scale_ratio /= float(upsampling);
	proportion = 0.1;
	thres = 0.01 * mean_signal;
	segments = 1;
	lambda_reg_bkg = 10;
	max_refl = 10 * mean_signal;
	step_size_reflec = 1 / (max_refl*sumH);
	step_size_depth = h_sigma * h_sigma / mean_ppp / float(upsampling*upsampling);// /2
	step_size_reflec_sketched = 0.01; // / spectral_norm;
	step_size_depth_sketched =  1. / mean_signal / float(upsampling*upsampling);
	subsampling = 1; // std::ceil(T / 500.);
	subsampling_sketched = 1; // std::ceil(T / 500.);
}

std::string LidarReconstruction::getHyperParametersFilename(std::string filename) {
	return filename + std::string(".hyp");
}

void LidarReconstruction::loadHyperParameters(std::string filename) {

	std::ifstream file(filename);

	std::string str;

	file.seekg(0, std::ios::end);
	str.reserve(file.tellg());
	file.seekg(0, std::ios::beg);

	str.assign((std::istreambuf_iterator<char>(file)),
		std::istreambuf_iterator<char>());

	file.close();

	//std::cout << str;

	int d; 
	float f;

	if (LoadEntry(str, "algo_iter", d))
		algo_iter = d;

	if (LoadEntryf(str, "SBR", f))
		SBR = f;

	if (LoadEntry(str, "upsampling", d))
		upsampling = d;

	if (LoadEntry(str, "pix_h", d))
		pix_h = d;

	if (LoadEntryf(str, "scale_ratio", f))
		scale_ratio = f;

	if (LoadEntryf(str, "weight_coeff", f))
		weight_coeff = f;

	if (LoadEntryf(str, "proportion", f))
		proportion = f;

	if (LoadEntry(str, "segments", d))
		segments = d;

	if (LoadEntryf(str, "lambda_reg_bkg", f))
		lambda_reg_bkg = f;

	if (LoadEntryf(str, "thres", f))
		thres = f;

	if (LoadEntryf(str, "max_refl", f))
		max_refl = f;

	if (LoadEntryf(str, "step_size_reflec", f))
		step_size_reflec = f;

	if (LoadEntryf(str, "step_size_depth", f))
		step_size_depth = f;

	if (LoadEntryf(str, "step_size_reflec_sketched", f))
		step_size_reflec_sketched = f;

	if (LoadEntryf(str, "step_size_depth_sketched", f))
		step_size_depth_sketched = f;


	if (LoadEntry(str, "subsampling", d))
		subsampling = d;

	if (LoadEntry(str, "subsampling_sketched", d))
		subsampling_sketched = d;

	if (LoadEntryf(str, "reg_dep", f))
		reg_dep = f;
}

bool LidarReconstruction::LoadEntryf(std::string & file, std::string name, float & number) {

	std::string::size_type pos1 = file.find(name);
	if (pos1 != std::string::npos) {
		pos1 = file.find("=", pos1);
		std::string::size_type pos2 = file.find(";", pos1);
		number = stof(file.substr(pos1 + 1, pos1 - pos2 - 1));
		return true;
	}
	else
		return false;
}

bool LidarReconstruction::LoadEntry(std::string & file, std::string name, int & number) {

	std::string::size_type pos1 = file.find(name);
	if (pos1 != std::string::npos) {
		pos1 = file.find("=", pos1);
		std::string::size_type pos2 = file.find(";", pos1);
		//std::cout << file.substr(pos1 + 1, pos1 - pos2 - 1);
		number = stoi(file.substr(pos1 + 1, pos1 - pos2 - 1));
		return true;
	}
	else
		return false;
}

void LidarReconstruction::SaveEntry(std::string & file, std::string name, int value) {
	file += name;
	file += "=";
	file += std::to_string(value);
	file += ";";
}

void LidarReconstruction::SaveEntry(std::string & file, std::string name, float value) {
	file += name;
	file += "=";
	file += std::to_string(value);
	file += ";";
}

void LidarReconstruction::saveHyperParameters(std::string filename) {

	std::string str;

	SaveEntry(str, "algo_iter", algo_iter);
	SaveEntry(str, "SBR", SBR);
	SaveEntry(str, "upsampling", upsampling);
	SaveEntry(str, "scale_ratio", scale_ratio);
	SaveEntry(str, "pix_h", pix_h);
	SaveEntry(str, "weight_coeff", weight_coeff);
	SaveEntry(str, "proportion", proportion);
	SaveEntry(str, "segments", segments);
	SaveEntry(str, "lambda_reg_bkg", lambda_reg_bkg);
	SaveEntry(str, "thres", thres);
	SaveEntry(str, "max_refl", max_refl);
	SaveEntry(str, "step_size_reflec", step_size_reflec);
	SaveEntry(str, "step_size_depth", step_size_depth);
	SaveEntry(str, "step_size_reflec_sketched", step_size_reflec_sketched);
	SaveEntry(str, "step_size_depth_sketched", step_size_depth_sketched);
	SaveEntry(str, "subsampling", subsampling);
	SaveEntry(str, "subsampling_sketched", subsampling_sketched);
	SaveEntry(str, "reg_dep", reg_dep);

	std::ofstream file(filename); // , std::ios_base::binary);
	file << str;
	file.close();
}

void LidarReconstruction::allocateGPUMemory(LidarData &data, bool print_info) {

	int cloud_size = height_cloud * width_cloud;
	int lidar_size = Nrow * Ncol;
	/************* Memory conf *************/
	// Allocate GPU buffers 
	gpuErrchk(cudaMalloc((void**)&d_points, cloud_size * MAX_POINTS_PER_PIX * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_points2, cloud_size * MAX_POINTS_PER_PIX * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_reflect, cloud_size * MAX_POINTS_PER_PIX * sizeof(float)*wavelengths));
	gpuErrchk(cudaMalloc((void**)&d_reflect2, cloud_size * MAX_POINTS_PER_PIX * sizeof(float)*wavelengths));


	gpuErrchk(cudaMalloc((void**)&d_normals, cloud_size * 3 * MAX_POINTS_PER_PIX * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_normals2, cloud_size * 3 * MAX_POINTS_PER_PIX * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_background, wavelengths*lidar_size * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&reg_background, wavelengths*lidar_size * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&reg_background2, wavelengths*lidar_size * sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_points_per_pix, cloud_size * sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_points_per_pix2, cloud_size * sizeof(int)));
	gpuErrchk(cudaMemset(d_points_per_pix, 0, cloud_size * sizeof(int)));
	gpuErrchk(cudaMemset(d_points_per_pix2, 0, cloud_size * sizeof(int)));

	if (data_type == DENSE) {
		d_frames_dense.resize(frames);
	}
	else if (data_type == SPARSE) {
		d_frames_bins_counts.resize(frames);
		d_frames_pix_counts.resize(frames);
		d_frames_pix_counts_idx.resize(frames);
	}
	else { // SKETCHED
		d_frames_sketched.resize(frames);
	}


	for (int i = 0; i < frames; i++) {
		AllocateFrame(data, i, false);
	}


	size_t mem_size;


	if (data_type == SKETCHED) {
		if (many_irf == 0)
			mem_size = 2 * m * sizeof(float)* wavelengths;
		else
			mem_size = Nrow * Ncol* 2 * m * sizeof(float);

		gpuErrchk(cudaMalloc((void**)&d_sketched_irf, mem_size));
		gpuErrchk(cudaMemcpy(d_sketched_irf, data.getSketchedIrfPtr(), mem_size, cudaMemcpyHostToDevice));

		if (many_irf == 0)
			mem_size = sizeof(float)* wavelengths;
		else
			mem_size = Nrow * Ncol * sizeof(float);

		gpuErrchk(cudaMalloc((void**)&d_cmean_irf, mem_size));
		gpuErrchk(cudaMemcpy(d_cmean_irf, data.getCircMeanPtr(), mem_size, cudaMemcpyHostToDevice));

	}
	else { // DENSE or SPARSE

		if (many_irf == 0)
			mem_size = impulse_len * sizeof(float)* wavelengths;
		else
			mem_size = Nrow * Ncol* impulse_len * sizeof(float);

		gpuErrchk(cudaMalloc((void**)&d_impulse, mem_size));
		gpuErrchk(cudaMemcpy(d_impulse, data.getImpulsePtr(), mem_size, cudaMemcpyHostToDevice));

		gpuErrchk(cudaMalloc((void**)&d_log_impulse, mem_size));
		gpuErrchk(cudaMemcpy(d_log_impulse, data.getLogImpulsePtr(), mem_size, cudaMemcpyHostToDevice));

		gpuErrchk(cudaMalloc((void**)&d_der_impulse, mem_size));
		gpuErrchk(cudaMemcpy(d_der_impulse, data.getDerImpulsePtr(), mem_size, cudaMemcpyHostToDevice));

		gpuErrchk(cudaMalloc((void**)&d_integrated_impulse, mem_size));
		gpuErrchk(cudaMemcpy(d_integrated_impulse, data.getIntImpulsePtr(), mem_size, cudaMemcpyHostToDevice));


		gpuErrchk(cudaMalloc((void**)&d_irf_norm, lidar_size * sizeof(float)));
		gpuErrchk(cudaMemcpy(d_irf_norm, data.getIrfNorm(), lidar_size * sizeof(float), cudaMemcpyHostToDevice));
	}


	gpuErrchk(cudaMalloc((void**)&d_gain, lidar_size * sizeof(float)));
	gpuErrchk(cudaMemcpy(d_gain, data.getGain(), lidar_size * sizeof(float), cudaMemcpyHostToDevice));

	if (wavelengths > 1) {
		gpuErrchk(cudaMalloc((void**)&d_coded_aperture, lidar_size * sizeof(int)));
		gpuErrchk(cudaMemcpy(d_coded_aperture, data.getCodedAperture(), lidar_size * sizeof(float), cudaMemcpyHostToDevice));
	}

	gpuErrchk(cudaMalloc((void**)&d_likelihood, lidar_size * sizeof(float)));




	// set in and out pointers
	in_points = d_points;
	out_points = d_points2;
	in_points_per_pix = d_points_per_pix;
	out_points_per_pix = d_points_per_pix2;
	in_reflect = d_reflect;
	out_reflect = d_reflect2;
	in_normals = d_normals;
	out_normals = d_normals2;
	in_background = d_background;

	
	/************* Grid conf *************/
	// calculate grid size
	//int block_size = round(sqrt((float)shared_memory_size / (sizeof(int) + sizeof(float) * (4+wavelengths) * MAX_POINTS_PER_PIX)) - 2 *pix_h);
	int block_size = round(sqrt((float)(4*shared_memory_size/3) / (3*(sizeof(int) + sizeof(float) * (4+wavelengths) * MAX_POINTS_PER_PIX))) - 2 *pix_h);

	if (block_size > 32)
		block_size = 32;

	//if (print_info)
		//std::cout << "Cloud block size: " << block_size << std::endl;


	block_cloud.x = block_size;
	block_cloud.y = block_size;
	block_cloud.z = 1;
	grid_cloud.x = std::ceil(float(height_cloud) / block_size); 
	grid_cloud.y = std::ceil(float(width_cloud) / block_size);
	grid_cloud.z = 1;

	int linear_block_size = 1024;
	block_lidar.x = linear_block_size;
	block_lidar.y = 1;
	block_lidar.z = 1;
	grid_lidar.x = std::ceil(float(Nrow*Ncol) / float(linear_block_size));
	grid_lidar.y = 1;
	grid_lidar.z = 1;


}


void LidarReconstruction::setBackgroundFilter(void) {

	// initialize bkg filtering
	gpuErrchk(cudaMalloc((void**)&complex_fft_data, Nrow* (Ncol/2+1) * sizeof(cufftComplex)));
	gpuErrchk(cudaMalloc((void**)&bkg_filter, Nrow* Ncol* sizeof(cufftReal)));
	setCudaFFT(Nrow, Ncol, &fft_plan, &ifft_plan);


	std::vector<float> h_Kernel(Nrow*Ncol, 0.);

	int x = 0, y = 0; h_Kernel[get_bkg_idx(x, y, Nrow, Ncol)] = 20;

	x = 1; y = 0;	h_Kernel[get_bkg_idx(x, y, Nrow, Ncol)] = -8;
	x = 0; y = 1;	h_Kernel[get_bkg_idx(x, y, Nrow, Ncol)] = -8;
	x = 0; y = Ncol - 1; h_Kernel[get_bkg_idx(x, y, Nrow, Ncol)] = -8;
	x = Nrow - 1; y = 0; h_Kernel[get_bkg_idx(x, y, Nrow, Ncol)] = -8;

	x = 1; y = 1;	h_Kernel[get_bkg_idx(x, y, Nrow, Ncol)] = 2;
	x = Nrow - 1; y = Ncol - 1; h_Kernel[get_bkg_idx(x, y, Nrow, Ncol)] = 2;
	x = Nrow - 1; y = 1;	h_Kernel[get_bkg_idx(x, y, Nrow, Ncol)] = 2;
	x = 1; y = Ncol - 1; h_Kernel[get_bkg_idx(x, y, Nrow, Ncol)] = 2;

	x = 2; y = 0;	h_Kernel[get_bkg_idx(x, y, Nrow, Ncol)] = 1;
	x = 0; y = 2;	h_Kernel[get_bkg_idx(x, y, Nrow, Ncol)] = 1;
	x = 0; y = Ncol - 2; h_Kernel[get_bkg_idx(x, y, Nrow, Ncol)] = 1;
	x = Nrow - 2; y = 0; h_Kernel[get_bkg_idx(x, y, Nrow, Ncol)] = 1;

	int lidar_size = Nrow * Ncol;


	gpuErrchk(cudaMemcpy(d_background, &h_Kernel[0], lidar_size * sizeof(float), cudaMemcpyHostToDevice));
 	run_FFT(&fft_plan, (cufftReal *)d_background, complex_fft_data);
	extract_real_symmetric_fft <<< grid_lidar, block_lidar >> > (complex_fft_data, bkg_filter, Nrow, Ncol);

} 

void LidarReconstruction::AllocateFrame(LidarData &data, int frame, bool print_info) {

	//allocate frame
	cudaEventRecord(start);
	if (data_type == DENSE)
	{
		gpuErrchk(cudaMalloc((void**)&(d_frames_dense[frame]), data.getNrow()* data.getNcol() * data.getHistLen() * sizeof(int)));
		gpuErrchk(cudaMemcpy(d_frames_dense[frame], data.getDense(frame), data.getNrow()* data.getNcol() * data.getHistLen() * sizeof(int), cudaMemcpyHostToDevice));

	}
	else if (data_type == SPARSE) {
		gpuErrchk(cudaMalloc((void**)&(d_frames_bins_counts[frame]), data.getTotalActive(frame) * 2 * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&(d_frames_pix_counts_idx[frame]), data.getNrow()* data.getNcol() * sizeof(int)));
		gpuErrchk(cudaMalloc((void**)&(d_frames_pix_counts[frame]), data.getNrow()* data.getNcol() * sizeof(int)));

		gpuErrchk(cudaMemcpy(d_frames_bins_counts[frame], data.getAllBinsCounts(frame), data.getTotalActive(frame) * 2 * sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_frames_pix_counts_idx[frame], data.getPerPixActiveBinsIdx(frame), data.getNrow()* data.getNcol() * sizeof(int), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_frames_pix_counts[frame], data.getPerPixActiveBins(frame), data.getNrow()* data.getNcol() * sizeof(int), cudaMemcpyHostToDevice));
	}
	else { // SKETCHED
		gpuErrchk(cudaMalloc((void**)&(d_frames_sketched[frame]), data.getNrow()* data.getNcol() * data.getm() * 2 * sizeof(float)));
		gpuErrchk(cudaMemcpy(d_frames_sketched[frame], data.getSketched(frame), data.getNrow()* data.getNcol() * data.getm() * 2 * sizeof(float), cudaMemcpyHostToDevice));

	}
	cudaEventRecord(stop);

	if (print_info){
		cudaEventSynchronize(stop);
		float miliseconds = 0;
		cudaEventElapsedTime(&miliseconds, start, stop);
		std::cout << "Frame allocation completed..." << std::endl;
		std::cout << "Elapsed time: " << miliseconds << std::endl;
	}
}


void LidarReconstruction::computeLikelihood(bool print_info) {

	cudaEventRecord(start);
	if (data_type == DENSE) {
		dense_likelihood_kernel << < grid_lidar, block_lidar >> > (d_likelihood, in_points, in_reflect, in_points_per_pix, in_background, reg_background,
			d_dense, d_impulse, d_integrated_impulse, impulse_len, T, Nrow, Ncol, upsampling, many_irf, d_gain);
	}
	else if (data_type == SPARSE) {
		if (wavelengths == 1) {
			likelihood_kernel << < grid_lidar, block_lidar >> > (d_likelihood, in_points, in_reflect, in_points_per_pix, in_background, reg_background,
				d_bins_counts, d_pix_counts, d_pix_counts_idx, d_impulse, d_integrated_impulse, impulse_len, T, Nrow, Ncol, upsampling, many_irf, d_gain);
		}
		else {
			likelihood_MS_kernel << < grid_lidar, block_lidar >> > (d_likelihood, in_points, in_reflect, in_points_per_pix, in_background, reg_background,
				d_bins_counts, d_pix_counts, d_pix_counts_idx, d_impulse, d_integrated_impulse, impulse_len, T, Nrow, Ncol, upsampling, many_irf, d_gain, d_coded_aperture, wavelengths);
		}
	}
	else { // sketched
		sketch_likelihood_kernel << < grid_lidar, block_lidar >> > (d_likelihood, in_points, in_reflect, in_points_per_pix,
			d_sketched, d_sketched_irf, m, T, Nrow, Ncol, upsampling, many_irf, d_gain);
	}

	global_reduce(d_likelihood, d_likelihood, Nrow*Ncol, shared_memory_size);
	h_likelihood.push_back(0.);
	gpuErrchk(cudaMemcpy(&h_likelihood.back(), d_likelihood, sizeof(float), cudaMemcpyDeviceToHost));
	cudaEventRecord(stop);


	std::cout << "likelihood: " << h_likelihood.back() << std::endl;

	if (print_info) {
		cudaEventSynchronize(stop);
		float miliseconds = 0;
		cudaEventElapsedTime(&miliseconds, start, stop);
		std::cout << "Likelihood computation completed..." << std::endl;
		std::cout << "Elapsed time: " << miliseconds << std::endl;
	}

}

void LidarReconstruction::initMatchFilter(bool print_info) {

	gpuErrchk(cudaMemset(in_points_per_pix, 0, height_cloud*width_cloud * sizeof(int)));

	cudaEventRecord(start);
	if (data_type == DENSE) {
		denseLogMatchedFilterOMP << < grid_lidar, block_lidar >> >
			(in_points, in_reflect, in_points_per_pix, in_background, d_impulse, d_integrated_impulse, d_dense,
				impulse_len, T, Nrow, Ncol, upsampling, subsampling, segments, max_bkg, many_irf, d_gain, d_irf_norm, SBR);
	}
	else if (data_type == SPARSE) {
		if (wavelengths==1){
			logMatchedFilterKernel << < grid_lidar, block_lidar >> >
				(in_points, in_reflect, in_points_per_pix, in_background, d_impulse, d_integrated_impulse, d_bins_counts,
					d_pix_counts, d_pix_counts_idx, impulse_len, T, Nrow, Ncol, upsampling, subsampling, segments, max_bkg, many_irf, d_gain);
		}
		else {
			if (segments == 1) {
				half_sample_mode << < grid_lidar, block_lidar >> >
					(in_points, in_reflect, in_points_per_pix, in_background, d_bins_counts, d_pix_counts, d_pix_counts_idx, impulse_len,
						T, Nrow, Ncol, upsampling, max_bkg, d_gain, wavelengths, attack);
			}
			else {
				logMatchedMSFilterKernel << < grid_lidar, block_lidar, shared_memory_size >> >
					(in_points, in_reflect, in_points_per_pix, in_background, d_impulse, d_integrated_impulse, d_bins_counts,
						d_pix_counts, d_pix_counts_idx, impulse_len, T, Nrow, Ncol, upsampling, subsampling, segments, max_bkg, d_gain, d_coded_aperture, wavelengths);
			}
		}
	}
	else { // sketched
		if (false) { // one peak
			circularMean << < grid_lidar, block_lidar >> >
				(in_points, in_reflect, in_points_per_pix, d_sketched_irf, d_cmean_irf, d_sketched, m,
					T, Nrow, Ncol, upsampling, many_irf, d_gain);
		}
		else {
			sketchedMultiPeakOMPInit << < grid_lidar, block_lidar >> >
				(in_points, in_reflect, in_points_per_pix, d_sketched_irf, segments, d_sketched, m,
					T, Nrow, Ncol, upsampling, subsampling_sketched, many_irf, d_gain, spectral_norm);
		}
	}

	cudaEventRecord(stop);


	if (print_info) {
		cudaEventSynchronize(stop);
		float miliseconds = 0;
		cudaEventElapsedTime(&miliseconds, start, stop);
		std::cout << "Log-match filtering completed..." << std::endl;
		std::cout << "Elapsed time: " << miliseconds << std::endl;
	}
}


void LidarReconstruction::ReInitMatchFilter(bool print_info) {

	/*
	float min_dist = weight_coeff * scale_ratio*(2 * pix_h + 1);

	if (min_dist < 2)
		min_dist = 2;

	gpuErrchk(cudaMemset(in_points_per_pix, 0, height_cloud*width_cloud * sizeof(int)));*/
	
	if (data_type == DENSE) {
		cudaEventRecord(start);
		denseLogMatchedFilterOMP << < grid_lidar, block_lidar >> >
			(in_points, in_reflect, in_points_per_pix, in_background, d_impulse, d_integrated_impulse, d_dense,
				impulse_len, T, Nrow, Ncol, upsampling, subsampling, segments, max_bkg, many_irf, d_gain, d_irf_norm, SBR);
		cudaEventRecord(stop);
	}
	else if (data_type == SPARSE) {
		cudaEventRecord(start);
		ReInitLogMatchedFilterKernel << < grid_lidar, block_lidar >> >
			(in_points, in_reflect, in_points_per_pix, in_background, d_log_impulse, d_integrated_impulse,
			d_bins_counts, d_pix_counts, d_pix_counts_idx, impulse_len, T, Nrow, Ncol, upsampling, subsampling,
			1, many_irf, d_gain, weight_coeff*scale_ratio*(2*pix_h + 1));
		cudaEventRecord(stop);
	} 
	else // TODO sketched
	{ 


	}

	if (print_info) {
		cudaEventSynchronize(stop);
		float miliseconds = 0;
		cudaEventElapsedTime(&miliseconds, start, stop);
		std::cout << "Re-Init Log-match filtering completed..." << std::endl;
		std::cout << "Elapsed time: " << miliseconds << std::endl;
	}
}

void LidarReconstruction::drunkInitMatchFilter(bool print_info) {


	gpuErrchk(cudaMemset(in_points_per_pix, 0, height_cloud*width_cloud * sizeof(int)));
	if (data_type == DENSE) {
		cudaEventRecord(start);
		denseLogMatchedFilterOMP << < grid_lidar, block_lidar >> >
			(in_points, in_reflect, in_points_per_pix, in_background, d_impulse, d_integrated_impulse, d_dense,
				impulse_len, T, Nrow, Ncol, upsampling, subsampling, segments, max_bkg, many_irf, d_gain, d_irf_norm,SBR);
		cudaEventRecord(stop);
	}
	else if (data_type == SPARSE) {
		cudaEventRecord(start);
		logMatchedFilterKernel << < grid_lidar, block_lidar >> >
			(in_points, in_reflect, in_points_per_pix, in_background, d_log_impulse, d_integrated_impulse, d_bins_counts,
				d_pix_counts, d_pix_counts_idx, impulse_len, T, Nrow, Ncol, upsampling, subsampling, segments, max_bkg, many_irf, d_gain);
		cudaEventRecord(stop);
	}
	else { // todo sketched


	}

	if (print_info) {
		cudaEventSynchronize(stop);
		float miliseconds = 0;
		cudaEventElapsedTime(&miliseconds, start, stop);
		std::cout << "Drunk Log-match filtering completed..." << std::endl;
		std::cout << "Elapsed time: " << miliseconds << std::endl;
	}
}

void LidarReconstruction::flip_pointer(void **p1, void **p2) {
	void * aux;
	aux = (*p1);
	(*p1) = (*p2);
	(*p2) = aux;
}


void LidarReconstruction::APSS_resampling(bool print_info) {

	cudaEventRecord(start);

	//std::cout << "sigmar2: " << sigmar2 << std::endl;
	/*APSS_with_normals_resample_bilateral << < grid_cloud, block_cloud, shared_memory_size >> >
		(in_points, in_normals, in_reflect, in_points_per_pix, out_points, out_normals,
			out_reflect, out_points_per_pix, T, height_cloud, width_cloud, pix_h, sigmar2, scale_ratio*weight_coeff, impulse_len, wavelengths);*/
	
	APSS_with_normals_resample<< < grid_cloud, block_cloud, shared_memory_size >> >
		(in_points, in_normals, in_reflect, in_points_per_pix, out_points, out_normals,
			out_reflect, out_points_per_pix, T, height_cloud, width_cloud, pix_h, proportion, scale_ratio*weight_coeff, impulse_len, wavelengths);
	
	
	cudaEventRecord(stop);


	// flip pointers
	flip_pointer((void **)&in_points, (void **)&out_points);
	flip_pointer((void **)&in_reflect, (void **)&out_reflect);
	flip_pointer((void **)&in_normals, (void **)&out_normals);
	flip_pointer((void **)&in_points_per_pix, (void **)&out_points_per_pix);

	if (print_info) {
		cudaEventSynchronize(stop);
		float miliseconds = 0;
		cudaEventElapsedTime(&miliseconds, start, stop);
		std::cout << "APSS step completed..." << std::endl;
		std::cout << "Elapsed time: " << miliseconds << std::endl;
	}
}


void LidarReconstruction::APSS_resampling_bilateral(bool print_info) {

	cudaEventRecord(start);

	//std::cout << "sigmar2: " << sigmar2 << std::endl;
	APSS_with_normals_resample_bilateral << < grid_cloud, block_cloud, shared_memory_size >> >
	(in_points, in_normals, in_reflect, in_points_per_pix, out_points, out_normals,
	out_reflect, out_points_per_pix, T, height_cloud, width_cloud, pix_h, sigmar2, scale_ratio*weight_coeff, impulse_len, wavelengths);

	cudaEventRecord(stop);


	// flip pointers
	flip_pointer((void **)&in_points, (void **)&out_points);
	flip_pointer((void **)&in_reflect, (void **)&out_reflect);
	flip_pointer((void **)&in_normals, (void **)&out_normals);
	flip_pointer((void **)&in_points_per_pix, (void **)&out_points_per_pix);

	if (print_info) {
		cudaEventSynchronize(stop);
		float miliseconds = 0;
		cudaEventElapsedTime(&miliseconds, start, stop);
		std::cout << "APSS step completed..." << std::endl;
		std::cout << "Elapsed time: " << miliseconds << std::endl;
	}
}




void LidarReconstruction::mergePoints(bool print_info) {

	cudaEventRecord(start);
	merge_points << <grid_cloud, block_cloud >> >
		(in_points, in_normals, in_reflect, in_points_per_pix, height_cloud, width_cloud, 2*ceil((pix_h+0.5)*scale_ratio*weight_coeff));
	cudaEventRecord(stop);


	if (print_info) {
		cudaEventSynchronize(stop);
		float miliseconds = 0;
		cudaEventElapsedTime(&miliseconds, start, stop);
		std::cout << "Merge step completed..." << std::endl;
		std::cout << "Elapsed time: " << miliseconds << std::endl;
	}
}

void LidarReconstruction::thresholdPoints(bool print_info) {

	
	cudaEventRecord(start);	
	threshold_points <<< grid_cloud, block_cloud >> >
		(in_points, in_normals, in_reflect, in_points_per_pix, height_cloud, width_cloud, thres, wavelengths);
	cudaEventRecord(stop);


	if (print_info) {
		cudaEventSynchronize(stop);
		float miliseconds = 0;
		cudaEventElapsedTime(&miliseconds, start, stop);
		std::cout << "Thresholding step completed..." << std::endl;
		std::cout << "Elapsed time: " << miliseconds << std::endl;
	}
}

void LidarReconstruction::resetNormals(bool print_info) {

	cudaEventRecord(start);
	reset_normals <<< grid_cloud, block_cloud >>> 
		(in_normals, in_points_per_pix, height_cloud, width_cloud);
	cudaEventRecord(stop);


	if (print_info) {
		cudaEventSynchronize(stop);
		float miliseconds = 0;
		cudaEventElapsedTime(&miliseconds, start, stop);
		std::cout << "Resetting normals step completed..." << std::endl;
		std::cout << "Elapsed time: " << miliseconds << std::endl;
	}
}


void LidarReconstruction::gradientPointCloud(bool print_info) {


	// get maximum reflectivity
	
	/*float maxx;
	max_reflect_kernel << < grid_lidar, block_lidar >> >(in_reflect, out_reflect, in_points_per_pix, height_cloud, width_cloud,wavelengths);
	global_max_reduce(out_reflect, out_reflect, height_cloud*width_cloud, shared_memory_size);
	gpuErrchk(cudaMemcpy(&maxx, out_reflect, sizeof(float), cudaMemcpyDeviceToHost));

	if (maxx < max_refl)
		max_refl = maxx;

	step_size_reflec = 1. / (max_refl*sumH);*/


	cudaEventRecord(start);
	if (data_type == DENSE) 
		simple_dense_point_cloud_gradient_kernel << < grid_lidar, block_lidar >> > (in_points, in_reflect, in_points_per_pix, in_background,	
		d_dense, d_impulse, d_der_impulse, d_integrated_impulse, impulse_len, T, Nrow, Ncol, upsampling, step_size_depth,
			step_size_reflec, max_refl, many_irf, d_gain);
	else if (data_type == SPARSE) {
		if (wavelengths == 1)
			point_cloud_gradient_kernel << < grid_lidar, block_lidar >> > (in_points, in_reflect, in_points_per_pix, in_background,
				d_bins_counts, d_pix_counts, d_pix_counts_idx, d_impulse, d_der_impulse, d_integrated_impulse,
				impulse_len, T, Nrow, Ncol, upsampling, step_size_depth, step_size_reflec, max_refl, many_irf, d_gain);
		else
			point_cloud_ms_gradient_kernel << < grid_lidar, block_lidar >> > (in_points, in_reflect, in_points_per_pix, in_background,
				d_bins_counts, d_pix_counts, d_pix_counts_idx, d_impulse, d_der_impulse, d_integrated_impulse,
				impulse_len, T, Nrow, Ncol, upsampling, step_size_depth, step_size_reflec, max_refl, many_irf, d_gain, d_coded_aperture, wavelengths);
	}
	else { // sketched 
		sketched_gradient_kernel << < grid_lidar, block_lidar >> > (in_points, in_reflect, in_points_per_pix,
			d_sketched, d_sketched_irf, m, T, Nrow, Ncol, upsampling, step_size_depth_sketched,
			step_size_reflec_sketched, max_refl, many_irf, d_gain);
	}
	cudaEventRecord(stop);

	if (print_info) {
		cudaEventSynchronize(stop);
		float miliseconds = 0;
		cudaEventElapsedTime(&miliseconds, start, stop);
		std::cout << "Max reflectivity " << max_refl << std::endl;
		std::cout << "Point Cloud gradient step completed..." << std::endl;
		std::cout << "Elapsed time: " << miliseconds << std::endl;
	}
}

void LidarReconstruction::proxBackground(bool print_info) {

	if (data_type != SKETCHED) {

		cudaEventRecord(start);

		float max;
		global_max_reduce(in_background, reg_background, Nrow*Ncol*wavelengths, shared_memory_size);
		gpuErrchk(cudaMemcpy(&max, reg_background, sizeof(float), cudaMemcpyDeviceToHost));
	
		step_size_bkg = 1/(exp(max)*T);

	
		if (data_type == DENSE) {
			dense_prox_bkg_kernel << < grid_lidar, block_lidar >> > (in_points, in_reflect, in_points_per_pix, in_background,
				d_dense, d_impulse, d_integrated_impulse,
				impulse_len, T, Nrow, Ncol, upsampling, step_size_bkg, max_bkg, many_irf, d_gain);
		}
		else if (data_type == SPARSE) {
			if (wavelengths == 1) {
			prox_bkg_kernel <<< grid_lidar, block_lidar >> > (in_points, in_reflect, in_points_per_pix, in_background,
				d_bins_counts, d_pix_counts, d_pix_counts_idx, d_impulse, d_integrated_impulse,
				impulse_len, T, Nrow, Ncol, upsampling, step_size_bkg, max_bkg, many_irf, d_gain);
			}
			else {
				prox_MSbkg_kernel << < grid_lidar, block_lidar >> > (in_points, in_reflect, in_points_per_pix, in_background,
					d_bins_counts, d_pix_counts, d_pix_counts_idx, d_impulse, d_integrated_impulse,
					impulse_len, T, Nrow, Ncol, upsampling, step_size_bkg, max_bkg, many_irf, d_gain, d_coded_aperture, wavelengths);
			}
		} 
	 
	
		if (coaxial_lidar) {
			for (int l = 0; l < wavelengths; l++) {
				run_FFT(&fft_plan, (cufftReal *) &in_background[l*Nrow*Ncol], complex_fft_data);
				fft_kernel << < grid_lidar, block_lidar >> > (complex_fft_data, bkg_filter, step_size_bkg*lambda_reg_bkg, Nrow, Ncol);
				run_IFFT(&ifft_plan, complex_fft_data, (cufftReal *)&in_background[l*Nrow*Ncol]);
			}
		}

		cudaEventRecord(stop);

		if (print_info) {
			cudaEventSynchronize(stop);
			float miliseconds = 0;
			cudaEventElapsedTime(&miliseconds, start, stop);
			std::cout << "max background: " << exp(max)*T << " photons " << std::endl;
			std::cout << "Proximal background step completed..." << std::endl;
			std::cout << "Elapsed time: " << miliseconds << std::endl;
		}

	}

}


void LidarReconstruction::FilterBackground(bool print_info) {

	dim3 block(16, 16, 1);
	dim3 grid(std::ceil(float(Nrow) / 16), std::ceil(float(Ncol) / 16), 1);

	cudaEventRecord(start);
	exp_kernel << < grid, block >> >(in_background, reg_background, Nrow, Ncol);
	convolutionRowGPU <<< grid, block >>>(reg_background2, reg_background, Nrow, Ncol);
	convolutionColGPU <<< grid, block >>>(reg_background, reg_background2, Nrow, Ncol);
	cudaEventRecord(stop);

	if (print_info) {
		cudaEventSynchronize(stop);
		float miliseconds = 0;
		cudaEventElapsedTime(&miliseconds, start, stop);
		std::cout << "Background filtering step completed..." << std::endl;
		std::cout << "Elapsed time: " << miliseconds << std::endl;
	}
}



// TODO: not supported yet
void LidarReconstruction::APSSWithoutNormals(bool print_info) {

	cudaEventRecord(start);
	// put kernel here
	cudaEventRecord(stop);

	// flip pointers
	flip_pointer((void **)&in_points, (void **)&out_points);
	flip_pointer((void **)&in_reflect, (void **)&out_reflect);
	flip_pointer((void **)&in_points_per_pix, (void **)&out_points_per_pix);

	if (print_info) {
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "APSS without normals step completed..." << std::endl;
		std::cout << "Elapsed time: " << milliseconds << std::endl;
	}
	mergePoints(print_info);
}


void LidarReconstruction::upsamplePointCloud(bool print_info) {

	cudaEventRecord(start);
	upsample_pointcloud << < grid_lidar, block_lidar >> > (in_points, in_reflect, in_points_per_pix, in_normals,
		out_points, out_reflect, out_points_per_pix, out_normals, Nrow, Ncol, upsampling);
	cudaEventRecord(stop);


	// flip pointers
	flip_pointer((void **)&in_points, (void **)&out_points);
	flip_pointer((void **)&in_reflect, (void **)&out_reflect);
	flip_pointer((void **)&in_normals, (void **)&out_normals);
	flip_pointer((void **)&in_points_per_pix, (void **)&out_points_per_pix);


	if (print_info) {
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << "upsampling pointcloud step completed..." << std::endl;
		std::cout << "Elapsed time: " << milliseconds << std::endl;
	}
}

void LidarReconstruction::SPSSWithoutNormals(bool print_info) {

	cudaEventRecord(start);

	SPSS_without_normals_resample <<< grid_cloud, block_cloud, shared_memory_size >> >(in_points, in_reflect, in_points_per_pix,
		out_points, out_reflect, out_points_per_pix, in_normals, T, height_cloud, width_cloud, pix_h, scale_ratio*weight_coeff, impulse_len, proportion, wavelengths);
	cudaEventRecord(stop);

	// flip pointers
	flip_pointer((void **)&in_points, (void **)&out_points);
	flip_pointer((void **)&in_reflect, (void **)&out_reflect);
	flip_pointer((void **)&in_points_per_pix, (void **)&out_points_per_pix);

	if (print_info) {
		cudaEventSynchronize(stop);
		float miliseconds = 0;
		cudaEventElapsedTime(&miliseconds, start, stop);
		std::cout << "SPSS without normals step completed..." << std::endl;
		std::cout << "Elapsed time: " << miliseconds << std::endl;
	}
}

void LidarReconstruction::setFrame(int fr) {

	if (data_type == DENSE) {
		d_dense = d_frames_dense[fr];
	}
	else if (data_type == SPARSE) {
		d_pix_counts = d_frames_pix_counts[fr];
		d_pix_counts_idx = d_frames_pix_counts_idx[fr];
		d_bins_counts = d_frames_bins_counts[fr];
	} else { 
		d_sketched = d_frames_sketched[fr];
	}
};

void LidarReconstruction::run(bool print) {
	/************* Main Algo *************/


	for (int fr = 0; fr < frames; fr++) {
		run_frame(fr,print);
	}

}




void LidarReconstruction::run_standard(int frame, bool print) {
	/************* Main Algo *************/

	setFrame(frame);

	auto start = std::chrono::high_resolution_clock::now();

	initMatchFilter(print);

	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();
	exec_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	std::cout << "Elapsed time total frame = " << exec_time << " ms " << std::endl;

	results_available = true;
}


void LidarReconstruction::run_standard_thres(int frame, bool print) {
	/************* Main Algo *************/

	setFrame(frame);

	auto start = std::chrono::high_resolution_clock::now();

	initMatchFilter(print);
	thresholdPoints(print);

	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();
	exec_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	std::cout << "Elapsed time total frame = " << exec_time << " ms " << std::endl;

	results_available = true;
}




void LidarReconstruction::run_frame(int frame, bool print) {
	/************* Main Algo *************/
	bool compute_likelihood = false; // askYesNo("Compute likelihood?");
	switch (algo) {
		case RT3D:
			run_palm_frame(frame, print, compute_likelihood);
			break;
		case RT3D_bilateral:
			run_palm_bilateral_frame(frame, print, compute_likelihood);
			break;
		case XCORR:
			run_standard(frame, print);
			break;
		case XCORR_THRES:
			run_standard_thres(frame, print);
			break;
		case GRAD_DESC:
			run_grad_desc(frame, print, compute_likelihood);
			break;
	};

	

}


void LidarReconstruction::change_params(void) {

	std::string s;
	std::getline(std::cin, s);


	thres = ask_for_paramf("Input the intensity threshold: ", 0, 10*mean_ppp, thres);
	proportion = ask_for_paramf("Input the proportion of dilated intensity: ", 0, 1., proportion);
	weight_coeff = ask_for_paramf("Input the APSS anisotropy coefficient: ", 1., 100., weight_coeff);
}

void LidarReconstruction::run_palm_frame(int frame, bool print, bool likelihood) {
	/************* Main Algo *************/
	setFrame(frame);

	auto start = std::chrono::high_resolution_clock::now();

	initMatchFilter(print);

	if (data_type == SKETCHED) {
		for (int i = 0; i < 5; i++)
			gradientPointCloud(print);
		//thresholdPoints(print);
	}

	SPSSWithoutNormals(print);

	if (likelihood)
		computeLikelihood(print);

	// only print last iteration
	bool print2 = false;
	for (int i = 0; i < algo_iter; i++) {
		if (i == algo_iter - 1)
			if (print)
				print2 = true;

		// alternated minimization
		// point cloud
		gradientPointCloud(print2);
		thresholdPoints(print2);

		APSS_resampling(print2);

		// background
		proxBackground(print2);

		if (likelihood)
			computeLikelihood(print2);
	}

	gradientPointCloud(print2);
	thresholdPoints(print2);

	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();
	exec_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	std::cout << "Elapsed time total frame = " << exec_time << " ms "<< std::endl;

	results_available = true;
}


void LidarReconstruction::run_grad_desc(int frame, bool print, bool likelihood) {
	/************* Main Algo *************/
	setFrame(frame);

	auto start = std::chrono::high_resolution_clock::now();

	initMatchFilter(print);

	if (likelihood)
		computeLikelihood(print);

	// only print last iteration
	bool print2 = false;
	for (int i = 0; i < algo_iter; i++) {
		if (i == algo_iter - 1)
			if (print)
				print2 = true;

		gradientPointCloud(print2);

		if (likelihood)
			computeLikelihood(print2);
	}

	gradientPointCloud(print2);
	thresholdPoints(print2);

	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();
	exec_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	std::cout << "Elapsed time total frame = " << exec_time << " ms " << std::endl;

	results_available = true;
}


void LidarReconstruction::run_palm_bilateral_frame(int frame, bool print, bool likelihood) {
	/************* Main Algo *************/


	setFrame(frame);

	auto start = std::chrono::high_resolution_clock::now();

	initMatchFilter(print);

	if (data_type == SKETCHED) {
		for (int i = 0; i < 5; i++)
			gradientPointCloud(print);
		//thresholdPoints(print);
	}

	SPSSWithoutNormals(print);

	if (likelihood)
		computeLikelihood(print);

	// only print last iteration
	bool print2 = false;
	for (int i = 0; i < algo_iter; i++) {
		if (i == algo_iter - 1)
			if (print)
				print2 = true;

		if (i<5)
			sigmar2 = 10000000.;
		else
			sigmar2 = (reg_dep*step_size_reflec*wavelengths*wavelengths*mean_signal*mean_signal*mean_signal) / 10;

		// alternated minimization
		// point cloud
		gradientPointCloud(print2);
		thresholdPoints(print2);

		APSS_resampling_bilateral(print2);

		// background
		proxBackground(print2);

		if (likelihood)
			computeLikelihood(print2);
	}

	//gradientPointCloud(print2);
	thresholdPoints(print2);

	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();
	exec_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	std::cout << "Elapsed time total frame = " << exec_time << " ms " << std::endl;

	results_available = true;
}

