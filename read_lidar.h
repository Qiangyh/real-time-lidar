#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Patate/grenaille.h"

typedef enum { SPARSE, DENSE, SKETCHED } type;

#define MAX_BINS_PER_PIXEL 6000
#define MAX_ACTIVE_BINS_PER_PIXEL 512
#define MAX_M 61
#define MAX_ACTIVE_BINS_SPARSE 256
#define MAX_DENSE_BINS_PER_PIXEL 512
#define MAX_WAVELENGTHS 4
#define PII 6.28318530718

struct photon{
	 int bin;
	 int counts;
};


MULTIARCH inline  photon getBinCounts(const  int x, const  int  y, const  int k, const  int * bins_counts, const  int start_idx,  const  int Nrow) {

	photon phot;
	phot.bin = bins_counts[2*(start_idx + k) + 0];
	phot.counts = bins_counts[2*(start_idx + k)  + 1];
	return phot;
};


class SparseLidarFrame {

public:
	std::vector<photon> & data() { return sparse_data; };
	std::vector< int> & bins_act() { return _bins_act; };
	void setPPP(float p) { _mean_photons_per_pix=p;	};
	float getPPP(void) { return _mean_photons_per_pix; };
	std::vector< int> & bins_act_idx() { return _bins_act_idx; };

private:
	std::vector<photon> sparse_data;
	std::vector<int>  _bins_act, _bins_act_idx;
	float _mean_photons_per_pix;
};




class DenseLidarFrame {

public:
	std::vector<int> & data() { return _data; };
	void setPPP(float p) { _mean_photons_per_pix = p; };
	float getPPP(void) { return _mean_photons_per_pix; };



	DenseLidarFrame(int Nrow, int Ncol, int T) {
		_data.resize(Nrow*Ncol*T);
	}


	inline int read(int i, int j, int t, int Nrow, int Ncol, int T) {
		return _data[idx(i, j, t, Nrow, Ncol, T)];
	};


	inline void inc(int i, int j, int t, int Nrow, int Ncol, int T) {
		_data[idx(i,j,t,Nrow,Ncol,T)]++;
	};


	inline void set(int counts, int i, int j, int t, int Nrow, int Ncol, int T) {
		_data[idx(i, j, t, Nrow, Ncol, T)]=counts;
	};


private:

	inline int idx(int i, int j, int t, int Nrow, int Ncol, int T) {
		return T * (i + j * Nrow) + t;
	}
	std::vector<int> _data;
	float _mean_photons_per_pix;
};




class SketchedLidarFrame {

public:
	std::vector<float> & data() { return _data; };
	//std::vector<int> & ppp() { return _ppp; };
	void setPPP(float p) { _mean_photons_per_pix = p; };
	float getPPP(void) { return _mean_photons_per_pix; };


	SketchedLidarFrame(int Nrow, int Ncol, int m) {
		_data.resize(Nrow*Ncol*2*m);
		//std::fill(_data.begin(), _data.end(), 0.);
	}


	inline void add(float val, int i, int j, int t, bool imag, int Nrow, int Ncol, int m) {
		 _data[idx(i, j, t, imag, Nrow, Ncol, m)] += val;
	};

	inline float read(int i, int j, int t, bool imag, int Nrow, int Ncol, int m) {
		return _data[idx(i, j, t, imag, Nrow, Ncol, m)];
	};


	inline void set(float counts, int i, int j, int t, bool imag, int Nrow, int Ncol, int m) {
		_data[idx(i, j, t, imag, Nrow, Ncol, m)] = counts;
	};


	/*inline void inc_ppp(int counts, int i, int j, int Nrow, int Ncol) {
		_ppp[i + j * Nrow] += counts;
	}; */


private:

	inline int idx(int i, int j, int t, bool imag, int Nrow, int Ncol, int m) {
		if (imag) 
			return 2 * m * (i + j * Nrow) + m + t;
		else
			return 2 * m * (i + j * Nrow) + t;
	}
	std::vector<float> _data;
	//std::vector<int> _ppp;
	float _mean_photons_per_pix;
};


class LidarData {

public:
	typedef enum { HW_ARRAY, MATLAB_RASTER_SCAN, SYNTHETIC} encoding;

	typedef std::pair<std::string, LidarData::encoding> lidar_file;
	typedef std::vector<lidar_file> file_list;
	typedef file_list file_types;

	LidarData(int sketches=0) { // if sketches = 0, no sketching assumed
		available_data = false;
		sbr_available = false;
		m = sketches;
	};


	// find available datasets, let the user choose one and load it
	bool LoadDataset(bool print = true);
	bool LoadDataset(int id, bool print = true);


	float * getImpulseResponse(void) { return &impulse_response[0]; };

	int getNrow(void) { return Nrow; };

	int getNcol(void) { return Ncol; };

	int  getImpulseLen(void) { return impulse_len; };

	int  getHistLen(void) { return T; };

	float getScaleRatio(void) { return scale_ratio; };

	float getm(void) { return m; };

	int getFrameNumber(void) { 
		if (data_type == SPARSE)  return int(sparse_frames.size());
		else if (data_type == DENSE) return int(dense_frames.size());
		else return int(sketched_frames.size()); 
	};

	size_t getTotalActive(int fr = 0) { if (data_type == SPARSE) return sparse_frames[fr].data().size(); else return size_t(T); };

	float *  getImpulsePtr(void) { return &impulse_response[0]; };

	float *  getSketchedIrfPtr(void) { return &sketched_irf[0]; };

	float *  getLogImpulsePtr(void) { return &log_impulse_response[0]; };

	float *  getIntImpulsePtr(void) { return &integrated_impulse_response[0]; };

	float *  getCircMeanPtr(void) { return &irf_circ_mean[0]; };

	photon *  getAllBinsCounts(int fr = 0) { if (data_type == SPARSE) return &(sparse_frames[fr].data()[0]); else return NULL; };


	int *  getDense(int fr = 0) {if (data_type == DENSE)  return  &(dense_frames[fr].data()[0]); else return NULL;};

	float *  getSketched(int fr = 0) { if (data_type == SKETCHED)  return  &(sketched_frames[fr].data()[0]); else return NULL; };

	int  *  getPerPixActiveBins(int fr = 0) { if (data_type == SPARSE) return &(sparse_frames[fr].bins_act()[0]); else return NULL; };

	int  *  getPerPixActiveBinsIdx(int fr = 0) {if (data_type == SPARSE) return &(sparse_frames[fr].bins_act_idx()[0]); else return NULL; };

	float * getDerImpulsePtr(void) { return &der_impulse_response[0]; };

	float getSumImpulse(void) { return sumH; };

	float getMeanPPP(int fr = 0) { 
		if(data_type==SPARSE) return sparse_frames[fr].getPPP(); 
		else if (data_type == DENSE) return dense_frames[fr].getPPP();
		else return sketched_frames[fr].getPPP();
	};

	bool dataAvailable(void) { return available_data; };

	int MultipleIrf(void) { return many_irf; };

	type getDataType(void) { return data_type; };

	float * getGain(void) { return &detector_gain[0]; };

	int getAttack(void) {
		float max = -999; int ind = 0;
		for (int i = 0; i < impulse_len; i++)
			if (impulse_response[i] > max) {
				max = impulse_response[i];
				ind = i;
			}
		return ind;
	}

	int getL(void) { return wavelenghts; };

	float getSpectralNorm(void) { return spectral_norm; };


	float * getIrfNorm(void) { return &irf_norm[0]; };

	int * getCodedAperture(void) { return &coded_aperture[0]; };

	std::string & getFilename(void) { return filename; };
	std::string & getFullFilename(void) { return full_filename; };
	int CreateDataset(lidar_file file);


	bool isDense(void) { if (data_type == DENSE) return true; else return false; };

	float  getMeanGain(void) { return mean_gain; };
	float getSigma(void);
	bool SBR_available(void) { return sbr_available; };

	float getSBR(int frame) { return SBR[frame]; };

private:
	// this function allocates the memory for a frame or video of lidar data
	int ReadLidarBinFile(lidar_file file);

	void readManyIrf(std::ifstream & input);
	file_list LidarData::getFileList(file_types);
	int ReadLidarOwnBinFile(std::string filename);
	int ReadLidarHeriotWattBinFile(std::string filename);
	bool ReadCodedAperture(std::string filename);
	void LidarData::sketchIrf(void);
	void readSingleIRF(std::ifstream & input);
	std::string filename, full_filename;
	type data_type;
	bool sbr_available;
	std::vector<SparseLidarFrame> sparse_frames;
	std::vector<DenseLidarFrame> dense_frames;
	std::vector<SketchedLidarFrame> sketched_frames;
	bool available_data;
	std::vector<float> impulse_response, irf_circ_mean, sketched_irf, SBR, log_impulse_response, der_impulse_response, detector_gain, integrated_impulse_response, irf_norm;
	std::vector<int> coded_aperture;
	int impulse_len, Nrow, Ncol, T, m,  max_active_bins_per_pix, many_irf, wavelenghts;
	float scale_ratio, sumH, mean_gain, spectral_norm;
};
