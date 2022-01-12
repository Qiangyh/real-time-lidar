#pragma once

#include <vector>


class GroundTruth {
public:

	GroundTruth(void) {};
	
	GroundTruth(int Nrow, int Ncol) {
		background.resize(Nrow*Ncol);
		points_per_pix.resize(Nrow*Ncol);
		total_points = 0;
	};

	std::vector<float> points, reflect;
	std::vector<int> points_per_pix;
	std::vector<float> background;

	float bin_width, sbr, ppp, scale_ratio;
	int	total_points;
};


class Performance {
public:

	Performance(void) {
		frames.clear();
	};

	void load_ground_truth(std::string filename, bool verbose = false);

	// input maximum distance in mm
	void compute_error(std::string & filename, int fr, std::vector<float> & points, std::vector<float> & reflect, std::vector<int> & in_points_per_pix,
		std::vector<float> & background, float time, float distance = 40, float hyperparam = 0);


	void save_timing(std::string & filename,  float time);


	void push_back_copy(int rep = 1, int fr = 0);

	bool ground_truth_available(int fr = 0) { 
		if (frames.empty()) return false;
		else return(fr <= frames.size()); 
	};

	void modify_scale(float sbr, float ppp, int fr = 0);

	float read_bkg(int pixel, int fr = 0) {
		return frames[fr].background[pixel];
	};

	int getNrow() {
		return Nrow;
	};

	int getNcol() {
		return Ncol;
	};

	int getT() {
		return T;
	};

	int getScaleRatio() {
		return scale_ratio;
	};

	float getSBR(int fr = 0) {
		return frames[fr].sbr;
	};

	float getPPP(int fr = 0) {
		return frames[fr].ppp;
	};

	std::vector<GroundTruth> frames;

private:

	int Nrow, Ncol, T;
	float bin_width, scale_ratio;
	int total_points;
};