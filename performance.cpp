#include "performance.h"
#include <iostream>
#include <fstream>
#include "Point.h"


void Performance::load_ground_truth(std::string filename, bool verbose) {

	size_t lastindex = filename.find_last_of(".");
	std::string rawname = filename.substr(0, lastindex);
	rawname += std::string(".gth");

	//std::cout << "filename is: "  << rawname << std::endl;
	std::ifstream input(rawname, std::ios::binary);

	if (input.is_open()) {

		uint16_t d;
		float f;
		// read Nrow
		input.read((char *)&d, sizeof(uint16_t));
		Nrow = d;
		// read Ncol
		input.read((char *)&d, sizeof(uint16_t));
		Ncol = d;

		// read T
		input.read((char *)&d, sizeof(uint16_t));
		T = d;

		// read bin width
		input.read((char *)&f, sizeof(float));
		bin_width = f;
		// read scale ratio
		input.read((char *)&f, sizeof(float));
		scale_ratio = f;

		// read frames
		input.read((char *)&d, sizeof(uint16_t));
		frames.resize(d);



		float ppp = 0, sbr = 0, mean_ref = 0;
		for (int fr = 0; fr < frames.size(); fr++) {
			GroundTruth gt(Nrow, Ncol);
			for (int j = 0; j < Ncol; j++) {
				for (int i = 0; i < Nrow; i++) {
					uint16_t d;
					// read points per pix
					input.read((char *)&d, sizeof(uint16_t));
					int lin_index = i + j * Nrow;
					gt.points_per_pix[lin_index] = d;
					for (int n = 0; n < d; n++) {
						// read position
						input.read((char *)&f, sizeof(float));
						gt.points.push_back(f);
						// read reflect
						input.read((char *)&f, sizeof(float));
						gt.reflect.push_back(f);
						mean_ref += gt.reflect.back();
						gt.total_points++;
					}
					// read background
					input.read((char *)&f, sizeof(float));
					gt.background[lin_index] = f;
					ppp += f * T;
				}
			}

			sbr = mean_ref / ppp;
			ppp += mean_ref;
			ppp /= (Nrow*Ncol);
			gt.sbr = sbr;
			gt.ppp = ppp;

			frames[fr] = gt;

			if (verbose) {
				std::cout << "Ground truth frame loaded!" << std::endl;
				std::cout << "Mean reflectivity ground-truth: " << mean_ref / float(gt.points.size()) << std::endl;
				std::cout << "Mean photons per pixel: " << ppp << std::endl;
				std::cout << "SBR: " << sbr << std::endl << std::endl;
			}
		}

		input.close();

	}

};


void Performance::compute_error(std::string & filename, int fr, std::vector<float> & points, std::vector<float> & reflect, std::vector<int> & in_points_per_pix, std::vector<float> & background, float time, float distance, float hyperparam) {


	int false_points = 0, true_points = 0;
	float ref_error = 0, bkg_error = 0, dep_error = 0;
	float bkg_power = 0.;

	float dist = distance/bin_width; 
	int k = 0;
	for (int j = 0; j < Ncol; j++) {
		for (int i = 0; i < Nrow; i++) {


			int lin_index = i + j * Nrow;
			std::vector<float> depths, reflectivity;

			bkg_power += (frames[fr].background[lin_index] * frames[fr].background[lin_index]);
			float aux = (frames[fr].background[lin_index] - background[lin_index]);

			bkg_error += (aux*aux);

			int est_number_points = in_points_per_pix[lin_index];
			
			for (int idx = 0; idx < est_number_points; idx++) {
				depths.push_back(ReadPointDepth(i, j, idx, Nrow, Ncol, &points[0]));
				reflectivity.push_back(ReadPointRef(i, j, idx, Nrow, Ncol, &reflect[0]));
			}

			for (int n = 0; n < frames[fr].points_per_pix[lin_index]; n++) {
				bool flag = false;
				for (int r = 0; r < depths.size(); r++) {
					if (abs(depths[r] - frames[fr].points[k]) < dist) {
						true_points++;
						ref_error += abs(reflectivity[r] - frames[fr].reflect[k]);
						dep_error += abs(depths[r] - frames[fr].points[k]);
						depths.erase(depths.begin() + r);
						reflectivity.erase(reflectivity.begin() + r);
						flag = true;
						break;
					}
				}
				if (!flag)
					ref_error += frames[fr].reflect[k];
				k++;
			}
			false_points += int(depths.size());
		}
	}

	bkg_error /= bkg_power;
	ref_error /= frames[fr].total_points;
	if (true_points > 0) {
		dep_error /= true_points;
	}
	dep_error *= bin_width;
	true_points = 100.*float(true_points) / frames[fr].total_points;


	std::ofstream output(filename, std::ios_base::app);

	output << getSBR(fr) << ";" << getPPP(fr) << ";" << true_points << ";" << false_points << ";" << dep_error << ";" << ref_error << ";" << bkg_error << ";" << time << ";" << hyperparam << ";";

	std::cout << "True points: " << true_points << " %"<<std::endl;
	std::cout << "False points: " << false_points << std::endl;
	std::cout << "Ref. error: " << ref_error << std::endl;
	std::cout << "Dep. error: " << dep_error << std::endl;
	std::cout << "Background NMSE: " << bkg_error << std::endl;
};


void Performance::save_timing(std::string & filename, float time) {

	std::ofstream output(filename, std::ios_base::app);
	output << time << ";";
}

void Performance::push_back_copy(int rep, int fr) {

	if (ground_truth_available(fr)) {
		GroundTruth gt = frames[fr];

		for (int i = 0; i < rep; i++) {
			frames.push_back(gt);
		}
	}

};



void Performance::modify_scale(float sbr, float ppp, int fr) {

	if (ground_truth_available(fr)) {

		float bkg_corr = ppp/frames[fr].ppp*(1. + frames[fr].sbr) / (1. + sbr);
		float ref_corr = sbr/ frames[fr].sbr*bkg_corr;
		int  k = 0;
		for (int j = 0; j < Ncol; j++) {
			for (int i = 0; i < Nrow; i++) {
				int lin_index = i + j * Nrow;
				frames[fr].background[lin_index] *= bkg_corr;
				for (int n = 0; n < frames[fr].points_per_pix[lin_index]; n++) {
					frames[fr].reflect[k] *= ref_corr;
					k++;
				}
			}
		}

		frames[fr].ppp = ppp;
		frames[fr].sbr = sbr;
	}

};

