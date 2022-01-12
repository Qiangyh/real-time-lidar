#include <stdio.h>
#include <ctime>
#include "read_lidar.h"
#include "visualization.h"
#include "algorithm.h"
#include "performance.h"
#include "misc.h"
#include <boost/atomic.hpp>


void run_all(void){
	int id = 0;

	int m = 0;
	if (askYesNo("Sketch the datasets?")) {
		m = ask_for_param("How many sketches per pixel?", 1, MAX_M / 2, 5);
	}

	while (1) {

		/************* Load data *************/
		LidarData data(m);
		std::cout << "Loading new dataset" << std::endl;
		if (data.LoadDataset(id++)) {

			Performance performance;
			performance.load_ground_truth(data.getFullFilename());

			for (int alg = 1; alg <= 5; alg++) {
				/************* Run Algorithm *************/
				LidarReconstruction algo;

				algo.loadParameters(data, false, alg);

				/************* Show results *************/
				bool plot = false;
				bool binary = true;
				bool plot_likelihood = false;
				visualization vis(plot, plot_likelihood, algo.getAlgoName() + data.getFilename());


				std::vector<float> reflect, background, points, normals, likelihood;
				std::vector<int>  points_per_pix;
				for (int i = 0; i < data.getFrameNumber(); i++) {

					if (data.SBR_available()) {
						algo.defaultHyperParameters(data.getMeanPPP(i), data.getSBR(i));
						performance.push_back_copy();
						performance.modify_scale(data.getSBR(i), data.getMeanPPP(i), i);
					}


					if (data.getFrameNumber() == 1) { // GPU Warm UP
						for (int j = 0; j < 2; j++)
							algo.run_frame(i, false);
					}

					algo.run_frame(i, false);

					if (algo.results(points, normals, reflect, points_per_pix, background, likelihood)) {
						// compute error if ground truth is available
						if (performance.ground_truth_available(i))
							performance.compute_error(vis.getFolderName() + std::string("performance.csv"), i, points, reflect, points_per_pix, background, algo.getExecTime(), 40); //, hyperparam[l]);
						else
							performance.save_timing(vis.getFolderName() + std::string("timing.csv"), algo.getExecTime());


						// show point cloud
						if (!vis.loadPointCloud(points, normals, reflect, points_per_pix, algo.getScaleRatio(), algo.getCloudHeight(), algo.getCloudWidth(), background, data.getNrow(), data.getNcol(), data.getL(), likelihood)) {
							//vis.saveBackground(std::string("background") + std::to_string(i) , binary);
							vis.savePly(std::string("frame") + std::to_string(i));
							//vis.savePointsPerPix(std::string("points") + std::to_string(i) , binary);
						}
					}
				}
			}
		}
		else
			break;
	}
};

void run_single(void) {

	/************* Load data *************/
	LidarData data;
	if (data.LoadDataset()) {

		Performance performance;
		performance.load_ground_truth(data.getFullFilename());

		/************* Run Algorithm *************/
		LidarReconstruction algo;
			
		while (1) {
			algo.loadParameters(data, false); // !data.SBR_available());

				/************* Show results *************/
			bool plot = (data.getFrameNumber() == 1);// askYesNo("Do you want to plot results? This will significantly degrade the execution performance");
			bool binary = true;
			bool plot_likelihood = false;
			visualization vis(plot, plot_likelihood, algo.getAlgoName() + data.getFilename());


			std::vector<float> reflect, background, points, normals, likelihood;
			std::vector<int>  points_per_pix;
			for (int i = 0; i < data.getFrameNumber(); i++) {

				//std::vector<float> hyperparam = logspace(float(.01), float(100), 20);

				if (data.SBR_available()) {
					algo.defaultHyperParameters(data.getMeanPPP(i), data.getSBR(i));
					performance.push_back_copy();
					performance.modify_scale(data.getSBR(i), data.getMeanPPP(i), i);
				}

				//for (int l = 0; l < hyperparam.size(); l++) {


				if (data.getFrameNumber() == 1) { // GPU Warm UP
					for (int j = 0; j < 2; j++)
						algo.run_frame(i, false);
				}

				algo.run_frame(i, false);

				if (algo.results(points, normals, reflect, points_per_pix, background, likelihood)) {
					// compute error if ground truth is available
					if (performance.ground_truth_available(i))
						performance.compute_error(vis.getFolderName() + std::string("performance.csv"), i, points, reflect, points_per_pix, background, algo.getExecTime(), 40); //, hyperparam[l]);
					else
						performance.save_timing(vis.getFolderName() + std::string("timing.csv"), algo.getExecTime());


					// show point cloud
					if (!vis.loadPointCloud(points, normals, reflect, points_per_pix, algo.getScaleRatio(), algo.getCloudHeight(), algo.getCloudWidth(), background, data.getNrow(), data.getNcol(), data.getL(), likelihood)) {
						//vis.saveBackground(std::string("background") + std::to_string(i) , binary);
						vis.savePly(std::string("frame") + std::to_string(i));
						//vis.savePointsPerPix(std::string("points") + std::to_string(i) , binary);
						//vis.saveLikelihood(std::string("likelihood") + std::to_string(i) , likelihood);
					}
				}
			}
		}
	}
};