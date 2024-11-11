#include "read_lidar.h"
#include "performance.h"
#include <algorithm>  // std::min_element, std::max_element
#include <numeric>
#include <boost/filesystem.hpp>
#include <random>
#include "misc.h"


LidarData::file_list LidarData::getFileList(file_types type)
{
	file_list m_file_list;
	namespace fs = boost::filesystem;

	fs::path apk_path = boost::filesystem::current_path();
	fs::recursive_directory_iterator end;

	for (fs::recursive_directory_iterator i(apk_path); i != end; ++i) {

		const fs::path cp = (*i);

		for (int i = 0; i < type.size(); i++) {
			if (cp.string().find(type[i].first) != std::string::npos)
				m_file_list.push_back(std::make_pair(cp.string(), type[i].second));
		}
	}

	return m_file_list;
}

float LidarData::getSigma(void) {


	if (many_irf) {

		int counter = 0;
		float max = -100;
		for (int pix = 0; pix < Nrow*Ncol; pix++) {
			int corr = pix * impulse_len;
			for (int i = 0; i < impulse_len; i++) {
				if (impulse_response[i + corr] > max)
					max = impulse_response[i + corr];
			}

			float thres = max * 0.6;
			for (int i = 0; i < impulse_len; i++) {
				if (impulse_response[i + corr] > thres)
					counter++;
			}
		}

		return float(counter)/float(Nrow*Ncol);
	}
	else {
		float sigma = 0;
		for (int l = 0; l < wavelenghts; l++) {
			float max = -100;
			int offset = l * impulse_len;
			for (int i = 0; i < impulse_len; i++) {
				if (impulse_response[i+ offset] > max)
					max = impulse_response[i+ offset];
			}

			float thres = max * 0.6;
			int counter = 0;
			for (int i = 0; i < impulse_len; i++) {
				if (impulse_response[i+ offset] > thres)
					counter++;
			}
			sigma += float(counter / 2.);
		}
		return sigma/wavelenghts;
	}

}


bool LidarData::LoadDataset(bool print_info) {


	file_types ext(3);
	ext[0].first = ".rbin";
	ext[0].second = MATLAB_RASTER_SCAN;
	ext[1].first = ".bin";
	ext[1].second = HW_ARRAY;
	ext[2].first = ".gth";
	ext[2].second = SYNTHETIC;
	LidarData::file_list file_list = getFileList(ext);

	bool ret = true;

	if (file_list.empty()) {
		
		std::cout << "There are no available datasets in the app path." << std::endl;
		std::cout << "App path: " << boost::filesystem::current_path().string() << std::endl;
		ret = false;
	}

	if (ret) {
		std::cout << "Available datasets: " << std::endl;
		int file_k = 1;
		for (file_list::iterator it = file_list.begin(); it != file_list.end(); ++it, file_k++) {

			std::cout << file_k << ": " << it->first << std::endl;
		}

		while (!dataAvailable()) {
			//std::cout << "File description: .rbin: MATLAB datasets || .bin: Princeton Lightwave datasets || .gth: Ground truth information to generate synthetic data" << std::endl;
			std::cout << "File description: .rbin: MATLAB datasets || .bin: Princeton Lightwave datasets " << std::endl;
			int dataset = ask_for_param("Choose dataset number ", 1, file_list.size(), 1);
			dataset--;
			// read Lidar binary file
			if (file_list[dataset].second == SYNTHETIC)
				ret = !CreateDataset(file_list[dataset]);
			else
				ret = !ReadLidarBinFile(file_list[dataset]);
		}

		// print impulse response attack and decay
		if (ret && print_info) {
			std::cout << "Lidar cube size:" << std::endl;
			std::cout << getNrow() << "x" << getNcol() << "x" << getHistLen() << std::endl;
			std::cout << "Impulse Response information:" << std::endl;
			std::cout << "Length " << getImpulseLen() << std::endl;
		}
	}

	return ret;
}

bool LidarData::LoadDataset(int id, bool print_info) {


	file_types ext(1);
	ext[0].first = ".rbin";
	ext[0].second = MATLAB_RASTER_SCAN;
	LidarData::file_list file_list = getFileList(ext);

	bool ret = true;

	if (file_list.empty()) {

		std::cout << "There are no available datasets in the app path." << std::endl;
		std::cout << "App path: " << boost::filesystem::current_path().string() << std::endl;
		ret = false;
	}

	if (ret) {

		while (!dataAvailable()) {
			// read Lidar binary file
			ret = !ReadLidarBinFile(file_list[id]);
		}

		// print impulse response attack and decay
		if (ret && print_info) {
			std::cout << "Lidar cube size:" << std::endl;
			std::cout << getNrow() << "x" << getNcol() << "x" << getHistLen() << std::endl;
			std::cout << "Impulse Response information:" << std::endl;
			std::cout << "Length " << getImpulseLen() << std::endl;
		}
	}

	return ret;
}


bool LidarData::ReadCodedAperture(std::string filename) {


	std::ifstream input(filename, std::ios::binary);
	
	if (input.is_open()) {
		uint16_t d;
		coded_aperture.resize(Nrow*Ncol);

		for (int pix = 0; pix < Nrow*Ncol; pix++) {
			//attack
			input.read((char *)&d, sizeof(uint16_t));

			coded_aperture[pix] = d;
		}

		return true;
	}
	else
		return false;


}

// Function to read a Lidar array of 32x32
int LidarData::ReadLidarHeriotWattBinFile(std::string filename) {

	scale_ratio = 1.7; // 0.1

	//scale_ratio = float(1.5);

	wavelenghts = 1;
	Nrow = 32;
	Ncol = 32;
	T = 153; // in indoor measurements was 57

	float binary_fr_freq = 150.4e3;


	if (askYesNo("Sketch the synthetic datasets?")) {
		m = ask_for_param("How many sketches per pixel?", 1, MAX_M / 2, 5);
		data_type = SKETCHED;
		std::cout << "Lidar dataset is sketched" << std::endl;
	}
	else {
		data_type = DENSE;
		std::cout << "Lidar dataset is dense" << std::endl;
	}

	int fps = ask_for_param("How many frames per second do you want? ", 1, 1000, 50);


	float secs = ask_for_paramf("How many seconds do you want to process? ", 1./float(fps), 1e5, 6);

	int total_frames = round(secs*fps);

	int n_frames = round(binary_fr_freq/fps);

	{
		std::ifstream input(filename, std::ios::binary);


		int bin;
		int min_bin = 4; // in indoor was 4

		if (input.is_open()) {

			uint16_t d;
			int k = 0;
			while (!input.eof() && k < total_frames) {
				int tot_counts = 0;
				if (data_type == DENSE)
					dense_frames.push_back(DenseLidarFrame(Nrow, Ncol, T));
				else if (data_type == SKETCHED)
					sketched_frames.push_back(SketchedLidarFrame(Nrow, Ncol, m));

				for (int s = 0; s < n_frames; s++) {
					for (int j = 0; j < Ncol; j++) {
						// read dummy values
						input.read((char *)&d, sizeof(uint16_t));
					}

					for (int i = 0; i < Nrow; i++) {
						for (int j = 0; j < Ncol; j++) {
							//pixel wise
							input.read((char *)&d, sizeof(uint16_t));
							if (d > 0) {
								bin = int(d) - min_bin;
								if (bin < 0 || bin>=T) {
									std::cout << "Error: bin out-of-bounds" << std::endl;
									return -1;
								}

								if (data_type == DENSE) {
									dense_frames[k].inc(i, j, bin, Nrow, Ncol, T);
								}
								else if (data_type == SKETCHED) {
									for (int r = 0; r < m; r++) {
										float arg = float(PII * (r + 1)*bin) / float(T);
										sketched_frames[k].add(std::cos(arg), i, j, r, false, Nrow, Ncol, m);
										sketched_frames[k].add(std::sin(arg), i, j, r, true, Nrow, Ncol, m);
									}
								}
								tot_counts++;
							}
						}
					}
				}

				if (data_type == DENSE) {
					dense_frames[k].setPPP(float(tot_counts) / (Nrow*Ncol));
					std::cout << "Mean photons per pixel: " << dense_frames[k].getPPP() << std::endl;
				}
				else if (data_type == SKETCHED) {
					sketched_frames[k].setPPP(float(tot_counts) / (Nrow*Ncol));
					std::cout << "Mean photons per pixel: " << sketched_frames[k].getPPP() << std::endl;
				}


				/*if (type == SPARSE) {
					SparseLidarFrame sparse_fr;
					// save values in list form
					for (int i = 0; i < Nrow; i++) {
						for (int j = 0; j < Ncol; j++) {
							int cs = 0;
							for (int t = 0; t < T; t++) {
								if (dense_frames[k].read(i, j, t, Nrow, Ncol, T) > 0) {
									photon p;
									p.bin = t;
									p.counts = dense_frames[k].read(i, j, t, Nrow, Ncol, T);
									sparse_fr.data().push_back(p);
									cs++;
								}
							}
							sparse_fr.bins_act().push_back(cs);
						}
					}

					sparse_fr.bins_act_idx().resize(sparse_fr.bins_act().size() + 1);
					std::partial_sum(sparse_fr.bins_act().begin(), sparse_fr.bins_act().end(), sparse_fr.bins_act_idx().begin() + 1);
					sparse_fr.bins_act_idx().pop_back();
					sparse_fr.setPPP(dense_frames[k].getPPP());

					int max_frame = *std::max_element(sparse_fr.bins_act().begin(), sparse_fr.bins_act().end());
					if (max_frame > max_active_bins_per_pix)
						max_active_bins_per_pix = max_frame;

					sparse_frames.push_back(sparse_fr);

				}
				*/


				k++;
			
			
			}

			input.close();
		}
		else {
			std::cout << "ERR: Could not open file " << filename;
			return -1;
		}
	}

	{
		// read impulse response
		std::ifstream input("datasets/IRF.irf", std::ios::binary);

		if (input.is_open()) {
			
			readManyIrf(input);
			input.close();
			if (data_type == SKETCHED)
				sketchIrf();

		}
		else {
			std::cout << "ERR: Could not open file " << filename;
			return -1;
		}
	}



	available_data = true;
	return 0;
}


void LidarData::sketchIrf(void) {
	
	if (many_irf) {
		int mem_size = 2*m * Nrow * Ncol;

		spectral_norm = 0;
		sketched_irf.resize(mem_size);
		for (int pix = 0; pix < Nrow*Ncol; pix++) {

			int corr = pix * impulse_len;
			int corr_s = pix * 2 * m;
			for (int i = 0; i < m; i++) {

				float real = 0., imag = 0.;
				for (int t = 0; t < impulse_len; t++) {
					float arg = float(PII * (i + 1) * t) / float(T);
					real += std::cos(arg)*impulse_response[corr + t];
					imag += std::sin(arg)*impulse_response[corr + t];
				}
				sketched_irf[corr_s + i] = real;
				sketched_irf[corr_s + i + m] = imag;

				if (pix == 0)
					spectral_norm += real * real + imag * imag;

				if (i == 0) {
					float aux = 0.;
					aux = std::atan2(imag, real)*float(T) / PII;
					if (aux < 0)
						aux += T;

					irf_circ_mean.push_back(aux);

				}
			}

			
		}
	}
	else {
		int mem_size = 2 * m;
		sketched_irf.resize(mem_size);
		spectral_norm = 0;
		for (int i = 0; i < m; i++) {

			float real = 0., imag = 0.;
			
			for (int t = 0; t < impulse_len; t++) {
				float arg = float(PII * (i + 1) * t) / float(T);
				real += std::cos(arg)*impulse_response[t];
				imag += std::sin(arg)*impulse_response[t];
			}
			sketched_irf[i] = real;
			sketched_irf[i + m] = imag;
			spectral_norm += real * real + imag * imag;
			if (i == 0) {
				float aux = 0.;
				aux = std::atan2(imag, real)*float(T) / PII;
				if (aux < 0)
					aux += T;

				irf_circ_mean.push_back(aux);

			}
		}

	}

	//spectral_norm = sqrt(spectral_norm);

}

void LidarData::readManyIrf(std::ifstream & input) {

	uint16_t d;
	//attack
	input.read((char *)&d, sizeof(uint16_t));
	impulse_len = d;
	int mem_size = impulse_len * Nrow * Ncol;

	many_irf = 1;
	// impulse response
	float maxH = 0;
	impulse_response.resize(mem_size);
	log_impulse_response.resize(mem_size);
	der_impulse_response.resize(mem_size);
	integrated_impulse_response.resize(mem_size);
	detector_gain.resize(Nrow * Ncol);
	irf_norm.resize(Nrow * Ncol);

	float b = 1 / T;
	for (int pix = 0; pix < Nrow*Ncol; pix++) {
		int corr = pix * impulse_len;
		for (int i = 0; i < impulse_len; i++) {
			input.read((char *)&d, sizeof(uint16_t));
			if (d == 0)
				d = 1;
			impulse_response[i + corr] = float(d);
			float log_f = std::log(float(d)+b);
			log_impulse_response[i + corr] = log_f;
		}

		sumH = 0; 
		for (int i = 0; i < impulse_len; i++) {
			sumH += impulse_response[i + corr];
		}


		detector_gain[pix] = sumH;
		if (sumH > maxH)
			maxH = sumH;

		float norm = 0;
		impulse_response[corr] /= sumH;
		for (int i = 1; i < impulse_len; i++) {
			impulse_response[i + corr] /= sumH;
			norm += impulse_response[i + corr] * impulse_response[i + corr];
			der_impulse_response[i - 1 + corr] = impulse_response[i + corr] - impulse_response[i - 1 + corr];
		}
		der_impulse_response[impulse_len - 1 + corr] = 0;

		irf_norm[pix] = norm;

		sumH = 1;
		float acc = float(sumH);
		for (int i = 0; i < impulse_len; i++) {
			acc -= impulse_response[corr + impulse_len - 1 - i];
			integrated_impulse_response[corr + i] = acc;
		}

	}

	for (int pix = 0; pix < Nrow*Ncol; pix++)
		detector_gain[pix] /= maxH;

	mean_gain = 0;
	for (int pix = 0; pix < Nrow*Ncol; pix++)
		mean_gain += detector_gain[pix];
	mean_gain /= (Nrow*Ncol);
}

std::string getFileName(std::string filePath, bool withExtension = false)
{
	namespace fs = boost::filesystem;
	// Create a Path object from File Path
	fs::path pathObj(filePath);

	// Check if file name is required without extension
	if (withExtension == false)
	{
		// Check if file has stem i.e. filename without extension
		if (pathObj.has_stem())
		{
			// return the stem (file name without extension) from path object
			return pathObj.stem().string();
		}
		return "";
	}
	else
	{
		// return the file name with extension from path object
		return pathObj.filename().string();
	}
	
}

int LidarData::ReadLidarBinFile(lidar_file file) {


	full_filename = file.first;
	filename = getFileName(file.first);

	std::cout << " file: " << filename << std::endl;

	switch (file.second) {
		case HW_ARRAY:
			//std::cout << "Do you want to read it in a dense way?";
			return ReadLidarHeriotWattBinFile(file.first);
			break;
		case MATLAB_RASTER_SCAN:
			return ReadLidarOwnBinFile(file.first);
			break;
		default:
			return -1;
	}


}


void LidarData::readSingleIRF(std::ifstream & input) {

	uint16_t d;
	input.read((char *)&d, sizeof(uint16_t));
	impulse_len = d;
	many_irf = 0;

	impulse_response.resize(impulse_len*wavelenghts);
	log_impulse_response.resize(impulse_len*wavelenghts);
	der_impulse_response.resize(impulse_len*wavelenghts);
	integrated_impulse_response.resize(impulse_len*wavelenghts);
	// impulse response
	for (int l = 0; l < wavelenghts; l++) {
		float f;

		int idx_offset = l * impulse_len;

		for (int i = 0; i < impulse_len; i++) {
			input.read((char *)&f, sizeof(float));
			impulse_response[i+ idx_offset] = f;
		}



		sumH = 0;
		for (int i = 0; i < impulse_len; i++) {
			sumH += impulse_response[i + idx_offset];
		}

		impulse_response[idx_offset] /= sumH;
		for (int i = 1; i < impulse_len; i++) {
			impulse_response[i + idx_offset] /= sumH;
			der_impulse_response[i + idx_offset - 1] = impulse_response[i + idx_offset] - impulse_response[i + idx_offset - 1];
		}
		der_impulse_response[impulse_len - 1 + idx_offset] = 0;


		int  off = 0;
		for (int k = 1; k < impulse_len; k *= 2) {
			for (int i = 0; i < k; i++) {
				float accum = 0;
				int binning = ceil(impulse_len / k);
				for (int j = i * binning; j < (i + 1)*binning; j++) {
					if (j >= impulse_len)
						break;
					accum += impulse_response[j + idx_offset];
				}
				if (off + i >= impulse_len)
					break;

				log_impulse_response[off + i + idx_offset] = accum;
			}
			off += k;
		}


		float norm = 0;
		for (int i = 0; i < impulse_len; i++) {
			norm = impulse_response[i + idx_offset] * impulse_response[i + idx_offset];
		}


		sumH = 1;
		float acc = float(sumH);
		for (int i = 0; i < impulse_len; i++) {
			acc -= impulse_response[impulse_len - 1 - i + idx_offset];
			integrated_impulse_response[i + idx_offset] = acc;
		}

	}

	detector_gain.resize(Nrow*Ncol);
	irf_norm.resize(Nrow*Ncol);
	for (int pix = 0; pix < Nrow*Ncol; pix++) {
		detector_gain[pix] = 1.;
		irf_norm[pix] = 1; // TODO: put correct value
	}
	mean_gain = 1;
}



int LidarData::CreateDataset(lidar_file file) {

	full_filename = file.first;
	filename = getFileName(file.first);

	std::cout << " file: " << filename << std::endl;

	float min_sbr = ask_for_paramf("Input minimum SBR", 0.001, 1000, 1.);
	float max_sbr = ask_for_paramf("Input maximum SBR", min_sbr, 1000, min_sbr);
	int step_size_sbr = ask_for_param("Input SBR interval size", 1, 100, 1);
	float min_phot = ask_for_paramf("Input minimum mean photons per pixel", 0.1, 1000, 1);
	float max_phot = ask_for_paramf("Input maximum mean photons per pixel", min_phot, 1000, min_phot);
	int step_size_phot = ask_for_param("Input photon interval size", 1, 100, 1);

	std::vector<float> mean_photons = logspace(min_phot, max_phot, step_size_phot);
	std::vector<float> sbr = logspace(min_sbr, max_sbr, step_size_sbr);
	sbr_available = true;


	wavelenghts = 1;
	// create dataset from ground truth

	Performance reference;
	reference.load_ground_truth(file.first);
	Nrow = reference.getNrow();
	Ncol = reference.getNcol();
	T = reference.getT();
	scale_ratio = reference.getScaleRatio();

	// generate counts
	int n_frames = mean_photons.size()*sbr.size();

	if (n_frames>0)
		available_data = true;

	// TODO: improve this
	std::ifstream input("synthetic.irf", std::ios::binary);
	if (input.is_open()) {
		readSingleIRF(input);
		input.close();
	}
	else {
		std::cout << "ERROR: no synthetic.irf file was found. Cannot generate data without impulse response" << std::endl;
		return -1;
	}

	int attack = getAttack();

	std::default_random_engine generator;

	
	if (askYesNo("Sketch the synthetic datasets?")) {
		m = ask_for_param("How many sketches per pixel?", 1, MAX_M/2, 5);
		data_type = SKETCHED;
		sketchIrf();
		std::cout << "Lidar dataset is sketched" << std::endl;
	} else {
		data_type = SPARSE;
		sparse_frames.resize(n_frames);
		std::cout << "Lidar dataset is sparse" << std::endl;
	}

	SBR.clear();
	//photons
	int fr = 0;
	for (int sbr_ind = 0; sbr_ind < sbr.size(); sbr_ind++) {
		for (int ppp_ind = 0; ppp_ind < mean_photons.size(); ppp_ind++) {

			SBR.push_back(sbr[sbr_ind]);
			std::cout << "Generating synthetic frame " << fr + 1 << " out of " << n_frames << std::endl;
			int tot_counts = 0;
			int k = 0;
			
			if (data_type == SPARSE)
				sparse_frames[fr].bins_act().resize(Nrow*Ncol);
			else if (data_type == SKETCHED)
				sketched_frames.push_back(SketchedLidarFrame(Nrow, Ncol, m));

			float bkg_corr = mean_photons[ppp_ind] / reference.getPPP()*(1.+reference.getSBR())/(1.+ sbr[sbr_ind]);

			float ref_corr = sbr[sbr_ind] / reference.getSBR()*bkg_corr;

			//std::cout << "ref corr: " << ref_corr << std::endl;
			//std::cout << "bkg corr: " << bkg_corr << std::endl;

			for (int pix = 0; pix < Nrow*Ncol; pix++) {
				std::vector<float> x(T); // intensity

				float bkg = reference.read_bkg(pix)*bkg_corr;

				// add background
				for (int t = 0; t < T; t++) {
					x[t] = bkg;
				}

				// add signal 
				for (int j = 0; j < reference.frames[0].points_per_pix[pix]; j++) {
					int depth = int(reference.frames[0].points[k]);
					float ref = reference.frames[0].reflect[k]*ref_corr;
					int init_t = 0 > (depth - attack) ? 0 : (depth - attack);
					int end_t = (depth - attack + impulse_len) > T ? T : (depth - attack + impulse_len);
					for (int t = init_t, r = 0; t < end_t; t++, r++) {
						x[t] += ref * impulse_response[r];	
					}
					k++;
				}


				// simulate photons and save
				int cs = 0;
				for (int t = 0; t < T; t++) {
					std::poisson_distribution<int> distribution(x[t]);
					int counts = distribution(generator);
					if (counts > 0) {
						photon phot;
						cs++;
						phot.bin = t;
						phot.counts = counts;
						tot_counts += phot.counts;

						if (data_type == SPARSE)
							sparse_frames[fr].data().push_back(phot);
						else if (data_type == SKETCHED) {

							int j = pix / Nrow;
							int i = pix - j * Nrow;

							for (int r = 0; r < m; r++) {
								float arg = float(PII * (r + 1)*phot.bin) / float(T);
								sketched_frames[fr].add(phot.counts * std::cos(arg), i, j, r, false, Nrow, Ncol, m);
								sketched_frames[fr].add(phot.counts * std::sin(arg), i, j, r, true, Nrow, Ncol, m);
							}
						}
						
					}
				}
				if (data_type == SPARSE)
					sparse_frames[fr].bins_act()[pix] = cs;
			}

			if (data_type == SPARSE) {
				sparse_frames[fr].bins_act_idx().resize(Nrow*Ncol + 1);
				std::partial_sum(sparse_frames[fr].bins_act().begin(), sparse_frames[fr].bins_act().end(), sparse_frames[fr].bins_act_idx().begin() + 1);
				sparse_frames[fr].bins_act_idx().pop_back();

				max_active_bins_per_pix = *std::max_element(sparse_frames[fr].bins_act().begin(), sparse_frames[fr].bins_act().end());

				sparse_frames[fr].setPPP(float(tot_counts) / (Nrow*Ncol));
				std::cout << "Mean photons per pixel: " << sparse_frames[fr].getPPP() << std::endl;
				std::cout << "Max active bins per pixel: " << max_active_bins_per_pix << std::endl;

				if (max_active_bins_per_pix > MAX_ACTIVE_BINS_PER_PIXEL) {
					std::cout << "ERROR: too many bins with photons per pixel (change to sketched representation)" << std::endl;
					return -1;
				}
			} else if (data_type == SKETCHED){
				sketched_frames[fr].setPPP(float(tot_counts) / (Nrow*Ncol));
				std::cout << "Mean photons per pixel: " << sketched_frames[fr].getPPP() << std::endl;
			}

			fr++;
		}
	}

	return 0;
}



int LidarData::ReadLidarOwnBinFile(std::string filename) {

	std::ifstream input(filename, std::ios::binary);

	if (input.is_open()) {
		uint16_t d;
		input.read((char *)&d, sizeof(uint16_t));
		if (d == 0) { // multispectral (UGLY code for backwards compatibility)
			input.read((char *)&d, sizeof(uint16_t));
			wavelenghts = d;

			if (d > MAX_WAVELENGTHS) {
				std::cout << "ERR: The dataset contains more wavelengths than allowed" << std::endl;
				return -1;
			}
			input.read((char *)&d, sizeof(uint16_t));
		}
		else { // single-wavelength
			wavelenghts = 1;
		}

		// Nrow
		Nrow = d;

		// Ncol
		input.read((char *)&d, sizeof(uint16_t));
		Ncol = d;

		// T - histogram_length
		input.read((char *)&d, sizeof(uint16_t));
		T = d ; // -268 ONLY A QUICK FIX FOR THE IBEO DATA

		// scale ratio
		float f;
		input.read((char *)&f, sizeof(float));
		scale_ratio = f;

		input.read((char *)&d, sizeof(uint16_t));
		if (d > 0) 
			readManyIrf(input);
		else 
			readSingleIRF(input);


		// frame number
		input.read((char *)&d, sizeof(uint16_t));
		int n_frames = d;

		// sparse or dense

		std::streampos previous_pos = input.tellg();
		int mean_act_bins = 0;
		int max_act_bins = 0;
		for (int pix = 0; pix < Nrow*Ncol; pix++) {
			int cs = 0;
			input.read((char *)&d, sizeof(uint16_t));
			uint16_t next_photon = d;
			while (next_photon != 0xFFFF) {
				photon phot;
				cs++;
				phot.bin = next_photon;
				input.read((char *)&d, sizeof(uint16_t));
				phot.counts = d;

				input.read((char *)&d, sizeof(uint16_t));
				next_photon = d;
				if (input.eof() && pix < Nrow*Ncol - 1) {
					std::cout << "ERROR: Corrupted data" << std::endl;
					return -1;
				}
			}
			mean_act_bins += cs;
			max_act_bins = cs>max_act_bins? cs : max_act_bins;
		}
		mean_act_bins /= Nrow;
		mean_act_bins /= Ncol;

		// ask for sketched dataset

		if (m>0 || askYesNo("Do you want to sketch the dataset?")) {
			if (m==0)
				m = ask_for_param("How many sketches per pixel?", 1, MAX_M/2, 5);
			data_type = SKETCHED;
			sketchIrf();
			std::cout << "Lidar dataset is sketched" << std::endl;
		}
		else { 
			if (mean_act_bins < MAX_ACTIVE_BINS_SPARSE && max_act_bins<MAX_ACTIVE_BINS_PER_PIXEL) {
				data_type = SPARSE;
				std::cout << "Lidar dataset is sparse" << std::endl;
				sparse_frames.resize(n_frames);
			}
			else {
					if (T < MAX_DENSE_BINS_PER_PIXEL) {
						data_type = DENSE;
						std::cout << "Lidar dataset is dense" << std::endl;
					}
					else {
						std::cout << "ERROR: Too many active bins per pixel, consider sketching" << std::endl;
						return -1;
					}
				}
		}
		input.seekg(previous_pos);



		//photons
		for (int fr = 0; fr < n_frames; fr++) {
			int tot_counts = 0;
			int max_counts = 0;

			if (data_type == SPARSE)
				sparse_frames[fr].bins_act().resize(Nrow*Ncol);
			else if (data_type == DENSE)
				dense_frames.push_back(DenseLidarFrame(Nrow, Ncol, T));
			else
				sketched_frames.push_back(SketchedLidarFrame(Nrow, Ncol, m));

			for (int pix = 0; pix < Nrow*Ncol; pix++) {
				input.read((char *)&d, sizeof(uint16_t));	
				uint16_t next_photon = d;
				int cs = 0;
				while(next_photon != 0xFFFF){
					photon phot;
					cs++;
					phot.bin = next_photon-1;
					input.read((char *)&d, sizeof(uint16_t));
					phot.counts = d;
					tot_counts += phot.counts;

					if (data_type == SPARSE)
						sparse_frames[fr].data().push_back(phot);
					else if (data_type == DENSE) {
						int j = pix / Nrow;
						int i = pix-j*Nrow;
						dense_frames[fr].set(phot.counts, i, j, phot.bin, Nrow, Ncol, T);
					}
					else {

						int j = pix / Nrow;
						int i = pix - j * Nrow;

						for (int r = 0; r < m; r++) {
							float arg = float(PII * (r+1)*phot.bin) / float(T);
							//std::cout << "cos " << std::cos(arg) << std::endl;
							sketched_frames[fr].add(phot.counts * std::cos(arg), i, j, r, false, Nrow, Ncol, m);
							sketched_frames[fr].add(phot.counts * std::sin(arg), i, j, r, true, Nrow, Ncol, m);
						}
					}

					input.read((char *)&d, sizeof(uint16_t));
					next_photon = d;
					if (input.eof() && pix<Nrow*Ncol-1) {
						std::cout << "ERROR: Corrupted data" << std::endl;
						return -1;
					}
				}

				if (data_type == SPARSE)
					sparse_frames[fr].bins_act()[pix] = cs;
			}
			
			if (data_type == SPARSE) {
				sparse_frames[fr].bins_act_idx().resize(Nrow*Ncol + 1);
				std::partial_sum(sparse_frames[fr].bins_act().begin(), sparse_frames[fr].bins_act().end(), sparse_frames[fr].bins_act_idx().begin() + 1);
				sparse_frames[fr].bins_act_idx().pop_back();


				max_active_bins_per_pix = *std::max_element(sparse_frames[fr].bins_act().begin(), sparse_frames[fr].bins_act().end());

				sparse_frames[fr].setPPP(float(tot_counts) / (Nrow*Ncol));
				std::cout << "Mean photons per pixel: " << sparse_frames[fr].getPPP() << std::endl;
				std::cout << "Max active bins per pixel: " << max_active_bins_per_pix << std::endl;
			}
			else if (data_type == DENSE) {
				dense_frames[fr].setPPP(float(tot_counts) / (Nrow*Ncol));
				std::cout << "Mean photons per pixel: " << dense_frames[fr].getPPP() << std::endl;
			}
			else {
				sketched_frames[fr].setPPP(float(tot_counts) / (Nrow*Ncol));
				std::cout << "Mean photons per pixel: " << sketched_frames[fr].getPPP() << std::endl;
			}

		}
		input.close();


		bool ap_ok = true;
		if (wavelenghts>1) {
			ap_ok = ReadCodedAperture("coded_aperture.cda");
		}
		
		if (ap_ok) {
			available_data = true;
		}
		else {
			std::cout << "ERR: WRONG CODED APERTURE FILE " << filename << std::endl;
		}

	}
	else {
		std::cout << "ERR: Could not open file " << filename << std::endl; 
		return -1;
	}

	return 0;
}
