#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "read_lidar.h"
#include "Point.h"
#include "math_functions.h"

#define MIN_PHOTONS_PER_PEAK 3


struct pos {
	int x;
	int y;
	int idz;
	float z;
	float r;
};


// this adds points
// log matched filter kernel- 1 thread per pixel
__global__ void logMatchedFilterKernel(float *points, float * reflect, int *points_per_pix, float *background, const float * log_impulse, const float *d_integrated_impulse,
	const int *bins_counts, const int *pix_counts, const int *pix_counts_idx, const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const int downsample, const int max_points, const float max_bkg, const int many_irf, const float *d_gain) {

	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	photon data[MAX_ACTIVE_BINS_PER_PIXEL];

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);

	// for pixelwise IRF
	int idx_offset = many_irf * linear_idx *impulse_len;

	//convolve
	int act_bin = pix_counts[linear_idx]; //get total active bins
	int start_idx = pix_counts_idx[linear_idx]; //get memory


	// read global
	int tot_photons = 0;
	for (int t = 0; t < act_bin; t++) {
		data[t] = getBinCounts(x, y, t, bins_counts, start_idx, Nrow);
		tot_photons += data[t].counts;
	}
	int non_bkg_bins = 0;

	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;

	float gain = d_gain[linear_idx];

	int iter = 0;
	int refs[MAX_POINTS_PER_PIX];
	int pos[MAX_POINTS_PER_PIX];
	int points_found = 0;

	if (act_bin > 0) {
		float thres = 0; // (tot_photons / 100.);
		while ((iter++) < max_points) {
			bool flag = false;
			float max_f = -1e25;
			int  max_ref = 0;
			int bin_max = 0;
			int init_t = 0;
			for (int i = 0; i < T; i += downsample) {
				float filt = 0;
				int ref = 0;
				for (int t = init_t; t < act_bin; t++) {
					int diff = (data[t].bin - i);
					if (diff < 0) {
						init_t = t;
					}
					else if (diff < impulse_len) {
						filt += data[t].counts * log_impulse[idx_offset + diff];
						ref += data[t].counts;
					}
					else {
						break;
					}
				}

				/*
				float sumH=1;

				int extra = i + impulse_len - T;
				if (extra > 0)
					sumH = d_integrated_impulse[idx_offset+extra - 1];
				else
					sumH = 1;

				filt = filt / sumH;
				*/

				if (filt > max_f &&  ref > thres) {
					max_f = filt;
					bin_max = i;
					max_ref = ref;
					flag = true;
				}
			}
			if (flag) {

				for (int t = 0; t < act_bin; t++) {
					int diff = (data[t].bin - bin_max);
					if (diff>= 0 && diff<impulse_len) {
						tot_photons -= data[t].counts;
						data[t].counts = 0;
						non_bkg_bins++;
					}
					else if(diff>=impulse_len ) {
						break;
					}
				}

				pos[points_found] = bin_max;
				refs[points_found] = max_ref;
				points_found++;

			}
			else
				break;
		}
	}


	float bkg;
	if (tot_photons == 0)
		bkg = 0.01 / float(T - non_bkg_bins);
	else
		bkg = float(tot_photons) / float(T - non_bkg_bins);

	for (int i = 0; i < points_found; i++) {
		float ref = float(refs[i]) - bkg * impulse_len;
		if (ref>0) {
			for (int dy = 0; dy < upsampling; dy++) {
				for (int dx = 0; dx < upsampling; dx++) {
					int mx = upsampling * x + dx;
					int my = upsampling * y + dy;


					int index = points_per_pix[get_pixel_idx(mx, my, height_cloud, width_cloud)];
					points[get_point_idx(mx, my, index, height_cloud, width_cloud)] = float(pos[i]);
					reflect[get_ref_idx(mx, my, index, height_cloud, width_cloud)] = ref;

					points_per_pix[get_pixel_idx(mx, my, height_cloud, width_cloud)] = index + 1;
				}
			}
		}
	}



	bkg = log(bkg / gain);

	if (bkg > max_bkg) {
		bkg = max_bkg;
	}

	//printf("background: %f \n", exp(bkg));
	
	background[linear_idx] = bkg;

}



// this adds points
// log matched filter kernel- 1 thread per pixel
__global__ void logMatchedMSFilterKernel(float *points, float * reflect, int *points_per_pix, float *background, const float * log_impulse, const float *d_integrated_impulse,
	const int *bins_counts, const int *pix_counts, const int *pix_counts_idx, const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, 
	const int downsample, const int max_points, const float max_bkg, const float *d_gain, const int * coded_aperture, const int L) {


	extern __shared__ float irf[];
	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	for (int t = tx; t < L*impulse_len; t+=blockDim.x) {
		irf[t] = log_impulse[t];
	}
	__syncthreads();


	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);


	photon data[MAX_ACTIVE_BINS_PER_PIXEL];

	//convolve
	int act_bin = pix_counts[linear_idx]; //get total active bins
	int start_idx = pix_counts_idx[linear_idx]; //get memory
	float gain = d_gain[linear_idx];
	int wth = coded_aperture[linear_idx];

	// for pixelwise IRF
	int irf_idx_offset = wth*impulse_len;

	// read global
	int tot_photons = 0;
	for (int t = 0; t < act_bin; t++) {
		data[t] = getBinCounts(x, y, t, bins_counts, start_idx, Nrow);
		tot_photons += data[t].counts;
	}
	int non_bkg_bins = 0;

	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;

	int iter = 0;
	int refs[MAX_POINTS_PER_PIX];
	int pos[MAX_POINTS_PER_PIX];
	int points_found = 0;


	if (act_bin > 0) {
		float thres = 0; // (tot_photons / 100.);
		while ((iter++) < max_points) {
			bool flag = false;
			float max_f = -1e25;
			int  max_ref = 0;
			int bin_max = 0;
			int init_t = 0;
			for (int i = 0; i < T-impulse_len; i += downsample) {
				float filt = 0;
				int ref = 0;
				for (int t = init_t; t < act_bin; t++) {
					int diff = (data[t].bin - i);
					if (diff < 0) {
						init_t = t;
					}
					else if (diff < impulse_len) {
						filt += data[t].counts* irf[irf_idx_offset+diff];
					}
					else {
						break;
					}
				}

				if (filt > max_f) {
					max_f = filt;
					bin_max = i;
					flag = true;
				}
			}
			if (flag) {

				for (int t = 0; t < act_bin; t++) {
					int diff = (data[t].bin - bin_max);
					if (diff >= 0 && diff < impulse_len) {
						max_ref += data[t].counts;
						if (iter < max_points)
							data[t].counts = 0;
					}
					else if (diff >= impulse_len) {
						break;
					}
				}

				tot_photons -= max_ref;
				non_bkg_bins+=impulse_len;

				pos[points_found] = bin_max;
				refs[points_found] = max_ref;
				points_found++;


			}
			else
				break;
		}
	}


	float bkg;

	if (tot_photons == 0)
		bkg = 0.01/ float(T - non_bkg_bins);
	else
		bkg = float(tot_photons) / float(T - non_bkg_bins);


	
	for (int i = 0; i < points_found; i++) {
		float ref = float(refs[i]) - bkg * impulse_len;
		ref /= (float(upsampling*upsampling) * gain);
		if (ref>0) {
			for (int dy = 0; dy < upsampling; dy++) {
				for (int dx = 0; dx < upsampling; dx++) {
					int mx = upsampling * x + dx;
					int my = upsampling * y + dy;

					int index = points_per_pix[get_pixel_idx(mx,my,height_cloud,width_cloud)];
					points[get_point_idx(mx,my,index,height_cloud,width_cloud)] = float(pos[i]);

					for (int l = 0; l<L; l++)
						reflect[get_ref_idx(mx,my,index,height_cloud,width_cloud,L,l)] = ref;

					points_per_pix[get_pixel_idx(mx, my, height_cloud, width_cloud)] = index + 1;

				}
			}
		}
	}



	bkg = log(bkg / gain);

	if (bkg > max_bkg) {
		bkg = max_bkg;
	}

	//printf("background: %f \n", bkg);

	for (int l = 0; l<L; l++)
		background[get_bkg_idx(x,y,Nrow,Ncol,l)] = bkg;

}



__global__ void half_sample_mode(float *points, float * reflect, int *points_per_pix, float *background, 
	const int *bins_counts, const int *pix_counts, const int *pix_counts_idx, const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling,
	const float max_bkg, const float *d_gain, const int L, const int attack) {


	photon data[MAX_ACTIVE_BINS_PER_PIXEL];
	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);

	//convolve
	int act_bin = pix_counts[linear_idx]; //get total active bins
	int start_idx = pix_counts_idx[linear_idx]; //get memory
	float gain = d_gain[linear_idx];
	//int wth = coded_aperture[linear_idx];


	// read global
	int tot_photons = 0;
	for (int t = 0; t < act_bin; t++) {
		data[t] = getBinCounts(x, y, t, bins_counts, start_idx, Nrow);
		tot_photons += data[t].counts;
	}
	int non_bkg_bins = 0;

	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;

	int iter = 0;
	float ref = 0;
	int pos;
	int points_found = 0;
	bool flag = false;

	if (act_bin > 0) {

		int init_idx = 0, last_idx = act_bin-1;
		int remaining_phots = tot_photons;

		while (remaining_phots > 2) {
			int idx = init_idx;
			int half_phots = (remaining_phots + 2 - 1) / 2; // this is equivalent to ceil(remaining_phots/2)
			int accum = data[idx].counts;
			int minJ = 2 * T;

			int start_idx = init_idx;
			int best_last_idx = last_idx;

			while (idx < last_idx) {
				int bin_prev = data[start_idx].bin;

				while (accum < half_phots && idx < last_idx) {
					idx++;
					accum += data[idx].counts;
				}

				if (data[idx].bin - bin_prev < minJ) {
					init_idx = start_idx;
					minJ = data[idx].bin - bin_prev;
					best_last_idx = idx;
				}

				accum -= data[start_idx].counts;
				start_idx++;
			}
				
			remaining_phots = half_phots;
			last_idx = best_last_idx;

		}

		pos = (data[init_idx].bin + data[last_idx].bin)/2;
		pos -= attack;

		if (pos >= 0) {
			for (int t = 0; t < act_bin; t++) {
				int diff = (data[t].bin - pos);
				if (diff >= 0 && diff < impulse_len) {
					ref += data[t].counts;
				}
				else if (diff >= impulse_len) {
					break;
				}
			}

			tot_photons -= ref;

			flag = true;
		}
	}


	float bkg;

	if (tot_photons == 0)
		bkg = 0.01 / float(T - non_bkg_bins);
	else
		bkg = float(tot_photons) / float(T - non_bkg_bins);


	if (flag) {
		ref -= bkg * impulse_len;
		ref /= (float(upsampling*upsampling) * gain);
		if (ref > 0 && pos >= 0) {
			for (int dy = 0; dy < upsampling; dy++) {
				for (int dx = 0; dx < upsampling; dx++) {
					int mx = upsampling * x + dx;
					int my = upsampling * y + dy;
					int index = points_per_pix[get_pixel_idx(mx, my, height_cloud, width_cloud)];
					points[get_point_idx(mx, my, index, height_cloud, width_cloud)] = float(pos);
					for (int l = 0; l < L; l++)
						reflect[get_ref_idx(mx, my, index, height_cloud, width_cloud, L, l)] = ref;

					points_per_pix[get_pixel_idx(mx, my, height_cloud, width_cloud)] = index + 1;

				}
			}
		}
	}

	bkg = log(bkg / gain);

	if (bkg > max_bkg) {
		bkg = max_bkg;
	}

	for (int l = 0; l<L; l++)
		background[get_bkg_idx(x, y, Nrow, Ncol, l)] = bkg;

}



/*
// this doesn't support multiple irf per pixel
// log matched filter kernel- 1 thread per pixel
__global__ void logMatchedMSFilterKernelSMem(float *points, float * reflect, int *points_per_pix, float *background, const float * log_impulse, const float *d_integrated_impulse,
	const int *bins_counts, const int *pix_counts, const int *pix_counts_idx, const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling,
	const int downsample, const int max_points, const float max_bkg, const float *d_gain, const int * coded_aperture, const int L) {


	extern __shared__ float irf[];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int linear_idx = blockIdx.x*blockDim.x + tx;
	photon data[MAX_ACTIVE_BINS_PER_PIXEL];

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	// for pixelwise IRF
	for (int t = tx; t < impulse_len*L; t += blockDim.x)
		irf[t] = log_impulse[t];
	__syncthreads();

	//printf("irf[900] %f \n", irf[900]);


	//convolve
	int act_bin = pix_counts[linear_idx]; //get total active bins
	int start_idx = pix_counts_idx[linear_idx]; //get memory
	float gain = d_gain[linear_idx];
	int wth = coded_aperture[linear_idx];

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);

	// read global
	int tot_photons = 0;
	for (int t = 0; t < act_bin; t++) {
		data[t] = getBinCounts(x, y, t, bins_counts, start_idx, Nrow);
		tot_photons += data[t].counts;
	}
	int non_bkg_bins = 0;

	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;

	int iter = 0;
	int refs[MAX_POINTS_PER_PIX];
	int pos[MAX_POINTS_PER_PIX];
	int points_found = 0;

	if (act_bin > 0) {
		int irf_offset = wth * impulse_len;


		int offset = -1;

		while ((iter++) < max_points) {
			int init_t = 0;
			int t0 = 0, accum_impulse_len = 0;

			for (int binning = 1 << 10; binning > 1; binning >>= 1) {

				int impulse_len_s;
				if (offset < 0)
					impulse_len_s = 1;
				else {
					int aux = 1 << offset;
					int diff = aux + accum_impulse_len;
					impulse_len_s = diff < impulse_len ? aux : (impulse_len - accum_impulse_len);
				}

				float val1 = 0., val2 = 0.;
				int accum = 0, prev_equiv_t = 0;
				int irf_idx = accum_impulse_len + irf_offset;
				for (int t = init_t; t < act_bin; t++) {
					int equiv_t = data[t].bin / binning;
					if (equiv_t == prev_equiv_t) {
						accum += data[t].counts;
					}
					else {
						int diff = prev_equiv_t - t0;
						if (diff >= 0) {
							if (diff < impulse_len_s)
								val1 += float(accum *irf[irf_idx + diff]);
							diff--;
							if (diff >= 0)
								if (diff < impulse_len_s)
									val2 += float(accum *irf[irf_idx + diff]);
							//else
							//break;
						}
						//else
						//init_t = t;
						accum = data[t].counts;
					}
					prev_equiv_t = equiv_t;
				}


				int diff = prev_equiv_t - t0;
				if (diff >= 0) {
					if (diff < impulse_len_s)
						val1 += float(accum*irf[irf_idx + diff]);
					diff--;
					if (diff >= 0 && diff < impulse_len_s)
						val2 += float(accum *irf[irf_idx + diff]);
				}


				if (val2 > val1)
					t0++;

				t0 <<= 1;

				offset++;
				if (offset>0)
					accum_impulse_len += impulse_len_s;
			}
			//printf("t0:%d \n", t0);
			//printf("offset =%d accum_impulse_len:%d\n", offset, accum_impulse_len);

			int  max_ref = 0;
			for (int t = 0; t < act_bin; t++) {
				int diff = (data[t].bin - t0);
				if (diff >= 0 && diff < impulse_len) {
					max_ref += data[t].counts;
					if (iter < max_points) {
						data[t].counts = 0;
					}
				}
				else if (diff >= impulse_len) {
					break;
				}
			}

			tot_photons -= max_ref;
			non_bkg_bins += impulse_len;
			pos[points_found] = t0;
			refs[points_found] = max_ref;
			points_found++;
		}
	}


	float bkg;

	if (tot_photons == 0)
		bkg = 0.01 / float(T - non_bkg_bins);
	else
		bkg = float(tot_photons) / float(T - non_bkg_bins);

	for (int i = 0; i < points_found; i++) {
		float ref = float(refs[i]);// -bkg * impulse_len;
		ref /= (float(upsampling*upsampling) * gain);
		if (ref>0) {
			for (int dy = 0; dy < upsampling; dy++) {
				for (int dx = 0; dx < upsampling; dx++) {
					int mx = upsampling * x + dx;
					int my = upsampling * y + dy;

					int index = points_per_pix[get_pixel_idx(mx, my, height_cloud, width_cloud)];
					points[get_point_idx(mx, my, index, height_cloud, width_cloud)] = float(pos[i]);
					for (int l = 0; l<L; l++)
						reflect[get_ref_idx(mx, my, index, height_cloud, width_cloud, L, l)] = ref;

					points_per_pix[get_pixel_idx(mx, my, height_cloud, width_cloud)] = index + 1;

				}
			}
		}
	}

	bkg = log(bkg / gain);

	if (bkg > max_bkg) {
		bkg = max_bkg;
	}

	//printf("background: %f \n", bkg);

	for (int l = 0; l<L; l++)
		background[get_bkg_idx(x, y, Nrow, Ncol, l)] = bkg;

}  */





// log matched filter kernel- 1 thread per pixel
__global__ void denseLogMatchedFilterKernel(float *points, float * reflect, int *points_per_pix, float *background, const float * log_impulse, const float *d_integrated_impulse,
	const int *data, const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const int downsample, 
	const int max_points, const float max_bkg, const int many_irf, const float *d_gain, const float SBR) {


	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int x = blockIdx.x*blockDim.x + tx;
	int y = blockIdx.y*blockDim.y + ty;
	int counts[MAX_ACTIVE_BINS_PER_PIXEL];
	float impulse[MAX_ACTIVE_BINS_PER_PIXEL];

	if (x >= Nrow || y >= Ncol) //finish out-of-scope threads
		return;


	int linear_idx = (x + y * Nrow);
	// for pixelwise IRF
	int idx_offset = many_irf * linear_idx *impulse_len;


	int bkg = 0;
	// read global
	int global_idx = linear_idx * T;
	for (int t = 0; t < T; t++) {
		counts[t] = data[global_idx +t];
		bkg += counts[t];
	}

	for (int t = 0; t < impulse_len; t++) {
		impulse[t] = log_impulse[idx_offset+t];
	}

	int tot_photons = bkg;

	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;

	float gain = d_gain[linear_idx];

	int iter = 0;
	float thres = (tot_photons / 100.);
	while ((iter++) < max_points) {
		bool flag = false;
		float max_f = -1e25;
		int  max_ref = 0;
		int bin_max = 0;
		for (int i = 0; i < T; i += downsample) {
			float filt = 0;
			int ref = 0;
			for (int t = 0; t < impulse_len; t++) {
				int ind = t + i;
				if (ind < T) {
					filt += counts[ind] * impulse[t];
					ref += counts[ind];
				}
			}


			float sumH;
			int extra = i + impulse_len - T;
			if (extra > 0)
				sumH = d_integrated_impulse[idx_offset+extra - 1];
			else
				sumH = 1;
			filt = filt / sumH;

			if (filt > max_f &&  ref > thres) {
				max_f = filt;
				bin_max = i;
				max_ref = ref* SBR/(SBR+1);
				flag = true;
			}
		}
		if (flag) {

			for (int t = bin_max; t < impulse_len+ bin_max; t++) {
				if (t >= T)
					break;
				counts[t] = 0;
			}

			//printf("bkg: %f, max_ref:%f\n", bkg,max_ref);
			bkg -= max_ref;


			for (int dy = 0; dy < upsampling; dy++) {
				for (int dx = 0; dx < upsampling; dx++) {
					int mx = upsampling * x + dx;
					int my = upsampling * y + dy;
					WritePoint(mx, my, float(bin_max), float(max_ref) / float(upsampling*upsampling) / gain, points_per_pix, height_cloud, width_cloud, points, reflect);
				}
			}
		}
		else
			break;
	}




	if (bkg < 0 || bkg>tot_photons) {
		// this should never happen...
		printf("this should not happen: bkg:%d \n", bkg);
	}

	if (bkg == 0) {
		bkg = 1;
	}

	float save_bkg = log(float(bkg) / gain / float(T));

	if (save_bkg > max_bkg) {
		save_bkg = max_bkg;
	}

	//printf("background: %f , %d\n", exp(bkg), aca);

	background[linear_idx] = save_bkg;

}


// log matched filter kernel- 1 thread per pixel
__global__ void denseLogMatchedFilterOMP(float *points, float * reflect, int *points_per_pix, float *background, const float * in_impulse, const float *d_integrated_impulse,
	const int *data, const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const int downsample,
	const int max_points, const float max_bkg, const int many_irf, const float *d_gain, const float *d_irf_norm, const float SBR) {


	float vst_counts[MAX_ACTIVE_BINS_PER_PIXEL];
	float impulse[MAX_ACTIVE_BINS_PER_PIXEL];
	float counts[MAX_ACTIVE_BINS_PER_PIXEL];
	bool bkg_bin[MAX_ACTIVE_BINS_PER_PIXEL];
	
	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);


	// for pixelwise IRF
	int idx_offset = many_irf * linear_idx *impulse_len;


	// read global
	int global_idx = linear_idx * T;

	//float mean_vst = 0;
	for (int t = 0; t < T; t++) {
		counts[t] = data[global_idx + t];
		vst_counts[t] = 2+sqrtf(counts[t]+0.375);
		bkg_bin[t] = true;
		//mean_vst += vst_counts[t];
	}

	// load impulse
	for (int t = 0; t < impulse_len; t++) {
		impulse[t] = in_impulse[idx_offset + t];
	}

	float norm_impulse = d_irf_norm[linear_idx];


	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;

	float gain = d_gain[linear_idx];

	int refs[MAX_POINTS_PER_PIX];
	int pos[MAX_POINTS_PER_PIX];
	int points_found = 0;
	int iter = 0;
	while ((iter++) < max_points) {
		bool flag = false;
		float max_f = -1e99;
		int  max_ref;
		int bin_max = 0;
		for (int i = 0; i < T; i += downsample) {
			float filt = 0;
			int ref = 0;
			for (int t = 0; t < impulse_len; t++) {
				int ind = t + i;
				if (ind >= T) // make it circular
					ind -= T;
				
				filt += vst_counts[ind] * impulse[t];
				ref += counts[ind];
			}

			/*
			float sumH;
			int extra = i + impulse_len - T;
			if (extra > 0)
				sumH = d_integrated_impulse[idx_offset + extra - 1];
			else
				sumH = 1;
			filt *= sumH;*/

			if (filt > max_f) {
				max_f = filt;
				pos[points_found] = i;
				refs[points_found] = ref;
				flag = true;
			}
		}
		if (flag) {
			max_f /= norm_impulse;
			for (int t = 0; t < impulse_len; t++) {
				int ind = t + pos[points_found];
				if (ind >= T) // make it circular
					ind -= T;
				vst_counts[ind] -= impulse[t]*max_f;
				bkg_bin[ind] = false;
			}
			points_found++;


		}
		else
			break;
	}

	float bkg = 0;
	int tot_bkg_bins = 0;
	for (int t = 0; t < T; t++) {
		if (bkg_bin[t]) {
			bkg += counts[t];
			tot_bkg_bins++;
		}
	}

	if (bkg <= 0)
		bkg=0.1;

	bkg /= tot_bkg_bins;



	for (int i = 0; i < points_found; i++) {
		float ref = float(refs[i]) - bkg * impulse_len;
		if (ref>0) {
			for (int dy = 0; dy < upsampling; dy++) {
				for (int dx = 0; dx < upsampling; dx++) {
					int mx = upsampling * x + dx;
					int my = upsampling * y + dy;
					WritePoint(mx, my, float(pos[i]), ref / float(upsampling*upsampling)/gain, points_per_pix, height_cloud, width_cloud, points, reflect);
				}
			}
		}
	}



	float save_bkg = log(float(bkg)/gain);

	if (save_bkg > max_bkg) {
		save_bkg = max_bkg;
	}

	//printf("background: %f , %d\n", exp(bkg), aca);

	background[linear_idx] = save_bkg;

}



__global__ void sketchedMultiPeakInit(float *points, float * reflect, int *points_per_pix, const float * sk_impulse, const int segments,
	const float *data, const int m, const int T, const int Nrow, const int Ncol, const int upsampling, const int subsampling,
	const int many_irf, const float *d_gain, const float spectral_norm) {

	float local_data[MAX_M];
	float local_impulse[MAX_M];

	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);


	// for pixelwise IRF
	int idx_offset = many_irf * linear_idx * 2 * m;

	// read global
	int global_idx = linear_idx * 2 * m;
	float real = data[global_idx];
	float imag = data[global_idx + m];
	int points_found = 0;

	if (abs(real) > 0.001 &  abs(imag) > 0.001) {

		// read global
		for (int t = 0, g = idx_offset; t < 2 * m; t++, g++) {
			local_impulse[t] = sk_impulse[g];
		}
		for (int t = 0, g = global_idx; t < 2 * m; t++, g++) {
			local_data[t] = data[g];
		}
		

		int refs[MAX_POINTS_PER_PIX];
		int pos[MAX_POINTS_PER_PIX];
		int height_cloud = Nrow * upsampling;
		int width_cloud = Ncol * upsampling;

		float gain = d_gain[linear_idx];


		for (int p = 0; p < segments; p++) {
			int start = p * (T / segments);
			int end = (p + 1) * (T / segments);

			float max_like = -1e99;
			for (int t = start; t < end; t+=subsampling) {

				float like = 0.;
				float ref = 0.;
				for (int i = 0; i < m; i++) {
					float arg = float(t * (i + 1)*2) / float(T);
					float cosarg, sinarg;
					
					sincospif(arg, &sinarg, &cosarg);

					float aux = (cosarg * local_impulse[i] - sinarg * local_impulse[i + m]);
					float err = local_data[i] - aux;
					ref += aux * local_data[i];
					like -= err * err;

					aux = (sinarg * local_impulse[i] + cosarg * local_impulse[i + m]);
					err = local_data[i + m] - aux;
					like -= err * err;
					ref += aux * local_data[i + m];
				}

				if (like > max_like) {
					max_like = like;
					pos[p] = t;
					refs[p] = ref/spectral_norm;
				}

			}

			points_found++;
		}


		for (int i = 0; i < points_found; i++) {
			for (int dy = 0; dy < upsampling; dy++) {
				for (int dx = 0; dx < upsampling; dx++) {
					int mx = upsampling * x + dx;
					int my = upsampling * y + dy;
					WritePoint(mx, my, float(pos[i]), refs[i] / float(upsampling*upsampling) / gain, points_per_pix, height_cloud, width_cloud, points, reflect);
				}
			}
		}
	}

}

__global__ void sketchedMultiPeakOMPInit(float *points, float * reflect, int *points_per_pix, const float * sk_impulse, const int segments,
	const float *data, const int m, const int T, const int Nrow, const int Ncol, const int upsampling, const int subsampling,
	const int many_irf, const float *d_gain, const float spectral_norm) {

	float local_data[MAX_M];
	float local_impulse[MAX_M];

	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);


	// for pixelwise IRF
	int idx_offset = many_irf * linear_idx * 2 * m;

	// read global
	int global_idx = linear_idx * 2 * m;
	float real = data[global_idx];
	float imag = data[global_idx + m];
	int points_found = 0;

	if (abs(real) > 0.001 &  abs(imag) > 0.001) {

		// read global
		for (int t = 0, g = idx_offset; t < 2 * m; t++, g++) {
			local_impulse[t] = sk_impulse[g];
		}
		for (int t = 0, g = global_idx; t < 2 * m; t++, g++) {
			local_data[t] = data[g];
		}


		int refs[MAX_POINTS_PER_PIX];
		int pos[MAX_POINTS_PER_PIX];
		int height_cloud = Nrow * upsampling;
		int width_cloud = Ncol * upsampling;

		float gain = d_gain[linear_idx];


		for (int p = 0; p < segments; p++) {

			float max_like = -1e99;
			for (int t = 0; t < T; t += subsampling) {

				float like = 0.;
				float ref = 0.;
				for (int i = 0; i < m; i++) {
					float arg = float(t * (i + 1) * 2) / float(T);
					float cosarg, sinarg;

					sincospif(arg, &sinarg, &cosarg);

					float aux = (cosarg * local_impulse[i] - sinarg * local_impulse[i + m]);
					float err = local_data[i] - aux;
					ref += aux * local_data[i];
					like -= err * err;

					aux = (sinarg * local_impulse[i] + cosarg * local_impulse[i + m]);
					err = local_data[i + m] - aux;
					like -= err * err;
					ref += aux * local_data[i + m];
				}

				if (like > max_like & ref>0.) {
					max_like = like;
					pos[p] = t;
					refs[p] = ref / spectral_norm;

				}

			}


			if (p < segments - 1) // remove effect of the last peak
				for (int i = 0; i < m; i++) {
					float arg = float(pos[p] * (i + 1) * 2) / float(T);
					float cosarg, sinarg;

					sincospif(arg, &sinarg, &cosarg);

					float aux = (cosarg * local_impulse[i] - sinarg * local_impulse[i + m]);
					local_data[i] -= aux * refs[p];
					aux = (sinarg * local_impulse[i] + cosarg * local_impulse[i + m]);
					local_data[i + m] -= aux * refs[p];
				}
			points_found++;
		}


		for (int i = 0; i < points_found; i++) {
			for (int dy = 0; dy < upsampling; dy++) {
				for (int dx = 0; dx < upsampling; dx++) {
					int mx = upsampling * x + dx;
					int my = upsampling * y + dy;
					WritePoint(mx, my, float(pos[i]), refs[i] / float(upsampling*upsampling) / gain, points_per_pix, height_cloud, width_cloud, points, reflect);
				}
			}
		}
	}

}

// circular mean for 1 peak using sketched data
__global__ void circularMean(float *points, float * reflect, int *points_per_pix, const float * sk_impulse, const float * cm_corr, const float *data, const int m, const int T,
	const int Nrow, const int Ncol, const int upsampling, const int many_irf, const float *d_gain) {


	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);


	// for pixelwise IRF
	int idx_offset = many_irf * linear_idx * 2 * m;

	// read global
	int global_idx = linear_idx * 2 * m;
	float real = data[global_idx];
	float imag = data[global_idx + m];

	if (abs(real) > 0.001 &  abs(imag) > 0.001) {

		float h_real = sk_impulse[idx_offset];
		float h_imag = sk_impulse[idx_offset + m];
		float correction = cm_corr[many_irf * linear_idx];
		//printf("h_real: %f, h_imag: %f, corr: %f", h_real, h_imag, correction);

		int height_cloud = Nrow * upsampling;
		int width_cloud = Ncol * upsampling;

		float gain = d_gain[linear_idx];

		float angle = atan2f(imag, real);

		float pos = angle * T / PII;

		if (pos < 0)
			pos += T;

		pos -= correction;

		angle = pos * 2 / float(T);
		float ref = 0.;

		float sinarg, cosarg;
		sincospif(angle, &sinarg, &cosarg);

		ref += real * (cosarg*h_real - sinarg * h_imag);
		ref += imag * (sinarg*h_real + cosarg * h_imag);
		ref /= (h_real * h_real + h_imag * h_imag);
		//printf("pos:%f \t ref:%f \n", pos, ref);

		for (int dy = 0; dy < upsampling; dy++) {
			for (int dx = 0; dx < upsampling; dx++) {
				int mx = upsampling * x + dx;
				int my = upsampling * y + dy;
				WritePoint(mx, my, pos, ref / float(upsampling*upsampling) / gain, points_per_pix, height_cloud, width_cloud, points, reflect);
			}
		}
	}

}




// log matched filter kernel- 1 thread per pixel
__global__ void denseLogMatchedFilterPeaks(float *points, float * reflect, int *points_per_pix, float *background, const float * log_impulse, const float *d_integrated_impulse,
	const int *data, const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const int downsample,
	const int max_points, const float max_bkg, const int many_irf, const float *d_gain, const float SBR, const float min_dist) {


	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int x = blockIdx.x*blockDim.x + tx;
	int y = blockIdx.y*blockDim.y + ty;
	int counts[MAX_ACTIVE_BINS_PER_PIXEL];
	float impulse[MAX_ACTIVE_BINS_PER_PIXEL];

	if (x >= Nrow || y >= Ncol) //finish out-of-scope threads
		return;


	int linear_idx = (x + y * Nrow);
	// for pixelwise IRF
	int idx_offset = many_irf * linear_idx *impulse_len;


	int sigma = 2;
	int bkg = 0;
	// read global
	int global_idx = linear_idx * T;
	for (int t = 0; t < T; t++) {
		counts[t] = data[global_idx + t];
		bkg += counts[t];
	}
	float thres = (bkg / 100.);


	for (int t = 0; t < impulse_len; t++) {
		impulse[t] = log_impulse[idx_offset + t];
	}

	float cand_ref[MAX_ACTIVE_BINS_PER_PIXEL];
	float cand_filt[MAX_ACTIVE_BINS_PER_PIXEL];

	for (int i = 0, int k ; i < T; i += downsample, k++) {

		float filt = 0;
		int ref = 0;
		for (int t = 0; t < impulse_len; t++) {
			int ind = t + i;
			if (ind < T) {
				filt += counts[ind] * impulse[t];
				ref += counts[ind];
			}
		}

		float sumH;
		int extra = i + impulse_len - T;
		if (extra > 0)
			sumH = d_integrated_impulse[idx_offset + extra - 1];
		else
			sumH = 1;

		cand_ref[k] = ref * SBR / (SBR+1);
		cand_filt[k] = filt / sumH;
	}

	float final_cand_ref[MAX_ACTIVE_BINS_PER_PIXEL];
	float final_cand_filt[MAX_ACTIVE_BINS_PER_PIXEL];
	float final_cand_pos[MAX_ACTIVE_BINS_PER_PIXEL];
	int final_candidates = 0;

	// find local minima
	for (int k = 0; k < T/downsample; k++) {

		bool flag = true;
		float mean = 0;
		int t = 0;
		for (int j = -sigma; j+k>=0, j+k<T/downsample, j < sigma; j++) {
			mean += cand_filt[k+j];
			t++;
		}
		mean /= t;

		if (cand_filt[k]>mean) {
			final_cand_ref[final_candidates] = cand_ref[k];
			final_cand_filt[final_candidates] = cand_filt[k];
			final_cand_pos[final_candidates] = k*downsample;
			final_candidates++;
		}
	}


	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;
	float gain = d_gain[linear_idx];

	for (int i = 0; i < max_points; i++) {

		float max_f = -1e25;
		int k_max = -10;
		// find higher value
		for (int k = 0; k < final_candidates; k++) {
			if (final_cand_filt[k] > max_f) {
				max_f = final_cand_filt[k];
				k_max = k;
			}
		}

		// save point
		if (k_max >= 0) {
			final_cand_filt[k_max] = -2e25;
			bkg -= final_cand_ref[k_max];
			for (int dy = 0; dy < upsampling; dy++) {
				for (int dx = 0; dx < upsampling; dx++) {
					int mx = upsampling * x + dx;
					int my = upsampling * y + dy;
					WritePoint(mx, my, float(final_cand_pos[k_max]), float(final_cand_ref[k_max]) / float(upsampling*upsampling) / gain, points_per_pix, height_cloud, width_cloud, points, reflect);
				}
			}
		}
	}


	if (bkg == 0) {
		bkg = 1;
	}

	float save_bkg = log(float(bkg) / gain / float(T));

	if (save_bkg > max_bkg) {
		save_bkg = max_bkg;
	}

	//printf("background: %f , %d\n", exp(bkg), aca);

	background[linear_idx] = save_bkg;

}

// this adds points
// log matched filter kernel- 1 thread per pixel
__global__ void ReInitLogMatchedFilterKernel(float *in_points, float * in_reflect, int *in_points_per_pix, float *in_background, const float * log_impulse, const float *d_integrated_impulse,
	const int *bins_counts, const int *pix_counts, const int *pix_counts_idx, const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const int downsample,
	const int max_points, const int many_irf, const float *d_gain, float min_dist) {


	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int x = blockIdx.x*blockDim.x + tx;
	int y = blockIdx.y*blockDim.y + ty;
	photon data[MAX_ACTIVE_BINS_PER_PIXEL];

	if (x >= Nrow || y >= Ncol) //finish out-of-scope threads
		return;


	int linear_idx = (x + y * Nrow);
	// for pixelwise IRF
	int idx_offset = many_irf * linear_idx *impulse_len;

	//convolve
	int act_bin = pix_counts[linear_idx]; //get total active bins
	int start_idx = pix_counts_idx[linear_idx]; //get memory


	// read global
	for (int t = 0; t < act_bin; t++) {
		data[t] = getBinCounts(x, y, t, bins_counts, start_idx, Nrow);
	}


	float gain = d_gain[linear_idx];

	int bkg = in_background[linear_idx] * T*gain;

	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;


	int iter = 0;
	if (act_bin > 0) {
		float thres = 0; //(bkg / 100.);
		while ((iter++) < max_points) {
			bool flag = false;
			float max_f = -1e25;
			int  max_ref = 0;
			int bin_max = 0;
			int init_t = 0;
			for (int i = 0; i < T; i += downsample) {
				float filt = 0;
				int ref = 0;
				for (int t = init_t; t < act_bin; t++) {
					int diff = (data[t].bin - i);
					if (diff < 0) {
						init_t = t;
					}
					else if (diff < impulse_len) {
						filt += data[t].counts * log_impulse[idx_offset + diff];
						ref += data[t].counts;
					}
					else {
						break;
					}
				}


				/*
				float sumH = 1;
				int extra = i + impulse_len - T;
				if (extra > 0)
					sumH = d_integrated_impulse[idx_offset + extra - 1];
				else
					sumH = 1;
				filt = filt / sumH;
				*/

				if (filt > max_f &&  ref > thres) {
					max_f = filt;
					bin_max = i;
					max_ref = ref;
					flag = true;
				}
			}
			if (flag) {

				for (int t = 0; t < act_bin; t++) {
					int diff = (data[t].bin - bin_max);
					if (diff<impulse_len) {
						data[t].counts = 0;
					}
					else if (diff >= impulse_len) {
						break;
					}
				}

				//printf("bkg: %f, max_ref:%f\n", bkg,max_ref);
				bkg -= max_ref;



				for (int dy = 0; dy < upsampling; dy++) {
					for (int dx = 0; dx < upsampling; dx++) {
						int mx = upsampling * x + dx;
						int my = upsampling * y + dy;

						int npoints = in_points_per_pix[mx + my * height_cloud];
						bool found = false;
						for (int z = 0; z < npoints; z++) {
							float depth_prev = ReadPointDepth(mx, my, z, height_cloud, width_cloud, in_points);
							if (abs(depth_prev - bin_max) <= min_dist) {
								found = true;
								break;
							}
						}

						if (!found) {
							WritePoint(mx, my, float(bin_max), float(max_ref) / float(upsampling*upsampling) / gain, in_points_per_pix, height_cloud, width_cloud, in_points, in_reflect);
						}
					}
				}
			}
			else
				break;
		}
	}



}



// log matched filter kernel- 1 thread per pixel
__global__ void denseReInitLogMatchedFilterKernel(float *in_points, float * in_reflect, int *in_points_per_pix, const float *in_background, const float * log_impulse, const float *d_integrated_impulse,
	const int *data, const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const int downsample,
	const int max_points, const float max_bkg, const int many_irf, const float *d_gain, const float min_dist) {


	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int x = blockIdx.x*blockDim.x + tx;
	int y = blockIdx.y*blockDim.y + ty;
	int counts[MAX_ACTIVE_BINS_PER_PIXEL];
	int impulse[MAX_ACTIVE_BINS_PER_PIXEL];



	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;




	if (x >= Nrow || y >= Ncol) //finish out-of-scope threads
		return;


	int linear_idx = (x + y * Nrow);
	// for pixelwise IRF
	int idx_offset = many_irf * linear_idx *impulse_len;


	// read global
	int global_idx = linear_idx * T;
	for (int t = 0; t < T; t++) {
		counts[t] = data[global_idx + t];
	}

	for (int t = 0; t < impulse_len; t++) {
		impulse[t] = log_impulse[idx_offset + t];
	}



	float gain = d_gain[linear_idx];

	int bkg = in_background[linear_idx] * T*gain;
	int iter = 0;
	float thres = bkg ;
	while ((iter++) < max_points) {
		bool flag = false;
		float max_f = -1e25;
		int  max_ref = 0;
		int bin_max = 0;
		for (int i = 0; i < T; i += downsample) {
			float filt = 0;
			int ref = 0;
			for (int t = 0; t < impulse_len; t++) {
				int ind = t + i;
				if (ind < T) {
					filt += counts[ind] * impulse[t];
					ref += counts[ind];
				}
			}


			float sumH;
			int extra = i + impulse_len - T;
			if (extra > 0)
				sumH = d_integrated_impulse[idx_offset+extra - 1];
			else
				sumH = 1;
			filt = filt / sumH;

			if (filt > max_f &&  ref > thres) {
				max_f = filt;
				bin_max = i;
				max_ref = ref;
				flag = true;
			}
		}
		if (flag) {

			for (int t = bin_max; t < impulse_len + bin_max; t++) {
				if (t >= T)
					break;
				counts[t] = 0;
			}

			//printf("bkg: %f, max_ref:%f\n", bkg,max_ref);
			bkg -= max_ref;


			for (int dy = 0; dy < upsampling; dy++) {
				for (int dx = 0; dx < upsampling; dx++) {
					int mx = upsampling * x + dx;
					int my = upsampling * y + dy;

					int npoints = in_points_per_pix[mx + my * height_cloud];
					bool found = false;
					for (int z = 0; z < npoints; z++) {
						float depth_prev = ReadPointDepth(mx, my, z, height_cloud, width_cloud, in_points);
						if (abs(depth_prev - bin_max) <= min_dist){
							found = true;
							break;
						}
					}

					if (!found) {
						WritePoint(mx, my, float(bin_max), float(max_ref) / float(upsampling*upsampling) / gain, in_points_per_pix, height_cloud, width_cloud, in_points, in_reflect);
					}
				}
			}
		}
		else
			break;
	}



}