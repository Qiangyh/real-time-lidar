#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #include "device_functions.h"
#include "read_lidar.h"
#include "Point.h"
// #include "math_functions.h"

/* one thread per pixel or one thread per point*/
struct pos {
	int x;
	int y;
	int idz;
	float z;
	float r;
};



__global__ void sketched_gradient_kernel(float * in_points, float * in_reflect, int * in_points_per_pix,
	const float * data, const float * sketched_irf, const int m, const int T, const int Nrow, const int Ncol,
	const int upsampling, const float step_size_depth, const float step_size_reflec, const float max_refl, const int many_irf, const float *d_gain) {


	float pix_density[MAX_M];
	pos points[MAX_DENSE_BINS_PER_PIXEL * MAX_UPSAMPLING];
	float local_impulse[MAX_M];

	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);

	float gain = d_gain[linear_idx];

	// for pixelwise IRF
	int idx_offset = many_irf * linear_idx * 2 * m;
	int global_idx = linear_idx * 2 * m;


	// read global
	for (int t = 0, g = idx_offset; t < 2 * m; t++, g++) {
		local_impulse[t] = sketched_irf[g];
	}


	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;
	int number_of_points = 0;
	int idz = 0;
	for (int dy = 0; dy < upsampling; dy++) {
		for (int dx = 0; dx < upsampling; dx++) {
			int mx = upsampling * x + dx;
			int my = upsampling * y + dy;
			int npoints = in_points_per_pix[mx + my * height_cloud];
			number_of_points += npoints;
			for (int z = 0; z < npoints; z++) {
				//printf("here\n");
				points[idz].x = mx;	points[idz].y = my;
				points[idz].idz = z;

				points[idz].z = ReadPointDepth(mx, my, z, height_cloud, width_cloud, in_points);
				points[idz].r = ReadPointRef(mx, my, z, height_cloud, width_cloud, in_reflect);
				idz++;
			}
		}
	}



	for (int i = 0; i < 2 * m; i++) {
		pix_density[i] = -data[global_idx + i]; // goes over reals (1<i<m) and imag (m<i<2m)
	}

	// add returns
	for (int z = 0; z < number_of_points; z++) {

		for (int i = 0; i < m; i++) {
			float arg = float(points[z].z * (i + 1)* 2) / float(T);
			float cosarg, sinarg;

			sincospif(arg, &sinarg, &cosarg);

			pix_density[i] += gain * points[z].r * (cosarg * local_impulse[i] - sinarg * local_impulse[i + m]);
			pix_density[i + m] += gain * points[z].r * (sinarg * local_impulse[i] + cosarg * local_impulse[i + m]);
		}

	}


	// adjust pos and ref
	for (int z = 0; z < number_of_points; z++) {
		float delta_ref = 0.;
		float delta_dep = 0.;

		for (int i = 0; i < m; i++) {
			float arg = float((i + 1)* PII) / float(T);
			float cosarg, sinarg;
			sincosf(points[z].z * arg, &sinarg, &cosarg);

			delta_dep += (arg*(-sinarg * local_impulse[i] - cosarg * local_impulse[i + m])) * pix_density[i];
			delta_dep += (arg*(cosarg * local_impulse[i] - sinarg * local_impulse[i + m])) * pix_density[i + m];


			delta_ref += (cosarg * local_impulse[i] - sinarg * local_impulse[i + m]) * pix_density[i];
			delta_ref += (sinarg * local_impulse[i] + cosarg * local_impulse[i + m]) * pix_density[i + m];
		}

		delta_dep *= points[z].r;
		delta_dep *= gain;

		points[z].r -= step_size_reflec * delta_ref;

		if (points[z].r > max_refl)
			points[z].r = max_refl;

		points[z].z -= step_size_depth * delta_dep;

		//printf("(%d,%d), %d, (%d,%d)\n", x, y, number_of_points, points[z].x, points[z].y);
		// save reflect and depth
		WritePoint(points[z].x, points[z].y, points[z].z, points[z].r, points[z].idz, height_cloud, width_cloud, in_points, in_reflect);

	}

};


__global__ void simple_dense_point_cloud_gradient_kernel(float *in_points, float * in_reflect, int *in_points_per_pix, float * in_background, 
	const int *data, const float * impulse, const float * der_impulse, const float *d_integrated_impulse, const int impulse_len, const int T,
	const int Nrow, const int Ncol, const int upsampling, const float step_size_depth, const float step_size_reflec, const float max_refl, const int many_irf, const float *d_gain) {

	float pix_density[MAX_DENSE_BINS_PER_PIXEL];
	pos points[MAX_DENSE_BINS_PER_PIXEL * MAX_UPSAMPLING];
	float local_impulse[MAX_DENSE_BINS_PER_PIXEL];
	float local_der_impulse[MAX_DENSE_BINS_PER_PIXEL];

	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);

	float gain = d_gain[linear_idx];

	// for pixelwise IRF
	int idx_offset = many_irf * linear_idx * impulse_len;



	// read global
	for (int t = 0, g = idx_offset; t < impulse_len; t++, g++) {
		local_impulse[t] = impulse[g];
		local_der_impulse[t] = der_impulse[g];
	}



	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;
	int number_of_points = 0;
	int idz = 0;
	for (int dy = 0; dy < upsampling; dy++) {
		for (int dx = 0; dx < upsampling; dx++) {
			int mx = upsampling * x + dx;
			int my = upsampling * y + dy;
			int npoints = in_points_per_pix[mx + my * height_cloud];
			number_of_points += npoints;
			for (int z = 0; z < npoints; z++) {
				//printf("here\n");
				points[idz].x = mx;	points[idz].y = my;
				points[idz].idz = z;

				points[idz].z = ReadPointDepth(mx, my, z, height_cloud, width_cloud, in_points);
				points[idz].r = ReadPointRef(mx, my, z, height_cloud, width_cloud, in_reflect);
				idz++;
			}
		}
	}
	float bkg = in_background[linear_idx];


	// compute density
	// add background
	float exp_bkg = exp(bkg);
	for (int t = 0; t < T; t++) {
		pix_density[t] = exp_bkg;
	}


	// add returns
	for (int z = 0; z < number_of_points; z++) {


		float sumH;
		int extra = int(points[z].z) + impulse_len - T;
		int begin, end;

		if (extra > 0) {
			begin = 0;
			sumH = d_integrated_impulse[idx_offset + extra - 1];
			end = 2 * impulse_len - int(points[z].z) - T;
		}
		else if (int(points[z].z) < 0) {
			begin = -int(points[z].z);
			end = impulse_len;
			sumH = 1. - d_integrated_impulse[idx_offset + impulse_len - begin];
		}
		else {
			sumH = 1;
			begin = 0;
			end = impulse_len;
		}

		//printf("point = %f, sumH = %f", points[z].z, sumH);
		for (int t = begin, g = begin + int(points[z].z); t < end; t++, g++) {
			pix_density[g] += points[z].r * local_impulse[t]*sumH;
		}

	}


	for (int t = 0, g = T * linear_idx; t < T; t++, g++) {
		pix_density[t] = data[g] / pix_density[t];
	}


	

	// adjust pos and ref
	for (int z = 0; z < number_of_points; z++) {

		
		float sumH;
		int extra = int(points[z].z) + impulse_len - T;
		int begin, end;
		

		if (extra > 0) {
			begin = 0;
			sumH = d_integrated_impulse[idx_offset + extra - 1];
			end = 2 * impulse_len - int(points[z].z) - T;
		}
		else if (int(points[z].z) < 0) {
			begin = -int(points[z].z);
			end = impulse_len;
			sumH = 1. - d_integrated_impulse[idx_offset + impulse_len - begin];
		}
		else {

			sumH = 1;
			begin = 0;
			end = impulse_len;
		}
		float delta_ref = gain;
		float delta_dep = 0;


		for (int t = begin, g = begin + int(points[z].z); t < end; t++, g++) {
			delta_ref -= local_impulse[t] * sumH * pix_density[g];
			delta_dep += local_der_impulse[t] * sumH * pix_density[g];
		}


		delta_ref *= points[z].r;
		delta_dep *= points[z].r;

		points[z].r *= exp(-step_size_reflec * delta_ref);

		if (points[z].r > max_refl)
			points[z].r = max_refl;

		points[z].z -= step_size_depth * delta_dep;

		//printf("(%d,%d), %d, (%d,%d)\n", x, y, number_of_points, points[z].x, points[z].y);
		// save reflect and depth
		WritePoint(points[z].x, points[z].y, points[z].z, points[z].r, points[z].idz, height_cloud, width_cloud, in_points, in_reflect);

	}


}






__global__ void likelihood_kernel(float * likelihood, float *in_points, float * in_reflect, int *in_points_per_pix, float * in_background, float *reg_background,
	const int *bins_counts, const int *pix_counts, const int * pix_counts_idx, const float * impulse, const float *d_integrated_impulse, 
	const int impulse_len,	const int T, const int Nrow, const int Ncol, const int upsampling, const int many_irf, const float *d_gain) {

	float pix_density[MAX_ACTIVE_BINS_PER_PIXEL];
	photon data[MAX_ACTIVE_BINS_PER_PIXEL];
	pos points[MAX_POINTS_PER_PIX*MAX_UPSAMPLING];

	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);

	// for pixelwise IRF
	int idx_offset = many_irf * linear_idx*impulse_len;

	//convolve
	int act_bin = pix_counts[linear_idx]; //get total active bins
	int start_idx = pix_counts_idx[linear_idx]; //get memory

	float gain = d_gain[linear_idx];

	// read global
	for (int t = 0; t < act_bin; t++) {
		data[t] = getBinCounts(x, y, t, bins_counts, start_idx, Nrow);
	}

	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;


	int number_of_points = 0;
	int idz = 0;
	for (int dy = 0; dy < upsampling; dy++) {
		for (int dx = 0; dx < upsampling; dx++) {
			int mx = upsampling * x + dx;
			int my = upsampling * y + dy;
			int npoints = in_points_per_pix[mx + my * height_cloud];
			number_of_points += npoints;
			for (int z = 0; z < npoints; z++) {
				points[idz].z = ReadPointDepth(mx, my, z, height_cloud, width_cloud, in_points);
				points[idz].r = ReadPointRef(mx, my, z, height_cloud, width_cloud, in_reflect);
				idz++;
			}
		}
	}


	float bkg = in_background[linear_idx];

	// add background
	float exp_bkg = exp(bkg);
	for (int t = 0; t < act_bin; t++) {
		pix_density[t] = exp_bkg;
	}


	// compute density
	for (int z = 0; z < number_of_points; z++) {
		int init_t = 0;



		float sumH;
		int extra = points[z].z + impulse_len - T;
		if (extra > 0)
			sumH = d_integrated_impulse[idx_offset + extra - 1];
		else if (int(points[z].z) < 0) {
			sumH = 1. - d_integrated_impulse[idx_offset + impulse_len + int(points[z].z)];
		}
		else
			sumH = 1;

		for (int t = init_t; t < act_bin; t++) {
			int diff = (data[t].bin - int(points[z].z));
			if (diff<0) {
				init_t = t;
			}
			else if (diff< impulse_len) {
				pix_density[t] += points[z].r * impulse[idx_offset + diff]*sumH;

			}
			else {
				break;
			}
		}
	}


	// compute likelihood
	float like = 0;

	for (int t = 0; t < act_bin; t++) {
		like += data[t].counts*log(gain*pix_density[t]);
	}

	like -= exp_bkg * T;
	for (int z = 0; z < number_of_points; z++) {
		like -= gain*points[z].r ;
	}


	// adjust pos and ref
	likelihood[linear_idx] = like;
}



__global__ void likelihood_MS_kernel(float * likelihood, float *in_points, float * in_reflect, int *in_points_per_pix, float * in_background, float *reg_background,
	const int *bins_counts, const int *pix_counts, const int * pix_counts_idx, const float * impulse, const float *d_integrated_impulse,
	const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const int many_irf, const float *d_gain, const int *coded_aperture, const int L) {

	float pix_density[MAX_ACTIVE_BINS_PER_PIXEL];
	photon data[MAX_ACTIVE_BINS_PER_PIXEL];
	pos points[MAX_POINTS_PER_PIX * MAX_UPSAMPLING];

	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);

	//convolve
	int act_bin = pix_counts[linear_idx]; //get total active bins
	int start_idx = pix_counts_idx[linear_idx]; //get memory
	int wth  = coded_aperture[linear_idx];
	float gain = d_gain[linear_idx];

	// read global
	for (int t = 0; t < act_bin; t++) {
		data[t] = getBinCounts(x, y, t, bins_counts, start_idx, Nrow);
	}

	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;


	int number_of_points = 0;
	int idz = 0;
	for (int dy = 0; dy < upsampling; dy++) {
		for (int dx = 0; dx < upsampling; dx++) {
			int mx = upsampling * x + dx;
			int my = upsampling * y + dy;
			int npoints = in_points_per_pix[get_pixel_idx(mx,my,height_cloud,width_cloud)];
			number_of_points += npoints;
			for (int z = 0; z < npoints; z++) {
				points[idz].x = mx;	points[idz].y = my;
				points[idz].idz = z;
				points[idz].z = in_points[get_point_idx(mx,my,z,height_cloud,width_cloud)];
				points[idz].r = in_reflect[get_ref_idx(mx, my, z, height_cloud, width_cloud, L, wth)];
				idz++;
			}
		}
	}


	float bkg = in_background[get_bkg_idx(x,y,Nrow,Ncol,wth)];

	// add background
	float exp_bkg = exp(bkg);
	for (int t = 0; t < act_bin; t++) {
		pix_density[t] = exp_bkg;
	}


	// for pixelwise IRF
	int idx_offset = (wth + many_irf * linear_idx ) *impulse_len;

	// compute density
	for (int z = 0; z < number_of_points; z++) {
		int init_t = 0;

		float sumH;
		int extra = points[z].z + impulse_len - T;
		if (extra > 0)
			sumH = d_integrated_impulse[idx_offset + extra - 1];
		else if (int(points[z].z) < 0) {
			sumH = 1. - d_integrated_impulse[idx_offset + impulse_len + int(points[z].z)];
		}
		else
			sumH = 1;

		for (int t = init_t; t < act_bin; t++) {
			int diff = (data[t].bin - int(points[z].z));
			if (diff<0) {
				init_t = t;
			}
			else if (diff< impulse_len) {
				pix_density[t] += points[z].r * impulse[idx_offset + diff] * sumH;

			}
			else {
				break;
			}
		}
	}


	// compute likelihood
	float like = 0;

	for (int t = 0; t < act_bin; t++) {
		like += data[t].counts*log(gain*pix_density[t]);

	}

	like -= exp_bkg * T;
	for (int z = 0; z < number_of_points; z++) {
		like -= gain * points[z].r;
	}


	// adjust pos and ref
	likelihood[linear_idx] = like;
}




__global__ void dense_likelihood_kernel(float * likelihood, float *in_points, float * in_reflect, int *in_points_per_pix, float * in_background, float *reg_background,
	const int *data, const float * impulse, const float * d_integrated_impulse,	const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling,
	const int many_irf, const float *d_gain) {

	float pix_density[MAX_DENSE_BINS_PER_PIXEL];
	pos points[MAX_POINTS_PER_PIX * MAX_UPSAMPLING];
	float local_impulse[MAX_DENSE_BINS_PER_PIXEL];

	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);

	float gain = d_gain[linear_idx];

	// for pixelwise IRF
	int idx_offset = many_irf * linear_idx*impulse_len;

	// read global
	for (int t = 0; t < impulse_len; t++) {
		local_impulse[t] = impulse[t + idx_offset];
	}

	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;
	int number_of_points = 0;
	int idz = 0;
	for (int dy = 0; dy < upsampling; dy++) {
		for (int dx = 0; dx < upsampling; dx++) {
			int mx = upsampling * x + dx;
			int my = upsampling * y + dy;
			int npoints = in_points_per_pix[mx + my * height_cloud];
			number_of_points += npoints;
			for (int z = 0; z < npoints; z++) {
				points[idz].x = mx;	points[idz].y = my;
				points[idz].idz = z;
				points[idz].z = ReadPointDepth(mx, my, z, height_cloud, width_cloud, in_points);
				points[idz].r = ReadPointRef(mx, my, z, height_cloud, width_cloud, in_reflect);
				idz++;
			}
		}
	}
	float bkg = in_background[linear_idx];


	// compute density
	// add background
	float exp_bkg = exp(bkg);
	for (int t = 0; t < T; t++) {
		pix_density[t] = exp_bkg;
	}


	// add returns
	for (int z = 0; z < number_of_points; z++) {
		int end = impulse_len;
		int begin = 0;
		float sumH = 1.;

		int extra = points[z].z + impulse_len - T;
		if (extra > 0) {
			sumH = d_integrated_impulse[idx_offset + extra - 1];
			end = 2 * impulse_len - int(points[z].z) - T;
		}
		else if (int(points[z].z) < 0) {
			begin = -int(points[z].z);
			sumH = 1. - d_integrated_impulse[idx_offset + impulse_len + int(points[z].z)];
		}

		for (int t = begin, g = begin+ int(points[z].z); t < end; t++, g++) {
			pix_density[g] += points[z].r * local_impulse[t]*sumH;
		}
	}

	// compute likelihood
	float like = 0;

	for (int t = 0, g= T * linear_idx; t < T; t++, g++) {
		like += data[g] *log(gain*pix_density[t]);
	}

	like -= exp_bkg * T;
	for (int z = 0; z < number_of_points; z++) {
		like -= gain * points[z].r;
	}


	// adjust pos and ref
	likelihood[linear_idx] = like;

}


__global__ void sketch_likelihood_kernel(float * likelihood, float *in_points, float * in_reflect, int *in_points_per_pix,
	const float *data, const float * sketched_irf,  const int m, const int T, const int Nrow, const int Ncol, const int upsampling,
	const int many_irf, const float *d_gain) {

	float pix_density[MAX_M];
	pos points[MAX_DENSE_BINS_PER_PIXEL * MAX_UPSAMPLING];
	float local_impulse[MAX_M];

	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);

	float gain = d_gain[linear_idx];

	// for pixelwise IRF
	int idx_offset = many_irf * linear_idx * 2 * m;
	int global_idx = linear_idx * 2 * m;


	// read global
	for (int t = 0, g = idx_offset; t < 2 * m; t++, g++) {
		local_impulse[t] = sketched_irf[g];
	}


	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;
	int number_of_points = 0;
	int idz = 0;
	for (int dy = 0; dy < upsampling; dy++) {
		for (int dx = 0; dx < upsampling; dx++) {
			int mx = upsampling * x + dx;
			int my = upsampling * y + dy;
			int npoints = in_points_per_pix[mx + my * height_cloud];
			number_of_points += npoints;
			for (int z = 0; z < npoints; z++) {
				//printf("here\n");
				points[idz].x = mx;	points[idz].y = my;
				points[idz].idz = z;

				points[idz].z = ReadPointDepth(mx, my, z, height_cloud, width_cloud, in_points);
				points[idz].r = ReadPointRef(mx, my, z, height_cloud, width_cloud, in_reflect);
				idz++;
			}
		}
	}


	for (int i = 0; i < 2 * m; i++) {
		pix_density[i] = -data[global_idx + i]; // goes over reals (1<i<m) and imag (m<i<2m)
	}

	// add returns
	for (int z = 0; z < number_of_points; z++) {

		//printf("point = %f, sumH = %f", points[z].z, sumH);
		for (int i = 0; i < m; i++) {
			float arg = float(points[z].z * (i + 1)* 2) / float(T);
			float cosarg, sinarg;
			sincospif(arg, &sinarg, &cosarg);
			pix_density[i] += gain * points[z].r * (cosarg * local_impulse[i] - sinarg * local_impulse[i + m]);
			pix_density[i + m] += gain * points[z].r * (sinarg * local_impulse[i] + cosarg * local_impulse[i + m]);
		}

	}

	// compute likelihood
	float like = 0;

	for (int i = 0; i < 2*m; i++) {
		float aux = pix_density[i];
		like +=  aux * aux;
	}


	// adjust pos and ref
	likelihood[linear_idx] = like/2.;

}



// WARNING: This function was not tested properly
__global__ void prox_bkg_kernel(float *in_points, float * in_reflect, int *in_points_per_pix, float * in_background, 
	const int *bins_counts, const int *pix_counts, const int * pix_counts_idx, const float * impulse, const float *d_integrated_impulse,
	const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const float step_size_bkg,
	const float max_bkg, const int many_irf, const float *d_gain) {

	float pix_density[MAX_ACTIVE_BINS_PER_PIXEL];
	photon data[MAX_ACTIVE_BINS_PER_PIXEL];
	pos points[MAX_POINTS_PER_PIX * MAX_UPSAMPLING];

	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);

	float gain = d_gain[linear_idx];
	// for pixelwise IRF
	int idx_offset = many_irf * linear_idx*impulse_len;

	//convolve
	int act_bin = pix_counts[linear_idx]; //get total active bins
	int start_idx = pix_counts_idx[linear_idx]; //get memory


	// read global
	for (int t = 0; t < act_bin; t++) {
		data[t] = getBinCounts(x, y, t, bins_counts, start_idx, Nrow);
	}

	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;


	int number_of_points = 0;
	int idz = 0;
	for (int dy = 0; dy < upsampling; dy++) {
		for (int dx = 0; dx < upsampling; dx++) {
			int mx = upsampling * x + dx;
			int my = upsampling * y + dy;
			int npoints = in_points_per_pix[mx + my * height_cloud];
			number_of_points += npoints;
			for (int z = 0; z < npoints; z++) {
				points[idz].x = mx;	points[idz].y = my;
				points[idz].idz = z;
				points[idz].z = ReadPointDepth(mx, my, z, height_cloud, width_cloud, in_points);
				points[idz].r = ReadPointRef(mx, my, z, height_cloud, width_cloud, in_reflect);
				idz++;
			}
		}
	}


	float bkg = in_background[linear_idx];

	// add background
	float exp_bkg = exp(bkg);
	for (int t = 0; t < act_bin; t++) {
		pix_density[t] = exp_bkg;
	}


	// compute density
	for (int z = 0; z < number_of_points; z++) {
		int init_t = 0;
		for (int t = init_t; t < act_bin; t++) {
			int diff = (data[t].bin - int(points[z].z));
			if (diff < 0) {
				init_t = t;
			}
			else if (diff < impulse_len) {
				pix_density[t] += points[z].r * impulse[idx_offset + diff];
			}
			else {
				break;
			}
		}
	}


	// adjust background
	float delta_bkg = gain * T;
	for (int t = 0; t < act_bin; t++) {
		delta_bkg -= data[t].counts / pix_density[t];
	}
	delta_bkg *= exp_bkg;



	if (delta_bkg == delta_bkg) 
		bkg -= step_size_bkg * delta_bkg;
	else
		printf("Error bkg update \n", exp_bkg);

	

	if (bkg > max_bkg) {
		bkg = max_bkg;
	}

	//savebackground 
	in_background[linear_idx] = bkg;
}



__global__ void prox_MSbkg_kernel(float *in_points, float * in_reflect, int *in_points_per_pix, float * in_background,
	const int *bins_counts, const int *pix_counts, const int * pix_counts_idx, const float * impulse, const float *d_integrated_impulse,
	const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const float step_size_bkg,
	const float max_bkg, const int many_irf, const float *d_gain, const int * coded_aperture, const int L) {

	float pix_density[MAX_ACTIVE_BINS_PER_PIXEL];
	photon data[MAX_ACTIVE_BINS_PER_PIXEL];
	pos points[MAX_POINTS_PER_PIX * MAX_UPSAMPLING];

	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);

	float gain = d_gain[linear_idx];
	int wth = coded_aperture[linear_idx];


	//convolve
	int act_bin = pix_counts[linear_idx]; //get total active bins
	int start_idx = pix_counts_idx[linear_idx]; //get memory


	// read global
	for (int t = 0; t < act_bin; t++) {
		data[t] = getBinCounts(x, y, t, bins_counts, start_idx, Nrow);
	}

	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;


	int number_of_points = 0;
	int idz = 0;
	for (int dy = 0; dy < upsampling; dy++) {
		for (int dx = 0; dx < upsampling; dx++) {
			int mx = upsampling * x + dx;
			int my = upsampling * y + dy;
			int npoints = in_points_per_pix[get_pixel_idx(mx,my,height_cloud,width_cloud)];
			number_of_points += npoints;
			for (int z = 0; z < npoints; z++) {
				points[idz].x = mx;	points[idz].y = my;
				points[idz].idz = z;
				points[idz].z = in_points[get_point_idx(mx,my,z,height_cloud,width_cloud)];
				points[idz].r = in_reflect[get_ref_idx(mx, my, z, height_cloud, width_cloud, L, wth)];
				idz++;
			}
		}
	}


	int bkg_idx = get_bkg_idx(x, y, Nrow, Ncol, wth);
	float bkg = in_background[bkg_idx];

	// add background
	float exp_bkg = exp(bkg);

	//printf("bkg: %f \n", exp_bkg);
	for (int t = 0; t < act_bin; t++) {
		pix_density[t] = exp_bkg;
	}


	// for pixelwise IRF
	int irf_idx_offset = (many_irf * linear_idx + wth)*impulse_len;

	// compute density
	for (int z = 0; z < number_of_points; z++) {
		int init_t = 0;
		for (int t = init_t; t < act_bin; t++) {
			int diff = (data[t].bin - int(points[z].z));
			if (diff < 0) {
				init_t = t;
			}
			else if (diff < impulse_len) {
				pix_density[t] += points[z].r * impulse[irf_idx_offset + diff];
			}
			else {
				break;
			}
		}
	}


	// adjust background
	float delta_bkg = gain * T;
	for (int t = 0; t < act_bin; t++) {
		delta_bkg -= data[t].counts / pix_density[t];
	}
	delta_bkg *= exp_bkg;



	if (delta_bkg == delta_bkg)
		bkg -= step_size_bkg * delta_bkg;
	else
		printf("Error bkg update \n", exp_bkg);



	if (bkg > max_bkg) {
		bkg = max_bkg;
	}

	//savebackground 
	in_background[bkg_idx] = bkg;
}

__global__ void point_cloud_gradient_kernel(float *in_points, float * in_reflect, int *in_points_per_pix, float * in_background, 
	const int *bins_counts, const int *pix_counts, const int * pix_counts_idx, const float * impulse, const float * der_impulse, const float *d_integrated_impulse,
	const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const float step_size_depth,
	const float step_size_reflec, const float max_refl, const int many_irf, const float *d_gain) {

	float pix_density[MAX_ACTIVE_BINS_PER_PIXEL];
	photon data[MAX_ACTIVE_BINS_PER_PIXEL];
	pos points[MAX_POINTS_PER_PIX * MAX_UPSAMPLING];

	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);

	float gain = d_gain[linear_idx];

	// for pixelwise IRF
	int idx_offset = many_irf * linear_idx*impulse_len;

	//convolve
	int act_bin = pix_counts[linear_idx]; //get total active bins
	int start_idx = pix_counts_idx[linear_idx]; //get memory

	// read global
	for (int t = 0; t < act_bin; t++) {
		data[t] = getBinCounts(x, y, t, bins_counts, start_idx, Nrow);
	}

	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;


	int number_of_points = 0;
	int idz = 0;
	for (int dy = 0; dy < upsampling; dy++) {
		for (int dx = 0; dx < upsampling; dx++) {
			int mx = upsampling * x + dx;
			int my = upsampling * y + dy;
			int npoints = in_points_per_pix[mx + my * height_cloud];
			number_of_points += npoints;
			for (int z = 0; z < npoints; z++) {
				points[idz].x = mx;	points[idz].y = my;
				points[idz].idz = z;
				points[idz].z = ReadPointDepth(mx, my, z, height_cloud, width_cloud, in_points);
				points[idz].r = ReadPointRef(mx, my, z, height_cloud, width_cloud, in_reflect);
				idz++;
			}
		}
	}


	float bkg = in_background[linear_idx];

	// add background
	float exp_bkg = exp(bkg);
	for (int t = 0; t < act_bin; t++) {
		pix_density[t] = exp_bkg;
	}


	// compute density
	for (int z = 0; z < number_of_points; z++) {
		int init_t = 0;

		float sumH;
		int extra = points[z].z + impulse_len - T;

		if (extra > 0) {
			sumH = d_integrated_impulse[idx_offset + extra - 1];
		}
		else if (int(points[z].z) < 0) {
			sumH = 1. - d_integrated_impulse[idx_offset + impulse_len + int(points[z].z)];
		}
		else {
			sumH = 1;
		}

		for (int t = init_t; t < act_bin; t++) {
			int diff = (data[t].bin - int(points[z].z));
			if (diff < 0) {
				init_t = t;
			}
			else if (diff < impulse_len) {
				pix_density[t] += points[z].r * impulse[idx_offset + diff]*sumH;
			}
			else {
				break;
			}
		}
	}


	// adjust pos and ref
	for (int z = 0; z < number_of_points; z++) {
		float delta_ref = gain;
		float delta_dep = 0;

		for (int t = 0; t < act_bin; t++) {
			int diff = (data[t].bin - int(points[z].z));
			if (diff < 0) {
			}
			else if (diff < impulse_len) {
				delta_ref -= data[t].counts*impulse[idx_offset + diff] / pix_density[t];
				delta_dep += data[t].counts*der_impulse[idx_offset + diff] / pix_density[t];
			}
			else {
				break;
			}
		}

		delta_ref *= points[z].r;
		delta_dep *= points[z].r;

		points[z].r *= exp(-step_size_reflec * delta_ref);

		if (points[z].r > max_refl)
			points[z].r = max_refl;

		points[z].z = points[z].z - step_size_depth *delta_dep;


		// save reflect and depth
		WritePoint(points[z].x, points[z].y, points[z].z, points[z].r, points[z].idz, height_cloud, width_cloud, in_points, in_reflect);

	}
}




__global__ void point_cloud_ms_gradient_kernel(float *in_points, float * in_reflect, int *in_points_per_pix, float * in_background,
	const int *bins_counts, const int *pix_counts, const int * pix_counts_idx, const float * impulse, const float * der_impulse, const float *d_integrated_impulse,
	const int impulse_len, const int T, const int Nrow, const int Ncol, const int upsampling, const float step_size_depth,
	const float step_size_reflec, const float max_refl, const int many_irf, const float *d_gain, const int *coded_aperture, const int L) {

	float pix_density[MAX_ACTIVE_BINS_PER_PIXEL];
	photon data[MAX_ACTIVE_BINS_PER_PIXEL];
	pos points[MAX_POINTS_PER_PIX * MAX_UPSAMPLING];

	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);

	float gain = d_gain[linear_idx];

	// read color coded aperture
	int wth = coded_aperture[linear_idx];


	//convolve
	int act_bin = pix_counts[linear_idx]; //get total active bins
	int start_idx = pix_counts_idx[linear_idx]; //get memory

	// read global
	for (int t = 0; t < act_bin; t++) {
		data[t] = getBinCounts(x, y, t, bins_counts, start_idx, Nrow);
	}

	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;

	int number_of_points = 0;
	int idz = 0;
	for (int dy = 0; dy < upsampling; dy++) {
		for (int dx = 0; dx < upsampling; dx++) {
			int mx = upsampling * x + dx;
			int my = upsampling * y + dy;
			int npoints = in_points_per_pix[get_pixel_idx(mx, my, height_cloud, width_cloud)];
			number_of_points += npoints;
			for (int z = 0; z < npoints; z++) {
				points[idz].x = mx;	points[idz].y = my;	points[idz].idz = z;
				points[idz].z = in_points[get_point_idx(mx,my,z,height_cloud,width_cloud)];
				points[idz].r = in_reflect[get_ref_idx(mx,my,z,height_cloud,width_cloud,L,wth)];
				idz++;
			}
		}
	}


	float bkg = in_background[get_bkg_idx(x,y,Nrow,Ncol,wth)];

	// add background
	float exp_bkg = exp(bkg);
	for (int t = 0; t < act_bin; t++) {
		pix_density[t] = exp_bkg;
	}


	// for pixelwise IRF
	int irf_idx_offset = (wth +many_irf * linear_idx)*impulse_len;

	// compute density
	for (int z = 0; z < number_of_points; z++) {
		int init_t = 0;

		float sumH;
		int extra = points[z].z + impulse_len - T;

		if (extra > 0) {
			sumH = d_integrated_impulse[irf_idx_offset + extra - 1];
		}
		else if (int(points[z].z) < 0) {
			sumH = 1. - d_integrated_impulse[irf_idx_offset + impulse_len + int(points[z].z)];
		}
		else { 
			sumH = 1;
		}

		for (int t = init_t; t < act_bin; t++) {
			int diff = (data[t].bin - int(points[z].z));
			if (diff < 0) {
				init_t = t;
			}
			else if (diff < impulse_len) {
				pix_density[t] += points[z].r * impulse[irf_idx_offset + diff] * sumH;
			}
			else {
				break;
			}
		}
	}


	// adjust pos and ref
	for (int z = 0; z < number_of_points; z++) {

		float delta_ref = gain;
		float delta_dep = 0;

		for (int t = 0; t < act_bin; t++) {
			int diff = (data[t].bin - int(points[z].z));
			if (diff < 0) {
			}
			else if (diff < impulse_len) {
				delta_ref -= data[t].counts*impulse[irf_idx_offset + diff] / pix_density[t];
				delta_dep += data[t].counts*der_impulse[irf_idx_offset + diff] / pix_density[t];
			}
			else {
				break;
			}
		}

		delta_ref *= points[z].r;
		delta_dep *= points[z].r;

		points[z].r *= exp(-step_size_reflec * delta_ref);

		if (points[z].r > max_refl)
			points[z].r = max_refl;

		points[z].z = points[z].z - step_size_depth * delta_dep;

		// save reflect and depth
		in_points[get_point_idx(points[z].x, points[z].y, points[z].idz,height_cloud,width_cloud)] = float(points[z].z);
		in_reflect[get_ref_idx(points[z].x, points[z].y, points[z].idz, height_cloud, width_cloud,L,wth)] = points[z].r;
	}
}


template <unsigned int blockSize>
__device__ void warpReduce(volatile float* sdata, int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

// n is the number of elements 
template <unsigned int blockSize>
__global__ void reduce(float *g_idata, float *g_odata, unsigned int n)
{
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 0;
	while (i < n) {
		if (i + blockSize < n)
			sdata[tid] += g_idata[i] + g_idata[i + blockSize];
		else
			sdata[tid] += g_idata[i];

		i += gridSize;
	}
	__syncthreads();

	if (blockSize >= 1024) {
		if (tid < 512) {
			sdata[tid] += sdata[tid + 512];
		} __syncthreads();
	}

	if (blockSize >= 512) {
		if (tid < 256) {
			sdata[tid] += sdata[tid + 256];
		} __syncthreads(); 
	}

	if (blockSize >= 256) {
		if (tid < 128) {
			sdata[tid] += sdata[tid + 128]; 
		} __syncthreads(); 
	}

	if (blockSize >= 128) {
		if (tid < 64) {
			sdata[tid] += sdata[tid + 64]; 
		} __syncthreads(); 
	}

	if (tid < 32) {
		warpReduce<blockSize>(sdata, tid);
	}

	if (tid == 0) 
		g_odata[blockIdx.x] = sdata[0];
}


template <unsigned int blockSize>
__device__ void warpMaxReduce(volatile float* sdata, int tid) {
	if (blockSize >= 64) sdata[tid] = sdata[tid] > sdata[tid + 32] ? sdata[tid] : sdata[tid+32];
	if (blockSize >= 32) sdata[tid] = sdata[tid] > sdata[tid + 16] ? sdata[tid] : sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] = sdata[tid] > sdata[tid + 8] ? sdata[tid] : sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] = sdata[tid] > sdata[tid + 4] ? sdata[tid] : sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] = sdata[tid] > sdata[tid + 2] ? sdata[tid] : sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] = sdata[tid] > sdata[tid + 1] ? sdata[tid] : sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce_max(float *g_idata, float *g_odata, unsigned int n)
{
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	float aux;
	sdata[tid] = -99999.;
	while (i < n) {
		if (i + blockSize < n)
			aux = (g_idata[i] > g_idata[i + blockSize]) ? g_idata[i] : g_idata[i + blockSize];
		else
			aux = g_idata[i];

		sdata[tid] = (aux > sdata[tid]) ? aux : sdata[tid];
		i += gridSize;
	}
	__syncthreads();

	if (blockSize >= 1024) {
		if (tid < 512) {
			sdata[tid] = (sdata[tid] > sdata[tid + 512]) ? sdata[tid] :sdata[tid + 512];
		} __syncthreads();
	}

	if (blockSize >= 512) {
		if (tid < 256) {
			sdata[tid] = (sdata[tid] > sdata[tid + 256]) ? sdata[tid] : sdata[tid + 256];
		} __syncthreads();
	}

	if (blockSize >= 256) {
		if (tid < 128) {
			sdata[tid] = (sdata[tid] > sdata[tid + 128]) ? sdata[tid] : sdata[tid + 128];
		} __syncthreads();
	}

	if (blockSize >= 128) {
		if (tid < 64) {
			sdata[tid] = (sdata[tid] > sdata[tid + 64]) ? sdata[tid] : sdata[tid + 64];
		} __syncthreads();
	}

	// within this warp the operations are synchronous
	if (tid < 32) {
		warpMaxReduce<blockSize>(sdata, tid);
	}

	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

void global_reduce(float *d_idata, float * d_odata, size_t data_size, int smemSize) {

	int dimBlock = 1024;
	int dimGrid = 1;
	int n = data_size;
	reduce<1024> <<< dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, n); 
}


// WARNING: data_size should be smaller than (dimBlock) 2048*2
void global_max_reduce(float *d_idata, float * d_odata, size_t data_size, int smemSize) {

	int dimBlock = 1024;
	int dimGrid = 1;
	int n = data_size;
	reduce_max<1024> << < dimGrid, dimBlock, smemSize >> > (d_idata, d_odata, n);
}



__global__ void dense_prox_bkg_kernel(float *in_points, float * in_reflect, int *in_points_per_pix, float * in_background, 
	const int *data, const float * impulse, const float *d_integrated_impulse, const int impulse_len, const int T,
	const int Nrow, const int Ncol, const int upsampling, const float step_size_bkg, const float max_bkg, const int many_irf, const float *d_gain) {

	float pix_density[MAX_ACTIVE_BINS_PER_PIXEL];
	pos points[MAX_POINTS_PER_PIX * 4];
	float local_impulse[MAX_ACTIVE_BINS_PER_PIXEL];

	int tx = threadIdx.x;
	int linear_idx = blockIdx.x*blockDim.x + tx;

	if (linear_idx >= Nrow * Ncol) //finish out-of-scope threads
		return;

	int y = get_y_idx(linear_idx, Nrow, Ncol);
	int x = get_x_idx(linear_idx, y, Nrow, Ncol);

	// for pixelwise IRF
	int idx_offset = many_irf * linear_idx*impulse_len;
	float gain = d_gain[linear_idx];


	// read global
	for (int t = 0, g = idx_offset; t < impulse_len; t++, g++) {
		local_impulse[t] = impulse[g];
	}



	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;
	int number_of_points = 0;
	int idz = 0;
	for (int dy = 0; dy < upsampling; dy++) {
		for (int dx = 0; dx < upsampling; dx++) {
			int mx = upsampling * x + dx;
			int my = upsampling * y + dy;
			int npoints = in_points_per_pix[mx + my * height_cloud];
			number_of_points += npoints;
			for (int z = 0; z < npoints; z++) {
				points[idz].x = mx;	points[idz].y = my;
				points[idz].idz = z;
				points[idz].z = ReadPointDepth(mx, my, z, height_cloud, width_cloud, in_points);
				points[idz].r = ReadPointRef(mx, my, z, height_cloud, width_cloud, in_reflect);
				idz++;
			}
		}
	}
	float bkg = in_background[linear_idx];


	// compute density
	// add background
	float exp_bkg = exp(bkg);
	for (int t = 0; t < T; t++) {
		pix_density[t] = exp_bkg;
	}


	// add returns
	for (int z = 0; z < number_of_points; z++) {


		float sumH;
		int extra = int(points[z].z) + impulse_len - T;
		int begin, end;

		if (extra > 0) {
			begin = 0;
			sumH = d_integrated_impulse[idx_offset + extra - 1];
			end = 2 * impulse_len - int(points[z].z) - T;
		}
		else if (int(points[z].z) < 0) {
			begin = -int(points[z].z);
			end = impulse_len;
			sumH = 1. - d_integrated_impulse[idx_offset + impulse_len - begin];
		}
		else {
			sumH = 1;
			begin = 0;
			end = impulse_len;
		}

		for (int t = begin, g = begin + int(points[z].z); t < end; t++, g++) {
			pix_density[g] += points[z].r * local_impulse[t]*sumH;
		}
	}


	for (int t = 0, g = T * linear_idx; t < T; t++, g++) {
		pix_density[t] = data[g] / pix_density[t];
	}


	// adjust background
	float delta_bkg = gain * T;
	for (int t = 0; t < T; t++) {
		delta_bkg -= pix_density[t];
	}
	delta_bkg *= exp_bkg;


	//printf("bkg: %f\n", bkg);
	delta_bkg *= step_size_bkg;

	if (delta_bkg == delta_bkg)
		bkg -= delta_bkg;

	if (bkg > max_bkg) {
		bkg = max_bkg;
	}

	//savebackground 
	in_background[linear_idx] = bkg;
}
