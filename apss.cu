#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "read_lidar.h"
#include "Eigen\Core"
#include "Point.h"
#include <Ponca/Fitting>


#define MIN_DIST_APSS_PROJECTION 0.1
#define MAX_NORMAL_ANGLE 0 // arccos(0.1) = 84 degrees 
#define MAX_PROJECTIONS 5

// simple struct to store point in shared mem
struct pwn {
	float pos;
	float norm[3];
};

struct pwnr {
	float pos;
	float norm[3];
	float r;
};
struct pr {
	float pos;
	float r;
};


struct gpr {
	float pos;
	float r[MAX_WAVELENGTHS];
};

struct gpwnr {
	float pos;
	float norm[3];
	float r[MAX_WAVELENGTHS];
};

// define weighting kernel
typedef Ponca::DistWeightFunc<BasicPoint, Ponca::SmoothWeightKernel<MyPoint::Scalar>> WeightFunc;
//typedef Ponca::DistWeightFunc<MyPoint, Ponca::ExpWeightKernel<MyPoint::Scalar>> WeightFunc;

// define algebraic sphere fit with normals
typedef Ponca::Basket<BasicPoint,WeightFunc, Ponca::OrientedSphereFit> FitNormals;

// define algebraic sphere fit without normals
typedef Ponca::Basket<BasicPoint, WeightFunc, Ponca::SphereFit> FitSphereWithoutNormals;
typedef Ponca::Basket<BasicPoint, WeightFunc, Ponca::CovariancePlaneFit> FitPlaneWithoutNormals;


/* one thread per pixel or one thread per point*/
__global__ void reset_normals(float * in_normals, int *in_points_per_pix, const int Nrow, const int Ncol) {

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int px = blockIdx.x*blockDim.y + tx; // pixel x idx
	int py = blockIdx.y*blockDim.x + ty; // pixel y idx

	if (px >= Nrow || py >= Ncol) //finish out-of-scope threads
		return;

	int number_of_points = in_points_per_pix[px + py * Nrow];

	int d = 3*(px + py * Nrow);
	for (int z = 0; z < number_of_points; z ++) {
		d += 3*(Nrow*Ncol)*z;
		in_normals[d] = 0.;
		in_normals[d+1] = 0.;
		in_normals[d+2] = -1.;
	}
}


/* shift the depth estimates by a constant (attack) */
__global__ void shift_depth_kernel(float * in_points,const  int * in_points_per_pix,const int Nrow, const int Ncol, const int attack) {


	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int px = blockIdx.x*blockDim.y + tx; // pixel x idx
	int py = blockIdx.y*blockDim.x + ty; // pixel y idx

	if (px >= Nrow || py >= Ncol) //finish out-of-scope threads
		return;


	int idx = get_pixel_idx(px, py, Nrow, Ncol);
	int number_of_points = in_points_per_pix[idx];

	for (int z = 0; z < number_of_points; z++) {
		in_points[idx + z * Nrow*Ncol] += attack;
	}

}


/* find maximum reflectivity in scene pixel */
__global__ void max_reflect_kernel(const float * in_reflect, float *out_reflect, const  int * in_points_per_pix, const int Nrow, const int Ncol, const int L) {


	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int px = blockIdx.x*blockDim.y + tx; // pixel x idx
	int py = blockIdx.y*blockDim.x + ty; // pixel y idx

	if (px >= Nrow || py >= Ncol) //finish out-of-scope threads
		return;


	int idx = get_pixel_idx(px, py, Nrow, Ncol);
	int number_of_points = in_points_per_pix[idx];

	float max = 0;
	for (int z = 0; z < number_of_points; z++) {
		for (int l = 0; l < L; l++) {
			float val = in_reflect[L*idx + l];
			max = max > val ? max : val;
		}
	}

	out_reflect[idx] = max;
}


__global__ void APSS_with_normals_resample(float *in_points, float * in_normals, float* in_reflect, int *in_points_per_pix,
	float *out_points, float *out_normals, float * out_reflect, int *out_points_per_pix,
	const int T, const int Nrow, const int Ncol, const int pixhr, const float proportion, const float scale_ratio, const int impulse_len, const int L) {

	extern __shared__ int smem[];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int px = blockIdx.x*blockDim.x + tx; // pixel x idx
	int py = blockIdx.y*blockDim.y + ty; // pixel y idx
	int blockX = int(blockDim.x);
	int blockY = int(blockDim.y);

	int grid_apron = pixhr;
	int grid_stride = (blockX + 2 * grid_apron);
	int grid_size = grid_stride * grid_stride;

	int *smem_points_per_pix = (int *)smem;
	int *smem_points_index = (int *)&smem_points_per_pix[grid_size];
	gpwnr *smem_points = (gpwnr *)&smem_points_index[grid_size];

	//load shared memory

	for (int sx = tx - blockX; sx < blockX + grid_apron; sx += blockX) {
		if (sx >= -grid_apron) {
			for (int sy = ty - blockY; sy < blockY + grid_apron; sy += blockY) {
				if (sy >= -grid_apron) {
					int nx = px - tx + sx;
					int ny = py - ty + sy;
					int d_mem = (sx + grid_apron) + (sy + grid_apron)*grid_stride;

					//mirror Lidar cube
					if (nx >= Nrow)
						nx = 2 * Nrow - (nx + 2);
					else if (nx < 0)
						nx = -nx;

					if (ny >= Ncol)
						ny = 2 * Ncol - (ny + 2);
					else if (ny < 0)
						ny = -ny;

					smem_points_per_pix[d_mem] = in_points_per_pix[get_pixel_idx(nx,ny,Nrow,Ncol)];

					//printf("sx:%d, sy:%d, d_mem: %d, value:%d \n",sx,sy, d_mem, smem_points_per_pix[d_mem] );
				}
			}
		}
	}
	__syncthreads();

	if (tx == 0 && ty == 0) {
		int prev = 0;
		for (int yy = 0; yy < grid_stride; yy++) {
			for (int xx = 0; xx < grid_stride; xx++) {
				smem_points_index[xx + yy * grid_stride] = prev;
				prev += smem_points_per_pix[xx + yy * grid_stride];
				//printf("prev: %d \n", prev);
			}
		}
	}
	__syncthreads();

	for (int sy = ty - blockY; sy < blockY + grid_apron; sy += blockY) {
		if (sy >= -grid_apron) {
			for (int sx = tx - blockX; sx < blockX + grid_apron; sx += blockX) {
				if (sx >= -grid_apron) {
					int nx = px - tx + sx;
					int ny = py - ty + sy;

					//mirror Lidar cube
					if (nx >= Nrow)
						nx = 2 * Nrow - (nx + 2);
					else if (nx < 0)
						nx = -nx;

					if (ny >= Ncol)
						ny = 2 * Ncol - (ny + 2);
					else if (ny < 0)
						ny = -ny;

					int aux_points = smem_points_per_pix[(sx + grid_apron) + (sy + grid_apron)*grid_stride];
					if (aux_points > 0) {
						int d_mem = smem_points_index[(sx + grid_apron) + (sy + grid_apron)*grid_stride];
						for (int i = 0; i < aux_points; i++, d_mem++) {
							gpwnr pp;
							pp.pos = in_points[get_point_idx(nx,ny,i,Nrow,Ncol)] / scale_ratio;

							for (int l=0;l<L;l++)
								pp.r[l] = in_reflect[get_ref_idx(nx,ny,i,Nrow,Ncol,L,l)];

							for (int n=0; n<3; n++)
								pp.norm[n] = in_normals[get_normal_idx(nx,ny,i,Nrow,Ncol,n)];

							smem_points[d_mem] = pp;
						}
					}
				}
			}
		}
	}

	__syncthreads();


	if (px >= Nrow || py >= Ncol) //finish out-of-scope threads
		return;


	// find main surfaces in the pixel
	int clusters = 0;
	gpr surfaces[MAX_POINTS_PER_PIX];
	int points_per_surf[MAX_POINTS_PER_PIX];
	float central_r[MAX_POINTS_PER_PIX*MAX_WAVELENGTHS];
	bool has_central[MAX_POINTS_PER_PIX];
	float min_dist = 2 * (pixhr + 1);
	//int max_neigh = (2 * pixhr + 1)*(2 * pixhr + 1)-1;


	// add the points in neighbouring pixels to the clusters
	for (int dy = -pixhr; dy <= pixhr; dy++) {
		for (int dx = -pixhr; dx <= pixhr; dx++) {
			int mx = dx + tx + grid_apron;
			int my = dy + ty + grid_apron;
			int npoints = smem_points_per_pix[mx + my * grid_stride];
			int idx = smem_points_index[mx + my * grid_stride];

			for (int nz = 0; nz < npoints; nz++) {
				// avoid reading the normals
				gpwnr * ppr = &smem_points[idx + nz];

				bool flag = false;
				for (int j = 0; j < clusters; j++) {
					float dist = ppr->pos - surfaces[j].pos / points_per_surf[j];

					if (dist < min_dist && dist>-min_dist) {
						points_per_surf[j]++;
						surfaces[j].pos += ppr->pos;
						if (dx == 0 && dy == 0) {
							has_central[j] = true;

							for (int l = 0; l<L; l++)
								central_r[L*j+l] += ppr->r[l];
						}
						else
							for (int l = 0; l<L; l++)
								surfaces[j].r[l] += ppr->r[l];

						flag = true;
						break;
					}
				}

				if (!flag && clusters<MAX_POINTS_PER_PIX) {
					points_per_surf[clusters] = 1;
					surfaces[clusters].pos = ppr->pos;
					if (dx == 0 && dy == 0) {
						has_central[clusters] = true;
						for (int l = 0; l < L; l++) {
							central_r[L*clusters + l] = ppr->r[l];
							surfaces[clusters].r[l] = 0;
						}
					}
					else {
						has_central[clusters] = false;

						for (int l = 0; l < L; l++) {
							surfaces[clusters].r[l] = ppr->r[l];
							central_r[L*clusters+l] = 0;
						}
					}
					clusters++;
				}
			}
		}
	}

	// compute mean position and reflectivity
	for (int j = 0; j < clusters; j++) {
		surfaces[j].pos /= points_per_surf[j];

		for (int l = 0; l < L; l++) {
			if (has_central[j]) {
				if (points_per_surf[j] > 1)
					surfaces[j].r[l] = central_r[L*j+l] * (1 - proportion) + surfaces[j].r[l] * proportion / (points_per_surf[j] - 1);
				else
					surfaces[j].r[l] = central_r[L*j+l] * (1 - proportion);
			}
			else
				surfaces[j].r[l] = surfaces[j].r[l] * proportion / points_per_surf[j];
		}
	}


	// fit surfaces
	FitNormals Fit;
	Fit.setWeightFunc(WeightFunc((float)(pixhr + 1)));
	int index = 0;
	int d = get_bkg_idx(px, py, Nrow, Ncol);
	for (int z = 0; z < clusters; z++) {
		if (points_per_surf[z] > 3) {
			// read point
			BasicPoint p(BasicPoint(MyPoint::VectorType(px, py, surfaces[z].pos)));

			// Fit APSS without normals
			bool flag = false;
			int iter = 0;

			while (!flag && (iter++) < MAX_PROJECTIONS) {
				Fit.init(p.pos());
				// Gather NEIGHBOURS
				for (int dy = -pixhr; dy <= pixhr; dy++) {
					for (int dx = -pixhr; dx <= pixhr; dx++) {
						int mx = dx + tx + grid_apron;
						int my = dy + ty + grid_apron;
						int nx = px + dx;
						int ny = py + dy;
						int npoints = smem_points_per_pix[mx + my * grid_stride];
						int idx = smem_points_index[mx + my * grid_stride];
						for (int nz = 0; nz < npoints; nz++) {
							gpwnr * pp = &smem_points[idx + nz];
							MyPoint::VectorType pos(nx, ny, pp->pos);
							MyPoint::VectorType normal(pp->norm[0], pp->norm[1], pp->norm[2]);
							Fit.addNeighbor(BasicPoint(pos, normal));
						}
					}
				}

				Ponca::FIT_RESULT fresult = Fit.finalize();


				if (fresult != Ponca::UNDEFINED) {
					float delta_z;

					// project into the same pixel position
					// I can avoid the recentering, as Fit.basisCenter()=p.pos()
					if (!Fit.isPlane()) {
						float sqr = sqrt(Fit.m_ul(2)*Fit.m_ul(2) - 4. * Fit.m_uq*Fit.m_uc) / (2.*Fit.m_uq);
						float cz = -Fit.m_ul(2) / (2.*Fit.m_uq);
						if (abs(cz - sqr) < abs(cz + sqr))
							delta_z = cz - sqr;
						else
							delta_z = cz + sqr;
					}
					else {
						delta_z = -Fit.m_uc / Fit.m_ul(2);
					}

					//printf("delta z: %f \n", delta_z);
					p.pos()(2) += delta_z;
					if (delta_z < MIN_DIST_APSS_PROJECTION && delta_z >-MIN_DIST_APSS_PROJECTION) {
						//printf("iter: %d\n", iter);
						// save point
						float z_out = scale_ratio * p.pos()(2);

						if (z_out > -impulse_len && z_out < T) {

							MyPoint::VectorType normal = Fit.primitiveGradient(p.pos());

							normal.normalize();
							if (normal(2) > 0)
								normal *= MyPoint::Scalar(-1.);

							int dd = d + (Nrow*Ncol)*index;
							out_points[dd] = z_out;
							for (int l=0;l<L;l++)
								out_reflect[L*dd+l] = surfaces[z].r[l];
							dd *= 3;
							out_normals[dd] = normal(0);
							out_normals[dd + 1] = normal(1);
							out_normals[dd + 2] = normal(2);
							index++;
						}
						flag = true;
					}
				}
				else {
					flag = true;
				}
			}
		}
	}

	out_points_per_pix[d] = index;

}




__global__ void APSS_with_normals_resample_bilateral(float *in_points, float * in_normals, float* in_reflect, int *in_points_per_pix,
	float *out_points, float *out_normals, float * out_reflect, int *out_points_per_pix,
	const int T, const int Nrow, const int Ncol, const int pixhr, const float sigma2_r, const float scale_ratio, const int impulse_len, const int L) {

	extern __shared__ int smem[];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int px = blockIdx.x*blockDim.x + tx; // pixel x idx
	int py = blockIdx.y*blockDim.y + ty; // pixel y idx
	int blockX = int(blockDim.x);
	int blockY = int(blockDim.y);

	int grid_apron = pixhr;
	int grid_stride = (blockX + 2 * grid_apron);
	int grid_size = grid_stride * grid_stride;

	int *smem_points_per_pix = (int *)smem;
	int *smem_points_index = (int *)&smem_points_per_pix[grid_size];
	gpwnr *smem_points = (gpwnr *)&smem_points_index[grid_size];

	//load shared memory

	for (int sx = tx - blockX; sx < blockX + grid_apron; sx += blockX) {
		if (sx >= -grid_apron) {
			for (int sy = ty - blockY; sy < blockY + grid_apron; sy += blockY) {
				if (sy >= -grid_apron) {
					int nx = px - tx + sx;
					int ny = py - ty + sy;
					int d_mem = (sx + grid_apron) + (sy + grid_apron)*grid_stride;

					//mirror Lidar cube
					if (nx >= Nrow)
						nx = 2 * Nrow - (nx + 2);
					else if (nx < 0)
						nx = -nx;

					if (ny >= Ncol)
						ny = 2 * Ncol - (ny + 2);
					else if (ny < 0)
						ny = -ny;

					smem_points_per_pix[d_mem] = in_points_per_pix[get_pixel_idx(nx, ny, Nrow, Ncol)];

					//printf("sx:%d, sy:%d, d_mem: %d, value:%d \n",sx,sy, d_mem, smem_points_per_pix[d_mem] );
				}
			}
		}
	}
	__syncthreads();

	if (tx == 0 && ty == 0) {
		int prev = 0;
		for (int yy = 0; yy < grid_stride; yy++) {
			for (int xx = 0; xx < grid_stride; xx++) {
				smem_points_index[xx + yy * grid_stride] = prev;
				prev += smem_points_per_pix[xx + yy * grid_stride];
				//printf("prev: %d \n", prev);
			}
		}
	}
	__syncthreads();

	for (int sy = ty - blockY; sy < blockY + grid_apron; sy += blockY) {
		if (sy >= -grid_apron) {
			for (int sx = tx - blockX; sx < blockX + grid_apron; sx += blockX) {
				if (sx >= -grid_apron) {
					int nx = px - tx + sx;
					int ny = py - ty + sy;

					//mirror Lidar cube
					if (nx >= Nrow)
						nx = 2 * Nrow - (nx + 2);
					else if (nx < 0)
						nx = -nx;

					if (ny >= Ncol)
						ny = 2 * Ncol - (ny + 2);
					else if (ny < 0)
						ny = -ny;

					int aux_points = smem_points_per_pix[(sx + grid_apron) + (sy + grid_apron)*grid_stride];
					if (aux_points > 0) {
						int d_mem = smem_points_index[(sx + grid_apron) + (sy + grid_apron)*grid_stride];
						for (int i = 0; i < aux_points; i++, d_mem++) {
							gpwnr pp;
							pp.pos = in_points[get_point_idx(nx, ny, i, Nrow, Ncol)] / scale_ratio;

							for (int l = 0; l < L; l++)
								pp.r[l] = in_reflect[get_ref_idx(nx, ny, i, Nrow, Ncol, L, l)];

							for (int n = 0; n < 3; n++)
								pp.norm[n] = in_normals[get_normal_idx(nx, ny, i, Nrow, Ncol, n)];

							smem_points[d_mem] = pp;
						}
					}
				}
			}
		}
	}

	__syncthreads();


	if (px >= Nrow || py >= Ncol) //finish out-of-scope threads
		return;


	// find main surfaces in the pixel
	int clusters = 0;
	gpr surfaces[MAX_POINTS_PER_PIX];
	int points_per_surf[MAX_POINTS_PER_PIX];
	float central_r[MAX_POINTS_PER_PIX*MAX_WAVELENGTHS];
	float weights[MAX_POINTS_PER_PIX];
	bool has_central[MAX_POINTS_PER_PIX];
	float min_dist = 2 * (pixhr + 1);
	//int max_neigh = (2 * pixhr + 1)*(2 * pixhr + 1)-1;
	

	// first get centers values
	int clin = (tx + grid_apron) + (ty + grid_apron)*grid_stride;
	int cnpoints = smem_points_per_pix[clin];
	int cidx = smem_points_index[clin];
	for (int nz = 0; nz < cnpoints; nz++) {
		// avoid reading the normals
		gpwnr * ppr = &smem_points[cidx + nz];
		points_per_surf[clusters] = 1;
		surfaces[clusters].pos = ppr->pos;
		has_central[clusters] = true;
		weights[clusters] = 1.;
		for (int l = 0; l < L; l++) {
			central_r[L*clusters + l] = ppr->r[l];
			surfaces[clusters].r[l] =  ppr->r[l];
		}
		clusters++;
	}

	// add the points in neighbouring pixels to the clusters
	for (int dy = -pixhr; dy <= pixhr; dy++) {
		for (int dx = -pixhr; dx <= pixhr; dx++) {
			if (dx != 0 || dy != 0) {
				int mx = dx + tx + grid_apron;
				int my = dy + ty + grid_apron;
				int npoints = smem_points_per_pix[mx + my * grid_stride];
				int idx = smem_points_index[mx + my * grid_stride];

				for (int nz = 0; nz < npoints; nz++) {
					// avoid reading the normals
					gpwnr * ppr = &smem_points[idx + nz];

					bool flag = false;
					for (int j = 0; j < clusters; j++) {
						float dist = ppr->pos - surfaces[j].pos / points_per_surf[j];

						
						if (dist < min_dist && dist>-min_dist) {
							points_per_surf[j]++;
							surfaces[j].pos += ppr->pos;
							float weight = expf(-float(dist * dist + dx * dx + dy * dy)*2); // spatial distance
							
							if (has_central[j]) { // bilateral filtering
								float l2_color_dist = 0;
								for (int l = 0; l < L; l++) {
									float aux2 = (ppr->r[l] - central_r[L*j + l]);
									l2_color_dist += aux2*aux2;
								}
								l2_color_dist /= sigma2_r;
								weight *= expf(- l2_color_dist);
								//printf("l2_dist: %f, weight = %f \n", l2_color_dist, weight);
							}
							for (int l = 0; l < L; l++) {
								surfaces[j].r[l] += ppr->r[l] * weight;
							}
							weights[j] += weight;

							//if (weight != weight)
								//printf("ERROR: weights\n");

							flag = true;
							break;
						}
							
					}

					if (!flag) {
						points_per_surf[clusters] = 1;
						surfaces[clusters].pos = ppr->pos;
						has_central[clusters] = false;
						weights[clusters] = expf(-float((dx*dx) + (dy*dy))*2); // spatial distance
						for (int l = 0; l < L; l++) {
							surfaces[clusters].r[l] = ppr->r[l]*weights[clusters];
						}
						clusters++;
					}
				}
			}
		}
	}

	if (clusters > MAX_POINTS_PER_PIX) {
		//printf("Error: too many points per pixel! \n");
		clusters = MAX_POINTS_PER_PIX;
	}

	// compute mean position and reflectivity
	for (int j = 0; j < clusters; j++) {
		surfaces[j].pos /= points_per_surf[j];
		for (int l = 0; l < L; l++) {
			surfaces[j].r[l] = surfaces[j].r[l]/weights[j];
			if (!has_central[j]) // reduce the amount dilated to regions without existing surfaces
				surfaces[j].r[l] /= 50.;
		}
	}


	// fit surfaces
	FitNormals Fit;
	Fit.setWeightFunc(WeightFunc((float)(pixhr + 1)));
	int index = 0;
	int d = get_bkg_idx(px, py, Nrow, Ncol);
	for (int z = 0; z < clusters; z++) {
		if (points_per_surf[z] > 3) {
			// read point
			BasicPoint p(BasicPoint(MyPoint::VectorType(px, py, surfaces[z].pos)));

			// Fit APSS without normals
			bool flag = false;
			int iter = 0;

			while (!flag && (iter++) < MAX_PROJECTIONS) {
				Fit.init(p.pos());
				// Gather NEIGHBOURS
				for (int dy = -pixhr; dy <= pixhr; dy++) {
					for (int dx = -pixhr; dx <= pixhr; dx++) {
						int mx = dx + tx + grid_apron;
						int my = dy + ty + grid_apron;
						int nx = px + dx;
						int ny = py + dy;
						int npoints = smem_points_per_pix[mx + my * grid_stride];
						int idx = smem_points_index[mx + my * grid_stride];
						for (int nz = 0; nz < npoints; nz++) {
							gpwnr * pp = &smem_points[idx + nz];
							MyPoint::VectorType pos(nx, ny, pp->pos);
							MyPoint::VectorType normal(pp->norm[0], pp->norm[1], pp->norm[2]);
							Fit.addNeighbor(BasicPoint(pos, normal));
						}
					}
				}

				Ponca::FIT_RESULT fresult = Fit.finalize();


				if (fresult != Ponca::UNDEFINED) {
					float delta_z;

					// project into the same pixel position
					// I can avoid the recentering, as Fit.basisCenter()=p.pos()
					if (!Fit.isPlane()) {
						float sqr = sqrt(Fit.m_ul(2)*Fit.m_ul(2) - 4. * Fit.m_uq*Fit.m_uc) / (2.*Fit.m_uq);
						float cz = -Fit.m_ul(2) / (2.*Fit.m_uq);
						if (abs(cz - sqr) < abs(cz + sqr))
							delta_z = cz - sqr;
						else
							delta_z = cz + sqr;
					}
					else {
						delta_z = -Fit.m_uc / Fit.m_ul(2);
					}

					//printf("delta z: %f \n", delta_z);
					p.pos()(2) += delta_z;
					if (delta_z < MIN_DIST_APSS_PROJECTION && delta_z >-MIN_DIST_APSS_PROJECTION) {
						//printf("iter: %d\n", iter);
						// save point
						float z_out = scale_ratio * p.pos()(2);

						if (z_out > -impulse_len && z_out < T) {

							MyPoint::VectorType normal = Fit.primitiveGradient(p.pos());

							normal.normalize();
							if (normal(2) > 0)
								normal *= MyPoint::Scalar(-1.);

							int dd = d + (Nrow*Ncol)*index;
							out_points[dd] = z_out;
							for (int l = 0; l<L; l++)
								out_reflect[L*dd + l] = surfaces[z].r[l];
							dd *= 3;
							out_normals[dd] = normal(0);
							out_normals[dd + 1] = normal(1);
							out_normals[dd + 2] = normal(2);
							index++;
						}
						flag = true;
					}
				}
				else {
					flag = true;
				}
			}
		}
	}

	out_points_per_pix[d] = index;

}



// one thread per pixel
__global__ void threshold_points(float *in_points, float * in_normals, float *in_reflect, int *in_points_per_pix, const int Nrow, const int Ncol, const float thres, const int L) {

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int x = blockIdx.x*blockDim.x + tx; // pixel x idx
	int y = blockIdx.y*blockDim.y + ty; // pixel y idx


	if (x >= Nrow || y >= Ncol) //finish out-of-scope threads
		return;


	int pixel_idx = get_pixel_idx(x, y, Nrow, Ncol);
	int number_of_points_in = in_points_per_pix[pixel_idx];
	int number_of_points_out = 0;

	if (number_of_points_in > 0) {
		float r[MAX_WAVELENGTHS];
		int d_out =pixel_idx;
		for (int z = 0, int d = pixel_idx; z < number_of_points_in; d += (Nrow*Ncol), z++) {
			
			//printf("r: %.5f /t", r);
			float max_r = 0;
			// compute summary of r
			for (int l = 0; l < L; l++) {
				r[l] = in_reflect[L*d+l];
				if (max_r<r[l])
					max_r = r[l];
			}
			//mean_r /= L;

			if (max_r > thres) {
				if (d != d_out) {
					in_points[d_out] = in_points[d];
					for(int l = 0; l < L;l++)
						in_reflect[L*d_out+l] = r[l];
					in_normals[3 * d_out] = in_normals[3*d];
					in_normals[3 * d_out + 1] = in_normals[3*d+1];
					in_normals[3 * d_out + 2] = in_normals[3*d+2];
				}
				d_out += (Nrow*Ncol);
				number_of_points_out++;
			}
		}
	}

	in_points_per_pix[pixel_idx] = number_of_points_out;
	//printf("points %d -> %d\n", number_of_points_in, number_of_points_out);
}

// one thread per pixel
__global__ void merge_points(float *in_points, float * in_normals, float *in_reflect, int *in_points_per_pix, const int Nrow, const int Ncol, const float min_dist) {

	int x = blockIdx.x*blockDim.x + threadIdx.x; // pixel x idx
	int y = blockIdx.y*blockDim.y + threadIdx.y; // pixel y idx
	pwnr pin[MAX_POINTS_PER_PIX];
	pwnr pout[MAX_POINTS_PER_PIX];
	int clust[MAX_POINTS_PER_PIX];
	float biggest_r[MAX_POINTS_PER_PIX];

	if (x >= Nrow || y >= Ncol) //finish out-of-scope threads
		return;
	
	int number_of_points_in = in_points_per_pix[x + y * Nrow];
	int number_of_points_out = 0;

	if (number_of_points_in > 0) {

		for (int z = 0, int d = x + y * Nrow; z < number_of_points_in; d+=(Nrow*Ncol), z++) {
			pin[z].pos = in_points[d];
			pin[z].r = in_reflect[d];
			pin[z].norm[0] = in_normals[3 * d];
			pin[z].norm[1] = in_normals[3 * d + 1];
			pin[z].norm[2] = in_normals[3 * d + 2];
		}

		pout[number_of_points_out] = pin[0];
		clust[number_of_points_out] = 1;
		biggest_r[number_of_points_out] = pin[0].r;
		number_of_points_out++;

		for (int z = 1; z < number_of_points_in; z++) {
			bool flag = false;

			// assign to a cluster
			for (int j = 0; j < number_of_points_out; j++) {
				if (abs(pin[z].pos - pout[j].pos) <= min_dist) {

					//pout[j].pos += pin[z].pos;
					pout[j].r += pin[z].r;
					if (biggest_r[j] < pin[z].r) {
						pout[j].norm[0] = pin[z].norm[0];
						pout[j].norm[1] = pin[z].norm[1];
						pout[j].norm[2] = pin[z].norm[2];
						pout[j].pos = pin[z].pos;
						biggest_r[j] = pin[z].r;
					}
					clust[j]++;
					flag = true;
					break;
				}
			}

			// if it was not assigned, create new cluster
			if (!flag) {
				pout[number_of_points_out] = pin[z];
				biggest_r[number_of_points_out] = pin[z].r;
				clust[number_of_points_out] = 1;
				number_of_points_out++;
			}
		}
	}

	// save stuff
	in_points_per_pix[x + y * Nrow] = number_of_points_out;
	for (int z = 0, int d = x + y * Nrow; z < number_of_points_in; d += (Nrow*Ncol), z++) {
		in_points[d] = pout[z].pos ;
		in_reflect[d] = pout[z].r;
		in_normals[3*d] = pout[z].norm[0];
		in_normals[3*d + 1] = pout[z].norm[1];
		in_normals[3*d + 2] = pout[z].norm[2];
	}
	//printf("points %d -> %d\n", number_of_points_in, number_of_points_out);
}






/* one thread per pixel or one thread per point for L wavelengths*/
__global__ void SPSS_without_normals_resample(const float *in_points,const float * in_reflect,const int *in_points_per_pix,
	float *out_points, float * out_reflect, int *out_points_per_pix, float * out_normals,
	const int T, const int Nrow, const int Ncol, const int pixhr, const float scale_ratio, const int impulse_len, const float proportion, const int L) {


	extern __shared__ int smem[];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int px = blockIdx.x*blockDim.x + tx; // pixel x idx
	int py = blockIdx.y*blockDim.y + ty; // pixel y idx
	int blockX = int(blockDim.x);
	int blockY = int(blockDim.y);

	int grid_apron = pixhr;
	int grid_stride = (blockX + 2 * grid_apron);
	int grid_size = grid_stride * grid_stride;

	int *smem_points_per_pix = (int *)smem;
	int *smem_points_index = (int *)&smem_points_per_pix[grid_size];
	gpr *smem_points = (gpr *)&smem_points_index[grid_size];

	//load shared memory

	for (int sx = tx - blockX; sx < blockX + grid_apron; sx += blockX) {
		if (sx >= -grid_apron) {
			for (int sy = ty - blockY; sy < blockY + grid_apron; sy += blockY) {
				if (sy >= -grid_apron) {
					int nx = px - tx + sx;
					int ny = py - ty + sy;
					int d_mem = (sx + grid_apron) + (sy + grid_apron)*grid_stride;

					//mirror Lidar cube
					if (nx >= Nrow)
						nx = 2 * Nrow - (nx + 2);
					else if (nx < 0)
						nx = -nx;

					if (ny >= Ncol)
						ny = 2 * Ncol - (ny + 2);
					else if (ny < 0)
						ny = -ny;

					smem_points_per_pix[d_mem] = in_points_per_pix[get_pixel_idx(nx,ny,Nrow,Ncol)];

					//printf("sx:%d, sy:%d, d_mem: %d, value:%d \n",sx,sy, d_mem, smem_points_per_pix[d_mem] );
				}
			}
		}
	}
	__syncthreads();

	if (tx == 0 && ty == 0) {
		int prev = 0;
		for (int yy = 0; yy < grid_stride; yy++) {
			for (int xx = 0; xx < grid_stride; xx++) {
				smem_points_index[xx + yy * grid_stride] = prev;
				prev += smem_points_per_pix[xx + yy * grid_stride];
				//printf("prev: %d \n", prev);
			}
		}
	}
	__syncthreads();

	for (int sy = ty - blockY; sy < blockY + grid_apron; sy += blockY) {
		if (sy >= -grid_apron) {
			for (int sx = tx - blockX; sx < blockX + grid_apron; sx += blockX) {
				if (sx >= -grid_apron) {
					int nx = px - tx + sx;
					int ny = py - ty + sy;

					//mirror Lidar cube
					if (nx >= Nrow)
						nx = 2 * Nrow - (nx + 2);
					else if (nx < 0)
						nx = -nx;

					if (ny >= Ncol)
						ny = 2 * Ncol - (ny + 2);
					else if (ny < 0)
						ny = -ny;

					int aux_points = smem_points_per_pix[(sx + grid_apron) + (sy + grid_apron)*grid_stride];
					if (aux_points > 0) {
						int d_mem = smem_points_index[(sx + grid_apron) + (sy + grid_apron)*grid_stride];
						for (int i = 0; i < aux_points; i++, d_mem++) {
							gpr pp;
							pp.pos = in_points[get_point_idx(nx,ny,i,Nrow,Ncol)] / scale_ratio;
							for (int l = 0; l < L; l++) {
								pp.r[l] = in_reflect[get_ref_idx(nx,ny,i,Nrow,Ncol,L,l)];
							}
							smem_points[d_mem] = pp;
						}
					}
				}
			}
		}
	}

	__syncthreads();


	if (px >= Nrow || py >= Ncol) //finish out-of-scope threads
		return;

	// clustering

	// find main surfaces in the pixel
	int clusters = 0;
	gpr surfaces[MAX_POINTS_PER_PIX];
	int points_per_surf[MAX_POINTS_PER_PIX];
	float central_r[MAX_POINTS_PER_PIX*MAX_WAVELENGTHS];
	bool has_central[MAX_POINTS_PER_PIX];
	float min_dist = 2 * (pixhr + 1);
	//int max_neigh = (2 * pixhr + 1)*(2 * pixhr + 1)-1;


	// add the points in neighbouring pixels to the clusters
	for (int dy = -pixhr; dy <= pixhr; dy++) {
		for (int dx = -pixhr; dx <= pixhr; dx++) {
			int mx = dx + tx + grid_apron;
			int my = dy + ty + grid_apron;
			int npoints = smem_points_per_pix[mx + my * grid_stride];
			int idx = smem_points_index[mx + my * grid_stride];

			for (int nz = 0; nz < npoints; nz++) {
				// avoid reading the normals
				gpr * ppr = &smem_points[idx + nz];

				bool flag = false;
				for (int j = 0; j < clusters; j++) {
					float dist = ppr->pos - surfaces[j].pos / points_per_surf[j];

					if (dist < min_dist && dist>-min_dist) {
						points_per_surf[j]++;
						surfaces[j].pos += ppr->pos;
						if (dx == 0 && dy == 0) {
							has_central[j] = true;
							for (int l = 0; l < L; l++)
								central_r[L*j+l] += ppr->r[l];
						}
						else
							for (int l = 0; l < L; l++)
								surfaces[j].r[l] += ppr->r[l];

						flag = true;
						break;
					}
				}

				if (!flag) {
					points_per_surf[clusters] = 1;
					surfaces[clusters].pos = ppr->pos;
					if (dx == 0 && dy == 0) {
						has_central[clusters] = true;
						for (int l = 0; l < L; l++) {
							central_r[L*clusters + l] += ppr->r[l];
							surfaces[clusters].r[l] = 0;
						}
					}
					else {
						has_central[clusters] = false;
						for (int l = 0; l < L; l++) {
							surfaces[clusters].r[l] = ppr->r[l];
							central_r[L*clusters + l] = 0;
						}
					}
					clusters++;
				}
			}
		}
	}

	if (clusters > MAX_POINTS_PER_PIX) {
		//printf("Error: too many points per pixel! \n");
		clusters = MAX_POINTS_PER_PIX;
	}

	// compute mean position
	for (int j = 0; j < clusters; j++) {
		surfaces[j].pos /= points_per_surf[j];
		for (int l = 0; l < L; l++) {
			if (has_central[j]) {
				if (points_per_surf[j] > 1)
					surfaces[j].r[l] = central_r[L*j + l] * (1 - proportion) 
						+ surfaces[j].r[l] * proportion / (points_per_surf[j] - 1);
				else
					surfaces[j].r[l] = central_r[L * j + l] * (1 - proportion);
			}
			else {
				surfaces[j].r[l] = surfaces[j].r[l] * proportion / points_per_surf[j];
			}
		}
	
	}

	//printf("clusters = %d \n", clusters);

	FitPlaneWithoutNormals Fit;
	Fit.setWeightFunc(WeightFunc((float)(pixhr + 1)));

	int index = 0;
	int d = get_pixel_idx(px, py, Nrow, Ncol);

	for (int z = 0; z < clusters; z++) {

		// read point
		BasicPoint p(BasicPoint(BasicPoint::VectorType(px, py, surfaces[z].pos)));

		//printf("pos = %f \n", p.pos()(2));
		// Fit APSS without normals
		bool flag = false;
		int iter = 0;

		while (!flag && (iter++) < MAX_PROJECTIONS) {
			// Fit APSS without normals
			Fit.init(p.pos());
			// Gather NEIGHBOURS
			for (int dy = -pixhr; dy <= pixhr; dy++) {
				for (int dx = -pixhr; dx <= pixhr; dx++) {
					int mx = dx + tx + grid_apron;
					int my = dy + ty + grid_apron;
					int nx = px + dx;
					int ny = py + dy;
					int npoints = smem_points_per_pix[mx + my * grid_stride];
					int idx = smem_points_index[mx + my * grid_stride];
					for (int nz = 0; nz < npoints; nz++) {
						gpr * pp = &smem_points[idx + nz];
						BasicPoint::VectorType pos(nx, ny, pp->pos);
						Fit.addNeighbor(BasicPoint(pos));
					}
				}
			}

			Ponca::FIT_RESULT fresult = Fit.finalize();

			if (fresult == Ponca::STABLE) {

				BasicPoint::VectorType normal = Fit.normal();
				//float c = Fit.m_p(3);
				float c = Fit.basisCenter()[3];

				// project into pixel
				float delta_z =  -p.pos()(2) - (p.pos()(0)*normal(0) + p.pos()(1)*normal(1) + c) / normal(2);
				//printf("plane = (%f,%f,%f,%f), z_in = %f, delta_z=%f \n", normal(0), normal(1), normal(2), c,p.pos()(2), delta_z);
				p.pos()(2) += delta_z;

				if (delta_z < MIN_DIST_APSS_PROJECTION && delta_z >-MIN_DIST_APSS_PROJECTION) {
					if (p.pos()(2) > -impulse_len && p.pos()(2) < T) {

						if (normal(2) > 0)
							normal *= BasicPoint::Scalar(-1.);

						int dd = d + (Nrow*Ncol)*index;
						out_points[dd] = scale_ratio*p.pos()(2);
						for (int l = 0; l < L; l++) {
							out_reflect[L*dd+l] = surfaces[z].r[l];
						}
						dd *= 3;
						out_normals[dd] = normal(0);
						out_normals[dd + 1] = normal(1);
						out_normals[dd + 2] = normal(2);
						index++;
					}
					flag = true;
				}
				else
					flag = false;
			}
			else 
				flag = true;
		}
	}

	out_points_per_pix[d] = index;
}





__global__ void scale_normals(int * points_per_pix, float * in_normals, const float factor, const int Nrow, const int Ncol) {


	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int px = blockIdx.x*blockDim.x + tx; // pixel x idx
	int py = blockIdx.y*blockDim.y + ty; // pixel y idx


	if (px >= Nrow || py >= Ncol) //finish out-of-scope threads
		return;


	int idx = get_bkg_idx(px, py, Nrow, Ncol);
	int points = points_per_pix[idx];

	for (int i = 0; i < points; i++) {
		int dd = 3*(idx + (Nrow*Ncol)*i);

		MyPoint::VectorType n(in_normals[dd], in_normals[dd + 1], in_normals[dd + 2]);

		n(2) /= factor;
		n.normalize();

		in_normals[dd] = n(0);
		in_normals[dd + 1] = n(1);
		in_normals[dd + 2] = n(2);
	}

}



__global__ void upsample_pointcloud(float *in_points, float * in_reflect, int *in_points_per_pix, float * in_normals,
	float *out_points, float * out_reflect, int *out_points_per_pix, float * out_normals,
	const int Nrow, const int Ncol, const int upsampling) {

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int x = blockIdx.x*blockDim.x + tx;
	int y = blockIdx.y*blockDim.y + ty;

	int id = x + Nrow * y;

	int points = in_points_per_pix[id];
	int height_cloud = Nrow * upsampling;
	int width_cloud = Ncol * upsampling;

	int short_stride = Nrow*Ncol;
	int long_stride = height_cloud*width_cloud;
	for (int i = 0; i < points; i++) {
		pwnr p;
		int dd = id + i * short_stride;
		p.pos = in_points[dd];
		p.r = in_reflect[id];
		dd *= 3;
		MyPoint::VectorType n(in_normals[dd], in_normals[dd+1], in_normals[dd+2]);
		n(2) *= upsampling;
		n.normalize();
		p.norm[0] = n(0);
		p.norm[1] = n(1);
		p.norm[2] = n(2);

		for (int dy = 0; dy < upsampling; dy++) {
			for (int dx = 0; dx < upsampling; dx++) {
				int mx = upsampling * x + dx;
				int my = upsampling * y + dy;
				int idx = mx + my * height_cloud;
				if (i==0)
					out_points_per_pix[idx] = points; 
				int d = idx + i * long_stride;
				out_points[d] = p.pos;
				out_reflect[d] = p.r;
				d *= 3;
				out_normals[d] = p.norm[0];
				out_normals[d+1] = p.norm[1];
				out_normals[d+2] = p.norm[2];
			}
		}
	}

}
