#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Patate/grenaille.h"


#define MAX_POINTS_PER_PIX 3
#define MEAN_POINTS_PER_PIX 3
#define MAX_WAVELENGTHS 4
#define MAX_UPSAMPLING 3

class BasicPoint {
public:
	enum { Dim = 3 };
	typedef float Scalar;
	typedef Eigen::Matrix<Scalar, Dim, 1> VectorType;
	typedef Eigen::Matrix<Scalar, Dim+1, 1> HVectorType;
	typedef Eigen::Matrix<Scalar, Dim, Dim> MatrixType;

	// constructor, by default everything to zero
	MULTIARCH inline BasicPoint(const VectorType &pos = VectorType::Zero(),
		const VectorType &normal = VectorType::Zero()) : _pos(pos), _normal(normal) {};

	MULTIARCH inline const VectorType & pos() const { return _pos; };
	MULTIARCH inline const VectorType & normal() const { return _normal; };

	MULTIARCH inline VectorType & pos() { return _pos; };
	MULTIARCH inline VectorType & normal() { return _normal; };

protected:
	VectorType _pos, _normal;

};



class MyPoint : public BasicPoint {
public:

	typedef Eigen::Matrix<Scalar, MAX_WAVELENGTHS, 1> ReflectivityType;

	MULTIARCH inline MyPoint(const VectorType &pos = VectorType::Zero(),
		const ReflectivityType &r = ReflectivityType::Zero(), const VectorType &normal = VectorType::Zero()) : BasicPoint(pos, normal), _r(r) {};

	MULTIARCH inline ReflectivityType & r() { return _r; };

	MULTIARCH inline const ReflectivityType & r() const { return _r; };

private:
	ReflectivityType _r;

};



/* GPU INDEXING FUNCTIONS */
MULTIARCH inline int get_pixel_idx(const int x, const int y, const int Nrow, const int Ncol) {
	return (x + y * Nrow);
}

MULTIARCH inline int get_y_idx(const int lin_idx, const int Nrow, const int Ncol) {
	return  lin_idx / Nrow;
}

MULTIARCH inline int get_x_idx(const int lin_idx, const int y, const int Nrow, const int Ncol) {
	return lin_idx - y*Nrow;
}


MULTIARCH inline int get_point_idx(const int x, const int y, const int n, const int Nrow, const int Ncol) {
	return get_pixel_idx(x, y, Nrow, Ncol) + n*Nrow*Ncol;
}

MULTIARCH inline int get_normal_idx(const int x, const int y, const int n, const int Nrow, const int Ncol, const int i) {
	return 3*get_point_idx(x,y,n,Nrow,Ncol)+i;
}

MULTIARCH inline int get_ref_idx(const int x, const int y, const int n, const int Nrow, const int Ncol) {
	return get_point_idx(x, y, n, Nrow, Ncol);
}

MULTIARCH inline int get_ref_idx(const int x, const int y, const int n, const int Nrow, const int Ncol, const int L, const int wth) {
	return L*get_point_idx(x, y, n, Nrow, Ncol) + wth;
}

MULTIARCH inline int get_bkg_idx(const int x, const int y, const int Nrow, const int Ncol) {
	return get_pixel_idx(x, y, Nrow, Ncol);
}

MULTIARCH inline int get_bkg_idx(const int x, const int y, const int Nrow, const int Ncol, const int wth) {
	return  (get_bkg_idx(x, y, Nrow, Ncol) + wth *Nrow*Ncol); // WARNING: changing this may affect the behaviour of cuFFT
}


MULTIARCH inline void WritePoint(const  int x, const  int y, const  float z, const float r, int * points_per_pix, const  int Nrow, const  int Ncol, float * points, float * reflect) {

	// make memory reads as coalesced as possible
	int index = points_per_pix[x + y * Nrow];
	points[(x + y * Nrow) + (Nrow*Ncol)*index] = float(z);
	reflect[(x + y * Nrow) + (Nrow*Ncol)*index] = r;
	points_per_pix[x + y * Nrow] = index + 1;
}


MULTIARCH inline void WritePoint(const  int x, const  int y, const float z, const float r, int index, const  int Nrow, const  int Ncol, float * points, float * reflect) {

	// make memory reads as coalesced as possible
	points[(x + y * Nrow) + (Nrow*Ncol)*index] = float(z);
	reflect[(x + y * Nrow) + (Nrow*Ncol)*index] = r;
}



MULTIARCH inline MyPoint ReadPoint(const  int x, const  int y, const  int  idx, const  int Nrow, const  int Ncol, const float * normals,
	const float * points, const float * reflect, const float scale_ratio, const int L) {

	// make memory reads as coalesced as possible
	int d = get_point_idx(x,y,idx,Nrow,Ncol);
	MyPoint::VectorType p, n;
	p << MyPoint::Scalar(x), MyPoint::Scalar(y), points[d] / scale_ratio;

	int d2 = d * 3;
	n << normals[d2], normals[d2 + 1], normals[d2 + 2];

	MyPoint::ReflectivityType r;
	for (int i = 0; i < L; i++)
		r(i) = reflect[get_ref_idx(x,y,idx,Nrow,Ncol,L,i)];

	return MyPoint(p, r, n);
}


/*
MULTIARCH inline MyPoint ReadPoint(const  int x, const  int y, const  int  idx, const  int Nrow, const  int Ncol, const float * points,float * reflect, const float scale_ratio, const int L) {

	// make memory reads as coalesced as possible
	int d = (x + y * Nrow) + (Nrow*Ncol)*idx;
	MyPoint::VectorType v;
	v << MyPoint::Scalar(x), MyPoint::Scalar(y), points[d] / scale_ratio;
	MyPoint::ReflectivityType r;
	for (int i = 0; i < L; i++)
		r(i) = reflect[i];
	return MyPoint(v, r);
}


MULTIARCH inline MyPoint ReadPoint(const  int x, const  int y, const  int  idx, const  int Nrow, const  int Ncol, const float * normals, const float * points, const float * reflect, const float scale_ratio) {

	// make memory reads as coalesced as possible
	int d = (x + y * Nrow) + (Nrow*Ncol)*idx;
	MyPoint::VectorType p, n;
	p << MyPoint::Scalar(x), MyPoint::Scalar(y), points[d] / scale_ratio;

	int d2 = d * 3;
	n << normals[d2], normals[d2+1], normals[d2+2];

	return MyPoint(p, reflect[d], n);
}


MULTIARCH inline MyPoint ReadPoint(const  int x, const  int y, const  int  idx, const  int Nrow, const  int Ncol, const float * normals, const float * points,  const float scale_ratio) {

	// make memory reads as coalesced as possible
	int d = (x + y * Nrow) + (Nrow*Ncol)*idx;
	MyPoint::VectorType p, n;
	p << MyPoint::Scalar(x), MyPoint::Scalar(y), points[d] / scale_ratio;

	int d2 = d * 3;
	n << normals[d2], normals[d2 + 1], normals[d2 + 2];

	return MyPoint(p, 0., n);
}

*/

MULTIARCH inline float ReadPointDepth(const  int x, const  int y, const  int  idx, const  int Nrow, const  int Ncol, float * points) {

	// make memory reads as coalesced as possible
	int d = (x + y * Nrow) + (Nrow*Ncol)*idx;
	return points[d];
}

MULTIARCH inline float ReadPointRef(const  int x, const  int y, const  int  idx, const  int Nrow, const  int Ncol, float * reflect) {
	// make memory reads as coalesced as possible
	int d = (x + y * Nrow) + (Nrow*Ncol)*idx;
	return reflect[d];
}
