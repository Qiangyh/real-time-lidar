#pragma once

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/visualization/pcl_plotter.h>
#include <boost/filesystem.hpp>


/*
struct MyPointType {
	PCL_ADD_POINT4D;
	PCL_ADD_UNION_NORMAL4D;
	float r[MAX_WAVELENGTHS];
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment


POINT_CLOUD_REGISTER_POINT_STRUCT(MyPointType,           // here we assume a XYZ + "test" (as fields)
(float, x, x)
(float, y, y)
(float, z, z)
(float, normal_x, normal_x)
(float, normal_y, normal_y)
(float, normal_z, normal_z)
(float[MAX_WAVELENGTHS], r, r)
)*/

class visualization {

public:

	typedef pcl::PointXYZINormal MyPointType;
	typedef struct  {
		unsigned char r;
		unsigned char g;
		unsigned char b;
	} rgb;

	typedef pcl::PointCloud<MyPointType> MyCloudType;
	typedef pcl::PointCloud<pcl::PointXYZRGB> ColorCloudType;

	void keyboard_callback(const pcl::visualization::KeyboardEvent &event, void* viewer_void);

	visualization(bool render = false, bool plot_likelihood = false, std::string filename = "") {
		data_available = false;
		plot_lhood = plot_likelihood;
		if (plot_lhood)
			plotter = new pcl::visualization::PCLPlotter("My plotter");
		if (render) {
			background_image = pcl::visualization::ImageViewer::Ptr(new pcl::visualization::ImageViewer("Background levels"));
			points_image = pcl::visualization::ImageViewer::Ptr(new pcl::visualization::ImageViewer("Points per pixel"));
			color_cloud = ColorCloudType::Ptr(new ColorCloudType);
			viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewer"));
			viewer->addCoordinateSystem(1.0);
			viewer->initCameraParameters();
			viewer->registerKeyboardCallback(&visualization::keyboard_callback, *this);
			/*
			viewer->setSize(512, 512);
			viewer->setPosition(0, 0);

			points_image->setSize(256, 256);
			points_image->setPosition(512, 0);
			background_image->setSize(256, 256);
			background_image->setPosition(512, 256); */
			first_frame = true;
			till_the_end = false;
		}


		namespace fs = boost::filesystem;

		fs::path path(std::string("output_")+ filename);

		foldername = path.string();
		foldername.append("\\");
		fs::remove_all(path);
		fs::create_directory(path);
		this->render = render;
	};

	~visualization() {


		if (render) {

			while (!viewer->wasStopped()) {
				spinAll();
			}

			viewer->close();
			points_image->close();
			if (plot_lhood)
				plotter->close();
			background_image->close();
		}
	};

	int loadPointCloud(std::vector<float> & points, std::vector<float> & normals, std::vector<float> &  reflect, std::vector<int> &  in_points_per_pix,
		float scale_ratio, int height, int width, std::vector<float> &  background, int Nrow, int Ncol, int L, std::vector<float> & likelihood);

	void savePly(std::string & filename);
	void savePointsPerPix(std::string & filename, bool binary=false);
	void saveBackground(std::string & filename, bool binary=false);
	void saveLikelihood(std::string & filename, std::vector<float> & likelihood);
	rgb colormap(unsigned char v);
	std::string getFolderName(void) { return foldername; };
	
protected:
	bool data_available, render;
	std::vector<float> bkg;
	std::vector<uint16_t> points_per_pix;
	ColorCloudType::Ptr color_cloud;
	std::vector<rgb> img_bkg, img_points_per_pix;
	std::vector< MyCloudType::Ptr, Eigen::aligned_allocator <MyCloudType::Ptr> > clouds;

	void plotLikelihood(std::vector<float> & data);
	//defining a plotter
	pcl::visualization::PCLPlotter * plotter;
	pcl::visualization::ImageViewer::Ptr background_image;
	pcl::visualization::ImageViewer::Ptr points_image;
	bool first_frame;
	std::string foldername;
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	//boost::thread thr;
	void spinAll(void);
	rgb map_reflectivity_to_color(std::vector<float> & r, float lim_max, float lim_min);

	void showPointCloud(void);

	//static boost::mutex mutex;
	bool finished_vis, till_the_end, plot_lhood;
	int Nrow, Ncol;
	int width_cloud, height_cloud;
	int wavelengths;
};

