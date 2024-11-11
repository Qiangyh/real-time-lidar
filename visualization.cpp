#include "Point.h"
#include "visualization.h"
#include <pcl/io/ply_io.h>
#include <pcl/io/png_io.h>

// boost::mutex visualization::mutex;

void visualization::plotLikelihood(std::vector<float> &data)
{

	if (data.size() > 1)
	{
		std::vector<std::pair<double, double>> xy;
		// defining the polynomial function, y = x^2. Index of x^2 is 1, rest is 0
		for (int i = 0; i < data.size(); i++)
		{
			xy.push_back(std::make_pair(double(i), double(data.at(i))));
		}

		plotter->clearPlots();
		plotter->addPlotData(xy, "likelihood");

		// display the plot, DONE!
		// plotter->plot();
	}
}

visualization::rgb visualization::colormap(unsigned char v)
{

	unsigned char r[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179, 183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 251, 247, 243, 239, 235, 231, 227, 223, 219, 215, 211, 207, 203, 199, 195, 191, 187, 183, 179, 175, 171, 167, 163, 159, 155, 151, 147, 143, 139, 135, 131, 128};
	unsigned char g[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179, 183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 251, 247, 243, 239, 235, 231, 227, 223, 219, 215, 211, 207, 203, 199, 195, 191, 187, 183, 179, 175, 171, 167, 163, 159, 155, 151, 147, 143, 139, 135, 131, 128, 124, 120, 116, 112, 108, 104, 100, 96, 92, 88, 84, 80, 76, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	unsigned char b[] = {131, 135, 139, 143, 147, 151, 155, 159, 163, 167, 171, 175, 179, 183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 251, 247, 243, 239, 235, 231, 227, 223, 219, 215, 211, 207, 203, 199, 195, 191, 187, 183, 179, 175, 171, 167, 163, 159, 155, 151, 147, 143, 139, 135, 131, 128, 124, 120, 116, 112, 108, 104, 100, 96, 92, 88, 84, 80, 76, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	rgb ret;
	ret.b = b[v];
	ret.g = g[v];
	ret.r = r[v];

	return ret;
}

void visualization::showPointCloud(void)
{
	float lim_max = 0;
	float lim_min = 9999999;

	if (data_available)
	{

		color_cloud->resize(clouds[0]->size());

		for (int l = 0; l < wavelengths; l++)
		{
			for (int i = 0; i < clouds[l]->size(); i++)
			{
				if (clouds[l]->at(i).intensity > lim_max)
					lim_max = clouds[l]->at(i).intensity;
				if (clouds[l]->at(i).intensity < lim_min)
					lim_min = clouds[l]->at(i).intensity;
			}
		}

		for (int i = 0; i < clouds[0]->size(); i++)
		{
			pcl::PointXYZRGB p;
			p.x = clouds[0]->at(i).x;
			p.y = clouds[0]->at(i).y;
			p.z = clouds[0]->at(i).z;
			std::vector<float> ref;
			for (int l = 0; l < wavelengths; l++)
			{
				ref.push_back(clouds[l]->at(i).intensity);
			}
			rgb color = map_reflectivity_to_color(ref, lim_max, lim_min);
			p.r = color.r;
			p.b = color.b;
			p.g = color.g;
			color_cloud->at(i) = p;
		}

		viewer->removePointCloud("cloud");
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> intensity(color_cloud);
		viewer->addPointCloud(color_cloud, intensity, "cloud");

		background_image->showRGBImage((unsigned char *)&img_bkg[0], Ncol, Nrow);
		points_image->showRGBImage((unsigned char *)&img_points_per_pix[0], height_cloud, width_cloud);
	}
	else
		std::cout << "The point cloud was not properly loaded" << std::endl;
}

visualization::rgb visualization::map_reflectivity_to_color(std::vector<float> &r, float lim_max, float lim_min)
{

	rgb color;

	if (wavelengths == 1)
	{
		unsigned char c = static_cast<unsigned char>(255. * (r[0] - lim_min) / (lim_max - lim_min));
		color = colormap(c);
	}
	else
	{
		color.b = static_cast<unsigned char>(255. * (r[0] - lim_min) / (lim_max - lim_min));
		color.g = static_cast<unsigned char>(255. * (r[1] - lim_min) / (lim_max - lim_min));
		color.r = static_cast<unsigned char>(255. * (r[3] - lim_min) / (lim_max - lim_min));
	}

	return color;
}

void visualization::keyboard_callback(const pcl::visualization::KeyboardEvent &event, void *viewer_void)
{
	if (event.keyDown() && event.getKeySym() == "i")
		finished_vis = true;
	else if (event.keyDown() && event.getKeySym() == "p")
	{
		till_the_end = true;
		finished_vis = true;
	}
}

void visualization::spinAll(void)
{

	if (first_frame)
	{
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
		viewer->setCameraPosition(-400., height_cloud / 2., width_cloud, 0.45, 0, 0.9);
		first_frame = false;
	}

	if (1)
	{ // till_the_end) {
		if (data_available)
		{
			background_image->spinOnce();
			points_image->spinOnce();
			viewer->spinOnce();

			if (plot_lhood)
				plotter->spinOnce();
		}
	}
	/*else {
		finished_vis = false;
		std::cout << "Please adjust the viewing as you desire and then press 'i' to continue (on any of the display windows)" << std::endl;
		std::cout << "Press 'p' if you are happy to keep this view until the end of the video" << std::endl;
		while (data_available && !finished_vis && !viewer->wasStopped()) {
			background_image->spinOnce();
			points_image->spinOnce();
			viewer->spinOnce();
			plotter->spinOnce();
		}
	}*/
}

int visualization::loadPointCloud(std::vector<float> &points, std::vector<float> &normals, std::vector<float> &reflect, std::vector<int> &in_points_per_pix,
								  float scale_ratio, int height, int width, std::vector<float> &background, int Nrow, int Ncol, int L, std::vector<float> &likelihood)
{

	// cloud.width = Nrow;
	// cloud.height = Nrow;
	wavelengths = L;

	clouds.clear();
	for (int l = 0; l < L; l++)
	{
		MyCloudType::Ptr cloud(new MyCloudType);
		clouds.push_back(cloud);
	}

	img_points_per_pix.resize(height * width);
	points_per_pix.resize(height * width);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int number_points = in_points_per_pix[i + j * height];

			points_per_pix[j + i * width] = number_points;
			img_points_per_pix[j + i * width] = colormap(static_cast<unsigned char>(32 * number_points));
			if (number_points > MAX_POINTS_PER_PIX || number_points < 0)
			{
				fprintf(stderr, "Error: %d points were found in pixel (%d,%d)\n", number_points, i, j);
				return -1;
			}
			for (int idx = 0; idx < number_points; idx++)
			{

				MyPointType point;

				MyPoint p = ReadPoint(i, j, idx, height, width, &normals[0], &points[0], &reflect[0], scale_ratio, wavelengths);

				point.normal[0] = p.normal()(2);
				point.normal[1] = -p.normal()(1);
				point.normal[2] = -p.normal()(0);

				point.x = p.pos()(2);
				point.y = width - p.pos()(1);
				point.z = height - p.pos()(0);
				point.z *= scale_ratio;
				point.y *= scale_ratio;
				point.x *= scale_ratio;

				for (int l = 0; l < wavelengths; l++)
				{
					point.intensity = p.r()(l);
					clouds[l]->push_back(point);
				}
			}
		}
	}

	height_cloud = width;
	width_cloud = height;

	this->Nrow = Nrow;
	this->Ncol = Ncol;
	img_bkg.resize(Nrow * Ncol);
	bkg.resize(Nrow * Ncol * L);

	float max_b = -99999., min_b = 99999.;

	for (int l = 0; l < L; l++)
	{
		for (int i = 0; i < Nrow; i++)
		{
			for (int j = 0; j < Ncol; j++)
			{
				int idx = get_bkg_idx(i, j, Nrow, Ncol, l);
				float b = background[idx];
				if (b < min_b)
					min_b = b;
				if (b > max_b)
					max_b = b;
				bkg[idx] = b;
			}
		}
	}

	// std::cout << "Bkg interval: (" << min_b << ","<< max_b << ")" << std::endl;

	// int clipped_values = -1;
	for (int i = 0; i < Nrow; i++)
	{
		for (int j = 0; j < Ncol; j++)
		{
			std::vector<float> b;

			for (int l = 0; l < L; l++)
			{
				b.push_back(bkg[get_bkg_idx(i, j, Nrow, Ncol, l)]);
			}

			img_bkg[j + i * Ncol] = map_reflectivity_to_color(b, max_b, min_b);
		}
	}

	// std::cout << "There were " << clipped_values << " clipped background pixels" << std::endl;

	std::cout << "The 3D reconstruction was performed succesfully!" << std::endl;
	std::cout << "Number of points: " << clouds[0]->size() << std::endl;
	data_available = true;

	if (render)
	{
		showPointCloud();
		spinAll();
		if (plot_lhood)
			plotLikelihood(likelihood);
	}
	return 0;
};

void visualization::savePly(const std::string &filename)
{
	bool binary = true;
	if (data_available)
	{
		pcl::PLYWriter writer;

		for (int l = 0; l < wavelengths; l++)
		{
			std::string fname(foldername + filename + std::string("_w") + std::to_string(l) + std::string(".ply"));
			writer.write(fname, *clouds[l], binary);
			std::cout << "file " << fname << " was saved." << std::endl;
		}
	}
}

template <typename T>
void saveImageBinary(const std::vector<T> &image, std::string &filename, const int Nrow, const int Ncol, const int wavelengths = 1)
{

	std::ofstream file(filename, ios::out | ios::binary);

	uint16_t d;
	d = wavelengths;
	file.write((const char *)&d, sizeof(uint16_t));
	d = Nrow;
	file.write((const char *)&d, sizeof(uint16_t));
	d = Ncol;
	file.write((const char *)&d, sizeof(uint16_t));
	for (int l = 0; l < wavelengths; l++)
	{
		for (int i = 0; i < Nrow; i++)
		{
			for (int j = 0; j < Ncol; j++)
			{
				int idx = get_bkg_idx(i, j, Nrow, Ncol, l);
				file.write((const char *)(&image[idx]), sizeof(T));
			}
		}
	}

	file.close();
}

void visualization::saveBackground(std::string &filename, bool binary)
{

	if (data_available)
	{

		std::string name;

		if (binary)
		{
			name = foldername + filename + std::string(".bkg");
			saveImageBinary(bkg, name, Nrow, Ncol, wavelengths);
		}
		else
		{
			name = foldername + filename + std::string(".png");
			pcl::io::saveRgbPNGFile(name, (unsigned char *)&img_bkg[0], Nrow, Ncol);
		}
		std::cout << "file " << name << " was saved." << std::endl;
	}
}

void visualization::saveLikelihood(std::string &filename, std::vector<float> &likelihood)
{

	if (data_available)
	{

		std::string name(foldername + filename + std::string(".txt"));
		std::ofstream file(name);

		for (int i = 0; i < likelihood.size(); i++)
		{
			file << likelihood[i] << std::endl;
		}
		file.close();

		std::cout << "file " << name << " was saved." << std::endl;
	}
}

void visualization::savePointsPerPix(std::string &filename, bool binary)
{

	if (data_available)
	{

		std::string name;
		if (binary)
		{
			name = foldername + filename + std::string(".ppp");
			saveImageBinary(points_per_pix, name, height_cloud, width_cloud);
		}
		else
		{
			name = foldername + filename + std::string(".png");
			pcl::io::saveRgbPNGFile(name, (unsigned char *)&img_points_per_pix[0], height_cloud, width_cloud);
		}
		std::cout << "file " << name << " was saved." << std::endl;
	}
}
