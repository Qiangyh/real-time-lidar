#include <iostream>
#include <string>
#include "programs.h"


void print_info(void) {
	std::cout << "Publications associated with this program:" << std::endl;
	std::cout << "J. Tachella et al., ''Real-time 3D reconstruction from single-photon lidar data using plug-and-play-denoisers'', Nature Communications, vol. 10, pp. 4984, November 2019" << std::endl;
	std::cout << "J. Tachella et al., ''Real-time color 3D reconstruction from single-photon lidar data'', in Proc. International Workshop on Computational Advances in Multi Sensor Adaptive Processing (CAMSAP), Guadaloupe, West Indies, December 2019" << std::endl;

}
void print_copyright(void) {
	std::cout << "Author: Julian Tachella, Heriot-Watt University, Edinburgh, UK. TeSA Laboratory, Toulouse, France. Contact: jat3@hw.ac.uk // tachella.github.io" << std::endl << std::endl;
}

int main()
{
	print_info();
	print_copyright();

	//run_all();
	run_single();

	std::cout << "Press enter to exit...";
	std::string s;
	std::getline(std::cin, s);
    return 0;
}

