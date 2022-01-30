cmake_minimum_required(VERSION "3.19")

project(real-time-lidar)

# Also, enable CUDA language support.
find_package(CUDA 9.2 REQUIRED)

enable_language(CUDA)

find_package(PCL 1.8.1)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_subdirectory(ponca)

add_executable(real-time-lidar main.cpp algorithm.cu algorithm.h apss.cu apss.h backgroundRegularization.cu backgroundRegularization.h GPU_macros.h gradient.cu gradient.h logmatch.cu logmatch.h misc.cpp misc.h performance.cpp performance.h programs.cpp programs.h read_lidar.cpp read_lidar.h Point.h visualization.cpp visualization.h)

target_link_libraries(real-time-lidar ${PCL_LIBRARIES} 
				${CUDA_cufft_LIBRARY}
			 )

target_include_directories(real-time-lidar PUBLIC
                                ${CMAKE_CURRENT_SOURCE_DIR}/ponca
                                ${EIGEN3_INCLUDE_DIR}
                          )

set(PONCA_CONFIGURE_EXAMPLES FALSE)
set(PONCA_CONFIGURE_TESTS FALSE)
set(PONCA_CONFIGURE_DOC FALSE)