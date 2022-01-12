cmake_minimum_required(VERSION "3.19")

project(real-time-lidar)

# Also, enable CUDA language support.
enable_language(CUDA)


find_package(PCL 1.8.1)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories(${EIGEN3_INCLUDE_DIR})


add_subdirectory(ponca)
set(PONCA_CONFIGURE_EXAMPLES BOOL OFF)
set(PONCA_CONFIGURE_TESTS    BOOL OFF)
set(PONCA_CONFIGURE_DOC      BOOL OFF)
include_directories(${PROJECT_SOURCE_DIR}/ponca)

add_executable(test main.cpp algorithm.cu algorithm.h apss.cu backgroundRegularization.cu backgroundRegularization.h GPU_macros.h gradient.cu gradient.cuh logmatch.cu logmatch.cuh misc.cpp misc.h performance.cpp performance.h read_lidar.cpp read_lidar.h visualization.cpp visualization.h)
target_link_libraries(test ${PCL_LIBRARIES})