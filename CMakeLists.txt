cmake_minimum_required(VERSION "3.19")

project(real-time-lidar)

# Enable CUDA language support
find_package(CUDA 11.6 REQUIRED)
enable_language(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})                                        

# Find and include PCL
find_package(PCL 1.8.1 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add ponca subdirectory
add_subdirectory(ponca)

# Add the executable
add_executable(real-time-lidar
    main.cpp
    algorithm.cu algorithm.h
    apss.cu apss.h
    backgroundRegularization.cu backgroundRegularization.h
    GPU_macros.h
    gradient.cu gradient.h
    logmatch.cu logmatch.h
    misc.cpp misc.h
    performance.cpp performance.h
    programs.cpp programs.h
    read_lidar.cpp read_lidar.h
    Point.h
    visualization.cpp visualization.h
)


# Add -fPIC only for C++ code
target_compile_options(real-time-lidar PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
)

# Add -Xcompiler -fPIC for CUDA code to allow its passing to nvcc
target_compile_options(real-time-lidar PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC>
)


# Link the executable to PCL and CUDA libraries
target_link_libraries(real-time-lidar ${PCL_LIBRARIES} 
    ${CUDA_cufft_LIBRARY}
)

# Add include directories
target_include_directories(real-time-lidar PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/ponca
    ${EIGEN3_INCLUDE_DIR}
)

# Disable examples, tests, and docs for Ponca
set(PONCA_CONFIGURE_EXAMPLES FALSE)
set(PONCA_CONFIGURE_TESTS FALSE)
set(PONCA_CONFIGURE_DOC FALSE)


