# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/qyh/real-time-sp-lidar

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/qyh/real-time-sp-lidar/build

# Include any dependencies generated for this target.
include ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/compiler_depend.make

# Include the progress variables for this target.
include ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/progress.make

# Include the compile flags for this target's objects.
include ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/flags.make

ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/ponca_basic_cpu.cpp.o: ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/flags.make
ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/ponca_basic_cpu.cpp.o: /home/qyh/real-time-sp-lidar/ponca/examples/cpp/ponca_basic_cpu.cpp
ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/ponca_basic_cpu.cpp.o: ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qyh/real-time-sp-lidar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/ponca_basic_cpu.cpp.o"
	cd /home/qyh/real-time-sp-lidar/build/ponca/examples/cpp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/ponca_basic_cpu.cpp.o -MF CMakeFiles/ponca_basic_cpu.dir/ponca_basic_cpu.cpp.o.d -o CMakeFiles/ponca_basic_cpu.dir/ponca_basic_cpu.cpp.o -c /home/qyh/real-time-sp-lidar/ponca/examples/cpp/ponca_basic_cpu.cpp

ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/ponca_basic_cpu.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ponca_basic_cpu.dir/ponca_basic_cpu.cpp.i"
	cd /home/qyh/real-time-sp-lidar/build/ponca/examples/cpp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qyh/real-time-sp-lidar/ponca/examples/cpp/ponca_basic_cpu.cpp > CMakeFiles/ponca_basic_cpu.dir/ponca_basic_cpu.cpp.i

ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/ponca_basic_cpu.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ponca_basic_cpu.dir/ponca_basic_cpu.cpp.s"
	cd /home/qyh/real-time-sp-lidar/build/ponca/examples/cpp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qyh/real-time-sp-lidar/ponca/examples/cpp/ponca_basic_cpu.cpp -o CMakeFiles/ponca_basic_cpu.dir/ponca_basic_cpu.cpp.s

# Object files for target ponca_basic_cpu
ponca_basic_cpu_OBJECTS = \
"CMakeFiles/ponca_basic_cpu.dir/ponca_basic_cpu.cpp.o"

# External object files for target ponca_basic_cpu
ponca_basic_cpu_EXTERNAL_OBJECTS =

ponca/examples/cpp/ponca_basic_cpu: ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/ponca_basic_cpu.cpp.o
ponca/examples/cpp/ponca_basic_cpu: ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/build.make
ponca/examples/cpp/ponca_basic_cpu: ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/qyh/real-time-sp-lidar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ponca_basic_cpu"
	cd /home/qyh/real-time-sp-lidar/build/ponca/examples/cpp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ponca_basic_cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/build: ponca/examples/cpp/ponca_basic_cpu
.PHONY : ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/build

ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/clean:
	cd /home/qyh/real-time-sp-lidar/build/ponca/examples/cpp && $(CMAKE_COMMAND) -P CMakeFiles/ponca_basic_cpu.dir/cmake_clean.cmake
.PHONY : ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/clean

ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/depend:
	cd /home/qyh/real-time-sp-lidar/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/qyh/real-time-sp-lidar /home/qyh/real-time-sp-lidar/ponca/examples/cpp /home/qyh/real-time-sp-lidar/build /home/qyh/real-time-sp-lidar/build/ponca/examples/cpp /home/qyh/real-time-sp-lidar/build/ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ponca/examples/cpp/CMakeFiles/ponca_basic_cpu.dir/depend

