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
include ponca/tests/src/CMakeFiles/kdtree_nearest.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include ponca/tests/src/CMakeFiles/kdtree_nearest.dir/compiler_depend.make

# Include the progress variables for this target.
include ponca/tests/src/CMakeFiles/kdtree_nearest.dir/progress.make

# Include the compile flags for this target's objects.
include ponca/tests/src/CMakeFiles/kdtree_nearest.dir/flags.make

ponca/tests/src/CMakeFiles/kdtree_nearest.dir/kdtree_nearest.cpp.o: ponca/tests/src/CMakeFiles/kdtree_nearest.dir/flags.make
ponca/tests/src/CMakeFiles/kdtree_nearest.dir/kdtree_nearest.cpp.o: /home/qyh/real-time-sp-lidar/ponca/tests/src/kdtree_nearest.cpp
ponca/tests/src/CMakeFiles/kdtree_nearest.dir/kdtree_nearest.cpp.o: ponca/tests/src/CMakeFiles/kdtree_nearest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qyh/real-time-sp-lidar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object ponca/tests/src/CMakeFiles/kdtree_nearest.dir/kdtree_nearest.cpp.o"
	cd /home/qyh/real-time-sp-lidar/build/ponca/tests/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT ponca/tests/src/CMakeFiles/kdtree_nearest.dir/kdtree_nearest.cpp.o -MF CMakeFiles/kdtree_nearest.dir/kdtree_nearest.cpp.o.d -o CMakeFiles/kdtree_nearest.dir/kdtree_nearest.cpp.o -c /home/qyh/real-time-sp-lidar/ponca/tests/src/kdtree_nearest.cpp

ponca/tests/src/CMakeFiles/kdtree_nearest.dir/kdtree_nearest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kdtree_nearest.dir/kdtree_nearest.cpp.i"
	cd /home/qyh/real-time-sp-lidar/build/ponca/tests/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qyh/real-time-sp-lidar/ponca/tests/src/kdtree_nearest.cpp > CMakeFiles/kdtree_nearest.dir/kdtree_nearest.cpp.i

ponca/tests/src/CMakeFiles/kdtree_nearest.dir/kdtree_nearest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kdtree_nearest.dir/kdtree_nearest.cpp.s"
	cd /home/qyh/real-time-sp-lidar/build/ponca/tests/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qyh/real-time-sp-lidar/ponca/tests/src/kdtree_nearest.cpp -o CMakeFiles/kdtree_nearest.dir/kdtree_nearest.cpp.s

# Object files for target kdtree_nearest
kdtree_nearest_OBJECTS = \
"CMakeFiles/kdtree_nearest.dir/kdtree_nearest.cpp.o"

# External object files for target kdtree_nearest
kdtree_nearest_EXTERNAL_OBJECTS =

ponca/tests/src/kdtree_nearest: ponca/tests/src/CMakeFiles/kdtree_nearest.dir/kdtree_nearest.cpp.o
ponca/tests/src/kdtree_nearest: ponca/tests/src/CMakeFiles/kdtree_nearest.dir/build.make
ponca/tests/src/kdtree_nearest: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
ponca/tests/src/kdtree_nearest: /usr/lib/x86_64-linux-gnu/libpthread.so
ponca/tests/src/kdtree_nearest: ponca/tests/src/CMakeFiles/kdtree_nearest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/qyh/real-time-sp-lidar/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable kdtree_nearest"
	cd /home/qyh/real-time-sp-lidar/build/ponca/tests/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kdtree_nearest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ponca/tests/src/CMakeFiles/kdtree_nearest.dir/build: ponca/tests/src/kdtree_nearest
.PHONY : ponca/tests/src/CMakeFiles/kdtree_nearest.dir/build

ponca/tests/src/CMakeFiles/kdtree_nearest.dir/clean:
	cd /home/qyh/real-time-sp-lidar/build/ponca/tests/src && $(CMAKE_COMMAND) -P CMakeFiles/kdtree_nearest.dir/cmake_clean.cmake
.PHONY : ponca/tests/src/CMakeFiles/kdtree_nearest.dir/clean

ponca/tests/src/CMakeFiles/kdtree_nearest.dir/depend:
	cd /home/qyh/real-time-sp-lidar/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/qyh/real-time-sp-lidar /home/qyh/real-time-sp-lidar/ponca/tests/src /home/qyh/real-time-sp-lidar/build /home/qyh/real-time-sp-lidar/build/ponca/tests/src /home/qyh/real-time-sp-lidar/build/ponca/tests/src/CMakeFiles/kdtree_nearest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ponca/tests/src/CMakeFiles/kdtree_nearest.dir/depend

