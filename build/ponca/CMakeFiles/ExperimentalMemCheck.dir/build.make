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

# Utility rule file for ExperimentalMemCheck.

# Include any custom commands dependencies for this target.
include ponca/CMakeFiles/ExperimentalMemCheck.dir/compiler_depend.make

# Include the progress variables for this target.
include ponca/CMakeFiles/ExperimentalMemCheck.dir/progress.make

ponca/CMakeFiles/ExperimentalMemCheck:
	cd /home/qyh/real-time-sp-lidar/build/ponca && /usr/local/bin/ctest -D ExperimentalMemCheck

ExperimentalMemCheck: ponca/CMakeFiles/ExperimentalMemCheck
ExperimentalMemCheck: ponca/CMakeFiles/ExperimentalMemCheck.dir/build.make
.PHONY : ExperimentalMemCheck

# Rule to build all files generated by this target.
ponca/CMakeFiles/ExperimentalMemCheck.dir/build: ExperimentalMemCheck
.PHONY : ponca/CMakeFiles/ExperimentalMemCheck.dir/build

ponca/CMakeFiles/ExperimentalMemCheck.dir/clean:
	cd /home/qyh/real-time-sp-lidar/build/ponca && $(CMAKE_COMMAND) -P CMakeFiles/ExperimentalMemCheck.dir/cmake_clean.cmake
.PHONY : ponca/CMakeFiles/ExperimentalMemCheck.dir/clean

ponca/CMakeFiles/ExperimentalMemCheck.dir/depend:
	cd /home/qyh/real-time-sp-lidar/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/qyh/real-time-sp-lidar /home/qyh/real-time-sp-lidar/ponca /home/qyh/real-time-sp-lidar/build /home/qyh/real-time-sp-lidar/build/ponca /home/qyh/real-time-sp-lidar/build/ponca/CMakeFiles/ExperimentalMemCheck.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ponca/CMakeFiles/ExperimentalMemCheck.dir/depend

