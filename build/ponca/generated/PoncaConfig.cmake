# - Config file for the Ponca package
# It defines the following variables
#  Ponca_INCLUDE_DIR - include directorie for Ponca
#
# To use
# ::
# find_package(Ponca REQUIRED)
# add_executable(foo foo.cc)
# target_link_libraries(foo Ponca::Ponca)



####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was Config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)
find_dependency(Eigen3 REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/PoncaTargets-Fitting.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/PoncaTargets-Common.cmake")


# Compute paths
set(Ponca_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include/")
set(Ponca_FOUND 1)
