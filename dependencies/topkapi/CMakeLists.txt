cmake_minimum_required(VERSION 3.9)
project(topkapi_project)

file(GLOB_RECURSE MY_FILES ${PROJECT_SOURCE_DIR}/*)

#find_package(OpenMP)
#if(OPENMP_FOUND)
#    set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#endif()
#
#add_library(topkapi SHARED ${MY_FILES})