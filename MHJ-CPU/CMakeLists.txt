cmake_minimum_required(VERSION 2.8)
project(GPU_operators_CPU)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ltbb")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread")

include_directories(../common-utils)

# Add all the header files automatically
file(GLOB_RECURSE HEADERS ${CMAKE_SOURCE_DIR}/*.h)

add_compile_options("-DUSE_LOG")
add_executable(exec-MHJ MHJ.cpp ${HEADERS})

set_target_properties(exec-MHJ PROPERTIES COMPILE_FLAGS -pthread LINK_FLAGS -pthread)


