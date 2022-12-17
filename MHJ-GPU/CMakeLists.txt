cmake_minimum_required(VERSION 2.8)
project(hash-join-chain)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-std=c++11")

set(MAIN_DIR ${CMAKE_SOURCE_DIR}/main)

include_directories(../dependencies/moderngpu/src)
include_directories(../dependencies/cub/)
include_directories(../dependencies/topkapi)
include_directories(../dependencies/cudpp/include)
include_directories(../common-utils)

# Add all the main files automatically
file(GLOB_RECURSE EXEC_FILES ${MAIN_DIR}/*)

find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

find_package(CUDA)
if (CUDA_FOUND)
    message("--- CUDA Found, Toolkit Path: ${CUDA_TOOLKIT_ROOT_DIR}/")
    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc) #for helper_cuda.h

    if (USE_ARCH)
        message(STATUS "User has specified ARCH: ${USE_ARCH}")
        set(ARCH -arch=${USE_ARCH})
    else() # checking the default arch of the current platform
        set(OUTPUTFILE ./cuda_script)
        set(CUDAFILE ${CMAKE_SOURCE_DIR}/check_cuda.cu)
        execute_process(COMMAND nvcc -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
        execute_process(COMMAND ${OUTPUTFILE}
                RESULT_VARIABLE CUDA_RETURN_CODE
                OUTPUT_VARIABLE ARCH)

        if(${CUDA_RETURN_CODE} EQUAL 0)
            set(CUDA_SUCCESS "TRUE")
        else()
            set(CUDA_SUCCESS "FALSE")
        endif()

        if (${CUDA_SUCCESS})
            message(STATUS "CUDA Default Architecture: ${ARCH}")
            message(STATUS "CUDA Version: ${CUDA_VERSION_STRING}")
            message(STATUS "CUDA Path: ${CUDA_TOOLKIT_ROOT_DIR}")
            message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
            message(STATUS "CUDA Performance Primitives: ${CUDA_npp_LIBRARY}")
        else()
            message(WARNING ${ARCH})
        endif()
    endif ()

    message(STATUS "Architecture specified: ${ARCH}")

    macro(cuda_add_compile_options options)
        list(APPEND CUDA_NVCC_FLAGS "${options}")
    endmacro()
    cuda_add_compile_options("-std=c++11 -O3 -g -lineinfo ${ARCH}")
    cuda_add_compile_options("--expt-extended-lambda -Wno-deprecated-gpu-targets")
    cuda_add_compile_options("-DPROJECT_DIR=\"${CMAKE_SOURCE_DIR}\" -DUSE_LOG -DDATA_ADDR=\"${CMAKE_SOURCE_DIR}/../datasets\"")
    cuda_add_compile_options("-Xcompiler=\"-fopenmp\"")
    cuda_add_compile_options("-Xcompiler=\"-Wno-deprecated\"")
    cuda_add_compile_options("-DDEVICE_ID=1")           #set device id
    cuda_add_compile_options("-DFREE_DATA")            #will free unused data, but add time cost
    cuda_add_compile_options("-DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT") #ignore the CUDA 11 Thrust warning for c++11

    #cuda_add_compile_options("-Xptxas -v")         #list the usage of registers and memory
    #cuda_add_compile_options("-DDEBUG")            #debug mode

    set(CUDA_SOURCE_FILES
            ../dependencies/topkapi/topkapi.cpp
            ../dependencies/topkapi/LossyCountMinSketch.cpp
            ../dependencies/topkapi/MurmurHash3.cpp
            ../dependencies/topkapi/HashFunction.cpp
            ../common-utils/md5.cpp)

    #link CUDPP library
    include_directories(../dependencies/cudpp/apps/common/include ../dependencies/cudpp/apps/cudpp_hash_testrig/)
    link_directories(${CMAKE_SOURCE_DIR}/../dependencies/cudpp/build/lib)

    cuda_add_executable(exec-MHJ exec_MHJ.cu ${CUDA_SOURCE_FILES}) # MHJ execution
    cuda_add_executable(exec-AMHJ exec_AMHJ.cu ${CUDA_SOURCE_FILES}) # AMHJ execution
    cuda_add_executable(exec-MHJ-oa exec_MHJ_open_addr.cu ${CUDA_SOURCE_FILES}) # MHJ_linear execution
    cuda_add_executable(exec-AMHJ-oa exec_AMHJ_open_addr.cu ${CUDA_SOURCE_FILES}) # AMHJ_linear execution
    cuda_add_executable(exec-MHJ-cuckoo exec_MHJ_cuckoo.cu ${CUDA_SOURCE_FILES}) # MHJ_cuckoo execution
    cuda_add_executable(exec-AMHJ-cuckoo exec_AMHJ_cuckoo.cu ${CUDA_SOURCE_FILES}) # AMHJ_cuckoo execution

    cuda_add_executable(exec-PW exec_PW.cu ${CUDA_SOURCE_FILES}) # pairwise join execution
    cuda_add_executable(exec-LFTJ exec_LFTJ.cu ${CUDA_SOURCE_FILES}) # LFTJ execution
    cuda_add_executable(uniform_writer uniform_writer_.cu ${CUDA_SOURCE_FILES}) #uniform data generator
    cuda_add_executable(skewed_writer skewed_writer_.cu ${CUDA_SOURCE_FILES}) #skewed data generator
    cuda_add_executable(tbl_2_db exec_tbl_2_db.cu) #.tbl file tranfer to .db file
    cuda_add_executable(tbl_2_csv exec_tbl_2_csv.cu) #.tbl file tranfer to .csv file
    cuda_add_executable(edgelist_2_db exec_el_2_db.cu) #.edgelist file tranfer to .db file

    cuda_add_executable(exec-partition exec_partition.cu ${CUDA_SOURCE_FILES}) #test different hash table
    target_link_libraries(exec-partition cudpp cudpp_hash) #link with the CUDPP libraries

    #test radix partitioning with different partition settings
    cuda_add_executable(exec-iterate-partitions exec_iterate_partitions.cu ${CUDA_SOURCE_FILES})
endif()

