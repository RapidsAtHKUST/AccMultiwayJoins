//
// Created by Bryan on 19/11/2018.
//
#pragma once

#ifdef __JETBRAINS_IDE__
#include "cuda_fake/fake.h"
#endif

#include <iostream>
#include "log.h"

/*version related macros*/
#define WARP_REDUCE(var)    { \
                                var += __shfl_down_sync(0xFFFFFFFF, var, 16);\
                                var += __shfl_down_sync(0xFFFFFFFF, var, 8);\
                                var += __shfl_down_sync(0xFFFFFFFF, var, 4);\
                                var += __shfl_down_sync(0xFFFFFFFF, var, 2);\
                                var += __shfl_down_sync(0xFFFFFFFF, var, 1);\
                            }

/*CUDA check error*/
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

/*
 * cudaPeekAtLastError(): get the code of last error, no resetting
 * udaGetLastError(): get the code of last error, resetting to cudaSuccess
 * */
#define CHECK_KERNEL(func)          if (cudaSuccess != cudaPeekAtLastError()) { \
                                        cudaError_t error = cudaGetLastError(); \
                                        log_fatal("Kernel %s: %s.", func, \
                                        cudaGetErrorString(error)); \
                                        exit(1); \
                                    }

#define SIN_L(inst)          {                   \
                                if (0 == lane) { \
                                    inst;        \
                                }                \
                                __syncwarp();    \
                             }

inline void check(
    cudaError_t code, char const *const func, const char *const file, int const line, bool abort=true) {
    if (code != cudaSuccess) {
        if (abort) {
            log_fatal("CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                      static_cast<unsigned int>(code), cudaGetErrorString(code), func);
            // Make sure we call CUDA Device Reset before exiting
            cudaDeviceReset();
            exit(static_cast<unsigned int>(code));
        } else {
            log_warn("CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                     static_cast<unsigned int>(code), cudaGetErrorString(code), func);
        }
    }
}

/*macro for non-templated kernel launch*/
/*normal execution without dynamic shared memory allocation*/
#define execKernel(kernel, gridSize, blockSize, timing, verbose, ...) \
{ \
    float singleKernelTime;\
    cudaEvent_t start, end; \
    checkCudaErrors(cudaEventCreate(&start)); \
    checkCudaErrors(cudaEventCreate(&end)); \
    dim3 grid(gridSize); \
    dim3 block(blockSize); \
    \
    checkCudaErrors(cudaEventRecord(start)); \
    kernel<<<grid,block>>>(__VA_ARGS__); \
    CHECK_KERNEL(#kernel)\
    checkCudaErrors(cudaEventRecord(end));\
    \
    checkCudaErrors(cudaEventSynchronize(start)); \
    checkCudaErrors(cudaEventSynchronize(end)); \
    checkCudaErrors(cudaDeviceSynchronize()); \
    checkCudaErrors(cudaEventElapsedTime(&singleKernelTime, start, end)); \
    \
    if (timing != nullptr)\
    {\
        if(verbose) log_info("Kernel: %s, time: %.2f ms.",#kernel, singleKernelTime); \
        timing->insert_record(__FILE__, __FUNCTION__, #kernel, singleKernelTime);\
    }\
}

#define execKernelWithStream(kernel, gridSize, blockSize, stream, timing, verbose, ...) \
{ \
    float singleKernelTime;\
    cudaEvent_t start, end; \
    checkCudaErrors(cudaEventCreate(&start)); \
    checkCudaErrors(cudaEventCreate(&end)); \
    dim3 grid(gridSize); \
    dim3 block(blockSize); \
    \
    checkCudaErrors(cudaEventRecord(start, stream)); \
    kernel<<<grid,block, 0, stream>>>(__VA_ARGS__); \
    CHECK_KERNEL(#kernel)\
    checkCudaErrors(cudaEventRecord(end, stream));\
    \
    checkCudaErrors(cudaEventSynchronize(start)); \
    checkCudaErrors(cudaEventSynchronize(end)); \
    checkCudaErrors(cudaStreamSynchronize(stream)); \
    checkCudaErrors(cudaEventElapsedTime(&singleKernelTime, start, end)); \
    \
    if (timing != nullptr)\
    {\
        if(verbose) log_info("Kernel: %s, time: %.2f ms.",#kernel, singleKernelTime); \
        timing->insert_record(__FILE__, __FUNCTION__, #kernel, singleKernelTime);\
    }\
}

/*execution with dynamic shared memory allocation*/
#define execKernelDynamicAllocation(kernel, gridSize, blockSize, sharedSize, timing, verbose, ...) \
{ \
    float singleKernelTime;\
    cudaEvent_t start, end; \
    checkCudaErrors(cudaEventCreate(&start)); \
    checkCudaErrors(cudaEventCreate(&end)); \
    dim3 grid(gridSize); \
    dim3 block(blockSize); \
    \
    checkCudaErrors(cudaEventRecord(start)); \
    kernel<<<grid,block, sharedSize>>>(__VA_ARGS__); \
    CHECK_KERNEL(#kernel); \
    checkCudaErrors(cudaPeekAtLastError());\
    checkCudaErrors(cudaEventRecord(end));\
    \
    checkCudaErrors(cudaEventSynchronize(start)); \
    checkCudaErrors(cudaEventSynchronize(end)); \
    checkCudaErrors(cudaDeviceSynchronize()); \
    checkCudaErrors(cudaEventElapsedTime(&singleKernelTime, start, end)); \
    \
    if (timing != nullptr)\
    {\
        if(verbose) log_info("Kernel: %s, time: %.2f ms.",#kernel, singleKernelTime); \
        timing->insert_record(__FILE__, __FUNCTION__, #kernel, singleKernelTime);\
    }\
}

/*execution with dynamic shared memory allocation*/
#define execKernelDynamicAllocationWithStream(kernel, gridSize, blockSize, sharedSize, stream, timing, verbose, ...) \
{ \
    float singleKernelTime;\
    cudaEvent_t start, end; \
    checkCudaErrors(cudaEventCreate(&start)); \
    checkCudaErrors(cudaEventCreate(&end)); \
    dim3 grid(gridSize); \
    dim3 block(blockSize); \
    \
    checkCudaErrors(cudaEventRecord(start, stream)); \
    kernel<<<grid,block, sharedSize, stream>>>(__VA_ARGS__); \
    CHECK_KERNEL(#kernel); \
    checkCudaErrors(cudaPeekAtLastError());\
    checkCudaErrors(cudaEventRecord(end, stream));\
    \
    checkCudaErrors(cudaEventSynchronize(start)); \
    checkCudaErrors(cudaEventSynchronize(end)); \
    checkCudaErrors(cudaStreamSynchronize(stream)); \
    checkCudaErrors(cudaEventElapsedTime(&singleKernelTime, start, end)); \
    \
    if (timing != nullptr)\
    {\
        if(verbose) log_info("Kernel: %s, time: %.2f ms.",#kernel, singleKernelTime); \
        timing->insert_record(__FILE__, __FUNCTION__, #kernel, singleKernelTime);\
    }\
}

/*for those third-party library functions*/
#define timingKernel(kernel, timing) \
{ \
    float myTime = 0.0f;\
    cudaEvent_t start, end; \
    checkCudaErrors(cudaEventCreate(&start)); \
    checkCudaErrors(cudaEventCreate(&end)); \
    checkCudaErrors(cudaEventRecord(start)); \
    kernel; \
    checkCudaErrors(cudaEventRecord(end));\
    \
    checkCudaErrors(cudaEventSynchronize(start)); \
    checkCudaErrors(cudaEventSynchronize(end)); \
    checkCudaErrors(cudaDeviceSynchronize()); \
    checkCudaErrors(cudaEventElapsedTime(&myTime, start, end)); \
    \
    if (timing != nullptr)\
    {\
        timing->insert_record(__FILE__, __FUNCTION__, #kernel, myTime);\
    }\
}



