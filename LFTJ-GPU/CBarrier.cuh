//
// Created by Bryan on 18/4/2020.
//

#pragma once

#include "types.h"

struct CBarrier {
    volatile int *numWarpsSync;
    int size;
    CUDAMemStat *memstat;

    __device__ void setActive(bool flag) {
        if (flag) atomicSub((int *)numWarpsSync, 1);
        else        atomicAdd((int *)numWarpsSync, 1);
    }

    __device__ bool isTerminated() {
        return (numWarpsSync[0] >= size);
    }

    void initWithWarps(int numWarps, CUDAMemStat *stat) {
        memstat = stat;
        CUDA_MALLOC((void**)&numWarpsSync, sizeof(int), memstat);
        numWarpsSync[0] = 0;
        size = numWarps;
    }

    void reset(cudaStream_t stream=0) {
        checkCudaErrors(cudaMemsetAsync((void*)numWarpsSync, 0, sizeof(int), stream));
    }

    void reset_with_warps(int num_warps, cudaStream_t stream=0) {
        checkCudaErrors(cudaMemsetAsync((void*)numWarpsSync, 0, sizeof(int), stream));
        size = num_warps;
    }

    size_t get_size() {
        return sizeof(int)*2 + sizeof(CUDAMemStat*);
    }

    void clear() {
        cudaDeviceSynchronize();
        CUDA_FREE((void*)numWarpsSync, memstat);
    }
};