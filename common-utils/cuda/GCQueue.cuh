//
// Created by Bryan on 21/11/2019.
//

/*The concurrent queue implementation on GPUs
 * Use of structure in the kernels requires that the struct allocated dynamically using cudaMallocManaged
 * todo: module the maximum size
 * */

#pragma once

#ifdef __JETBRAINS_IDE__
#include "../../common-utils/cuda/cuda_fake/fake.h"
#endif

#ifndef DEVICE_ID
#define DEVICE_ID (0)
#endif

#include "CUDAStat.cuh"
#define GCQUEUE_INVALID_DATA    (INT_MAX-1)

/*
 * Idea from the paper "A practical nonblocking queue algorithm using compare-and-swap"
 * DataType can only be 32-bit long
 * */
typedef unsigned long long int QType;

template<typename DataType, typename CntType>
struct GCQueue {
    QType *data;    //idx in the TaskBook

    /* volatile is necessary for qHead and qRear
     * since qHead and qRear are read and changed frequently
     * in enqueue and dequeue ops*/
    volatile CntType *qHead;
    volatile CntType *qRear;

    CntType capacity;
    CUDAMemStat *memstat;

    __device__ void enqueue(DataType val) {
        while (1) {
            auto rear = qRear[0];
            volatile auto x = data[rear]; //should be volatile to ensure reading the fresh data value
            if (rear != qRear[0]) continue;

            if (((x >> 32) == GCQUEUE_INVALID_DATA) &&
                (x == atomicCAS(&data[rear], x, ((QType)val << 32) | ((x&0xffffffff)+1)))) {
                auto a = atomicCAS((CntType *)qRear, rear, rear+1);
                return;
            }
        }
    }

    __device__ bool dequeue(DataType &val) {
        while (1) {
            auto head = qHead[0];
            auto x = data[head];
            if (head != qHead[0]) continue;
            if (head == qRear[0]) return false;

            if (((x >> 32) != GCQUEUE_INVALID_DATA) &&
                (x == atomicCAS(&data[head], x, ((QType)GCQUEUE_INVALID_DATA << 32) | ((x&0xffffffff)+1)))) {
                atomicCAS((CntType *)qHead, head, head+1);
                val = x >> 32;
                return true;
            }
        }
    }

    /*isEmpty in concurrent queue may not truly show whether the queue is empty*/
    __device__ bool isEmpty() {
        return (qHead[0] == qRear[0]);
    }

    void init(CntType capacity, CUDAMemStat *memStat) {
        this->capacity = capacity;
        this->memstat = memStat;

        CUDA_MALLOC(&data, sizeof(QType)*capacity, memStat);
        for(auto i = 0; i < capacity; i++)
            data[i] = ((QType)GCQUEUE_INVALID_DATA << 32) | 0;
        cudaMemPrefetchAsync(data, sizeof(QType)*capacity, DEVICE_ID);
        CUDA_MALLOC((void**)&qHead, sizeof(CntType), memStat);
        CUDA_MALLOC((void**)&qRear, sizeof(CntType), memStat);

        checkCudaErrors(cudaMemset((void*)qHead, 0, sizeof(CntType)));
        checkCudaErrors(cudaMemset((void*)qRear, 0, sizeof(CntType)));
    }

    void reset(cudaStream_t stream=0) {
        checkCudaErrors(cudaMemsetAsync((void*)qHead, 0, sizeof(CntType), stream));
        checkCudaErrors(cudaMemsetAsync((void*)qRear, 0, sizeof(CntType), stream));
    }

    size_t get_size() {
        return sizeof(QType)*capacity + sizeof(CntType)*3 + sizeof(CUDAMemStat*);
    }

    void clear() {
        cudaDeviceSynchronize(); //need to synchronize
        CUDA_FREE(data, memstat);
        CUDA_FREE((void*)qHead, memstat);
        CUDA_FREE((void*)qRear, memstat);
    }
};