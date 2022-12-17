//
// Created by Bryan on 9/2/2020.
//

#pragma once

#ifdef __JETBRAINS_IDE__
#include "../../common-utils/cuda/cuda_fake/fake.h"
#endif

#include <iostream>
#include "cuda/CUDAStat.cuh"

#ifndef DEVICE_ID
#define DEVICE_ID (1)
#endif

using bsize_t = unsigned long long;
using AttrType = uint32_t;  //indicating the attrs
using RelType = uint32_t;   //indicating the relations

/*data distribution type*/
enum Dis_type {
    UNIFORM, SKEWED
};

/*Algorithm type of LFTJ*/
enum LFTJ_type {
    TYPE_BFS, TYPE_DFS
};

template<typename DataType, typename CntType, typename RelType>
struct AttrDataMapping {
    CntType num;             //number of relations this attribute is in
    RelType *rel;            //the relations the data are in, [0, num)
    uint32_t *levels;        //the levels in the relations the data are in, [0, num)
    DataType **data;         //[0,num)[0,data_len)
    CntType *data_len;       //length of the data, [0, num)
    CntType **buc_ptrs;      //hash bucket ptrs, only for hash-based LFTJ
    uint32_t *next_se_idx;   //[0, num), next start&end index in the same table
    CUDAMemStat *memstat;

    void init(CntType num, CUDAMemStat *memstat) {
        this->num = num;
        this->memstat = memstat;
        CUDA_MALLOC(&rel, sizeof(RelType)*num, memstat);
        CUDA_MALLOC(&levels, sizeof(uint32_t)*num, memstat);
        CUDA_MALLOC(&data, sizeof(DataType*)*num, memstat);
        CUDA_MALLOC(&data_len, sizeof(CntType)*num, memstat);
        CUDA_MALLOC(&buc_ptrs, sizeof(CntType*)*num, memstat);
        CUDA_MALLOC(&next_se_idx, sizeof(uint32_t)*num, memstat);
    }

    void clear() {
        CUDA_FREE(rel, memstat);
        CUDA_FREE(levels, memstat);
        CUDA_FREE(data, memstat);
        CUDA_FREE(data_len, memstat);
        CUDA_FREE(buc_ptrs, memstat);
        CUDA_FREE(next_se_idx, memstat);
    }
};

/*Task description for work-sharing*/
template<typename DataType, typename CntType, typename BookCntType, int MAX_REL_PER_ATTR=2>
struct LFTJTaskBook {
    /*
     * intersect
     * small_array[small_range_start,small_range_end)
     * and
     * larger_array[large_range_start,large_range_end)
     * */
    volatile CntType *rel_0_start;
    volatile CntType *rel_0_end;
    volatile CntType *rel_1_start;
    volatile CntType *rel_1_end;

    volatile CntType *iterators;   //iterators of each task
    volatile char *cur_attr;         //idx of the attribute currently intersected
    volatile DataType *iRes;         //storing the iRes data of each task

    BookCntType capacity;               //capacity of the task book
    BookCntType *cnt;                   //current count of the tasks
    int res_len;                    //length of each output tuple
    CUDAMemStat *memstat;

    void init(BookCntType cap, int res_len, CUDAMemStat *mem_stat) {
        this->capacity = cap;
        this->memstat = mem_stat;
        this->res_len = res_len;

        /*linear data store*/
        CUDA_MALLOC((void**)&iterators, sizeof(CntType)*(capacity*MAX_REL_PER_ATTR*res_len), mem_stat);
        CUDA_MALLOC((void**)&iRes, sizeof(DataType)*(capacity*res_len), mem_stat);
        CUDA_MALLOC((void**)&rel_0_start, sizeof(CntType)*this->capacity, mem_stat);
        CUDA_MALLOC((void**)&rel_0_end, sizeof(CntType)*this->capacity, mem_stat);
        CUDA_MALLOC((void**)&rel_1_start, sizeof(CntType)*this->capacity, mem_stat);
        CUDA_MALLOC((void**)&rel_1_end, sizeof(CntType)*this->capacity, mem_stat);
        CUDA_MALLOC((void**)&cur_attr, sizeof(char)*this->capacity, mem_stat);
        CUDA_MALLOC((void**)&cnt, sizeof(BookCntType), mem_stat);
        cnt[0] = 0;
    }

    /*return the task id (position)*/
    __device__ CntType push_task(
            CntType rel_0_start, CntType rel_0_end,
            CntType rel_1_start, CntType rel_1_end,
            CntType *iter, DataType *ires, char cur_attr) {
        auto cur_idx = atomicAdd(cnt, 1);
        this->rel_0_start[cur_idx] = rel_0_start;
        this->rel_0_end[cur_idx] = rel_0_end;
        this->rel_1_start[cur_idx] = rel_1_start;
        this->rel_1_end[cur_idx] = rel_1_end;
        this->cur_attr[cur_idx] = cur_attr;
        auto offset = cur_idx*MAX_REL_PER_ATTR*res_len;
        for(auto i = 0; i < MAX_REL_PER_ATTR*res_len; i++) {
            iterators[offset+i] = iter[i];
        }
        if (ires != nullptr) {
            offset = cur_idx * res_len;
            for(auto i = 0; i < res_len; i++) {
                iRes[offset+i] = ires[i];
            }
        }
        return cur_idx;
    }

    void reset(cudaStream_t stream=0) {
        checkCudaErrors(cudaMemsetAsync(cnt, 0, sizeof(BookCntType), stream));
    }

    void clear() {
        cudaDeviceSynchronize();
        CUDA_FREE((void*)iterators, memstat);
        CUDA_FREE((void*)iRes, memstat);
        CUDA_FREE((void*)cnt, memstat);
        CUDA_FREE((void*)cur_attr, memstat);
        CUDA_FREE((void*)rel_0_start, memstat);
        CUDA_FREE((void*)rel_0_end, memstat);
        CUDA_FREE((void*)rel_1_start, memstat);
        CUDA_FREE((void*)rel_1_end, memstat);
    }
};


