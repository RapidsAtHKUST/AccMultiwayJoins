//
// Created by Bryan on 8/7/2020.
//

#pragma once

#include "types.h"

template<typename DataType, typename CntType>
struct BreakPoints {
    uint32_t _num_bps;    //number of breakpoints
    CntType *_iters;      //storing the iterator information
    CntType *_buc_ends;   //storing the current buc ends
    DataType *_iRes;      //storing the iRes information
    CntType *_head;       //pointing to the first breakpoint
    char *_start_tables;  //the start table each breakpoint has

    CUDAMemStat *_memstat;

    void init(uint32_t num_bps, int iter_width, int iRes_width, CUDAMemStat *memstat) {
        _num_bps = num_bps;
        _memstat = memstat;
        CUDA_MALLOC(&_iters, sizeof(CntType)*_num_bps*iter_width, _memstat);
        CUDA_MALLOC(&_buc_ends, sizeof(CntType)*_num_bps*(iter_width-1), _memstat); //todo : number of hash tables

        CUDA_MALLOC(&_iRes, sizeof(DataType)*_num_bps*iRes_width, _memstat);
        CUDA_MALLOC(&_head, sizeof(CntType), _memstat);
        CUDA_MALLOC(&_start_tables, sizeof(char)*_num_bps, _memstat);
        checkCudaErrors(cudaMemset(_head, 0, sizeof(CntType)));
    }

    void destroy() {
        CUDA_FREE(_iters, _memstat);
        CUDA_FREE(_buc_ends, _memstat);
        CUDA_FREE(_iRes, _memstat);
        CUDA_FREE(_head, _memstat);
        CUDA_FREE(_start_tables, _memstat);
    }

    void reset(cudaStream_t stream=0) {
        checkCudaErrors(cudaMemsetAsync(_head, 0, sizeof(CntType), stream));
    }
};