//
// Created by zlai on 11/5/19.
//
#pragma once

#include <iostream>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <moderngpu/kernel_segsort.hxx>
#include <moderngpu/kernel_load_balance.hxx>
#include <cstdlib>

#include "../types.h"
#include "../conf.h"
#include "cuda/primitives.cuh"
#include "cuda/CUDAStat.cuh"
#include "helper.h"
#include "pretty_print.h"
#include "timer.h"

using namespace std;
using namespace mgpu;

template<typename DataType>
struct ZipComparator_2
{
    __host__ __device__
    inline bool operator() (
            const thrust::tuple<DataType, DataType> &a,
            const thrust::tuple<DataType, DataType> &b)
    {
        auto a_0 = thrust::get<0>(a); auto b_0 = thrust::get<0>(b);
        auto a_1 = thrust::get<1>(a); auto b_1 = thrust::get<1>(b);

        return (a_0 < b_0) ||
               ((a_0 == b_0) && (a_1 < b_1));
    }
};

template<typename DataType>
struct ZipComparator_3
{
    __host__ __device__
    inline bool operator() (
            const thrust::tuple<DataType, DataType, DataType> &a,
            const thrust::tuple<DataType, DataType, DataType> &b)
    {
        auto a_0 = thrust::get<0>(a); auto b_0 = thrust::get<0>(b);
        auto a_1 = thrust::get<1>(a); auto b_1 = thrust::get<1>(b);
        auto a_2 = thrust::get<2>(a); auto b_2 = thrust::get<2>(b);

        return (a_0 < b_0) ||
               ((a_0 == b_0) && (a_1 < b_1)) ||
               ((a_0 == b_0) && (a_1 == b_1) && (a_2 < b_2)) ;
    }
};

template<typename DataType>
struct ZipComparator_4
{
    __host__ __device__
    inline bool operator() (
            const thrust::tuple<DataType, DataType, DataType, DataType> &a,
            const thrust::tuple<DataType, DataType, DataType, DataType> &b)
    {
        auto a_0 = thrust::get<0>(a); auto b_0 = thrust::get<0>(b);
        auto a_1 = thrust::get<1>(a); auto b_1 = thrust::get<1>(b);
        auto a_2 = thrust::get<2>(a); auto b_2 = thrust::get<2>(b);
        auto a_3 = thrust::get<3>(a); auto b_3 = thrust::get<3>(b);

        return (a_0 < b_0) ||
               ((a_0 == b_0) && (a_1 < b_1)) ||
               ((a_0 == b_0) && (a_1 == b_1) && (a_2 < b_2)) ||
               ((a_0 == b_0) && (a_1 == b_1) && (a_2 == b_2) && (a_3 < b_3));
    }
};


template<typename DataType>
struct ZipComparator_5
{
    __host__ __device__
    inline bool operator() (
            const thrust::tuple<DataType, DataType, DataType, DataType, DataType> &a,
            const thrust::tuple<DataType, DataType, DataType, DataType, DataType> &b)
    {
        auto a_0 = thrust::get<0>(a); auto b_0 = thrust::get<0>(b);
        auto a_1 = thrust::get<1>(a); auto b_1 = thrust::get<1>(b);
        auto a_2 = thrust::get<2>(a); auto b_2 = thrust::get<2>(b);
        auto a_3 = thrust::get<3>(a); auto b_3 = thrust::get<3>(b);
        auto a_4 = thrust::get<4>(a); auto b_4 = thrust::get<4>(b);

        return (a_0 < b_0) ||
               ((a_0 == b_0) && (a_1 < b_1)) ||
               ((a_0 == b_0) && (a_1 == b_1) && (a_2 < b_2)) ||
               ((a_0 == b_0) && (a_1 == b_1) && (a_2 == b_2) && (a_3 < b_3)) ||
               ((a_0 == b_0) && (a_1 == b_1) && (a_2 == b_2) && (a_3 == b_3) && (a_4 < b_4));
    }
};

/*kernel functions*/
/*
 * input:   multi-column data
 * bits:    bit filter
 * num:     number of input items
 * key_len: length of each key for comparison
 * */
template<typename DataType, typename FrontBitType, typename LastBitType, typename CntType>
__global__
void extractMasks(DataType **input, FrontBitType **front_bits_arr, LastBitType *last_bits, CntType num, int ref_key_len)
{
    auto gtid = (CntType)(threadIdx.x + blockIdx.x * blockDim.x);
    auto gtnum = (CntType)(blockDim.x * gridDim.x);

    if (front_bits_arr && (gtid < ref_key_len-1))  front_bits_arr[gtid][0] = (bool)1;
    if (gtid == ref_key_len-1) last_bits[0] = 1;

    while (gtid < (num-1))
    {
        bool acc_boolean = false;
#pragma unroll
        for(auto k = 0; k < ref_key_len-1; k++) {
            acc_boolean = acc_boolean || (input[k][gtid+1] != input[k][gtid]);
            front_bits_arr[k][gtid+1] = acc_boolean;
        }
        last_bits[gtid+1] = (acc_boolean || (input[ref_key_len-1][gtid+1] != input[ref_key_len-1][gtid]));
        gtid += gtnum;
    }
}

/*
 * Sort R(A,B,C,...) according to A, B, C ..., using thrust
 * and then build the Trie structure
 * Support up to 5 columns
 * */
template <typename DataType, typename CntType, int numCol>
void constructSortedTrie(
        DataType **data_in, CntType len,
        DataType **&data_out, CntType **offset_out, CntType *&data_len_out,
        CUDAMemStat *memstat, CUDATimeStat *timing) {
    Timer t;
    t.reset();
    using data_ptr = thrust::device_ptr<DataType>;
    vector<data_ptr> ptrs;

    /*switch DataType* type to thrust device_ptr type*/
    for(auto i = 0; i < numCol; i++) {
        if (!data_in[i]) {
            log_error("Column %d is nullptr.", i);
            exit(1);
        }
        data_ptr temp(data_in[i]);
        ptrs.emplace_back(temp);
    }

    /*sort with thrust zip iterator*/
    Timer tt;
    switch (numCol) {
        case 1: {
            timingKernel(thrust::sort(thrust::device, ptrs[0], ptrs[0]+len), timing);
            break;
        }
        case 2: {
            timingKernel(thrust::sort(thrust::device,
                    thrust::make_zip_iterator(thrust::make_tuple(ptrs[0], ptrs[1])),
                    thrust::make_zip_iterator(thrust::make_tuple(ptrs[0]+len, ptrs[1]+len)),
                    ZipComparator_2<DataType>()), timing);

//            CUBSortPairsInPlace(data_in[0], data_in[1], len, memstat, timing); //can only handle no more than 2G items!
            break;
        }
        case 3: {
            timingKernel(
            thrust::sort(thrust::device,
                    thrust::make_zip_iterator(thrust::make_tuple(ptrs[0], ptrs[1], ptrs[2])),
                    thrust::make_zip_iterator(thrust::make_tuple(ptrs[0]+len, ptrs[1]+len, ptrs[2]+len)),
                    ZipComparator_3<DataType>()), timing);
            break;
        }
        case 4: {
            timingKernel(
            thrust::sort(thrust::device,
                    thrust::make_zip_iterator(thrust::make_tuple(ptrs[0], ptrs[1], ptrs[2], ptrs[3])),
                    thrust::make_zip_iterator(thrust::make_tuple(ptrs[0]+len, ptrs[1]+len, ptrs[2]+len, ptrs[3]+len)), ZipComparator_4<DataType>()), timing);
            break;
        }
        case 5: {
            timingKernel(
            thrust::sort(thrust::device,
                    thrust::make_zip_iterator(thrust::make_tuple(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4])), thrust::make_zip_iterator(thrust::make_tuple(ptrs[0]+len, ptrs[1]+len, ptrs[2]+len, ptrs[3]+len, ptrs[4]+len)),
                    ZipComparator_5<DataType>()), timing);
            break;
        }
        default: {
            log_error("Unsupported number of columns: %d.", numCol);
            exit(1);
        }
    }
    log_info("sort time in Trie construction: %.2f s", tt.elapsed());

    /*check sorting*/
#ifdef DEBUG
    cout<<"check sorting for these "<<len<<" items"<<endl;
#pragma omp parallel for
    for(CntType i = 0; i <len-1; i++) {
        if ((data_in[0][i] > data_in[0][i+1]) || ((data_in[0][i] == data_in[0][i+1]) && (data_in[1][i] > data_in[1][i+1]))) {
            printf("wrong idx %llu: (%d,%d) and (%d,%d)\n", i, data_in[0][i], data_in[1][i], data_in[0][i+1], data_in[1][i+1]);
        }
        assert((data_in[0][i] < data_in[0][i+1]) || ((data_in[0][i] == data_in[0][i+1]) && (data_in[1][i] <= data_in[1][i+1])));
    }
    cout<<"check sorting finished"<<endl;
#endif

    /*construct the Trie*/
    CntType *ascending_offsets = nullptr;
    if (numCol > 1) {
        CUDA_MALLOC(&ascending_offsets, sizeof(CntType)*len, memstat);
        thrust::counting_iterator<CntType> iter(0);
        timingKernel(thrust::copy(iter, iter + len, ascending_offsets), timing);
    }
    auto total = len;

    /*init the dif_bits boolean arrays*/
    bool** diff_bits_arr = nullptr;
    if (numCol > 2) CUDA_MALLOC(&diff_bits_arr, sizeof(bool*)*abs(numCol-2), memstat);
    for(auto i = 0; i < numCol-2; i++) {
        bool *diff_bits = nullptr;
        CUDA_MALLOC(&diff_bits, sizeof(bool)*len, memstat);
        diff_bits_arr[i] = diff_bits;
    }

    /*init the dif_bits CntType array*/
    CntType *diff_bits_cnttype = nullptr;
    if (numCol > 1) {
        CUDA_MALLOC(&diff_bits_cnttype, sizeof(CntType)*len, memstat);

        /*Extract the diff_bits to retrieve the distinct values*/
        execKernel(extractMasks, GRID_SIZE, BLOCK_SIZE, timing, false, data_in, diff_bits_arr, diff_bits_cnttype, len, numCol-1);
    }

    for(int c = numCol - 2; c >= 0; c--) {
        DataType *out_data_temp = nullptr;
        CntType *out_offset_temp = nullptr;
        CUDA_MALLOC(&out_data_temp, sizeof(DataType)*len, memstat);
        CUDA_MALLOC(&out_offset_temp, sizeof(CntType)*(len+1), memstat);

        CntType sum;
        if (c == numCol - 2) {
//            sum = CUBSelect(data_in[c], out_data_temp, diff_bits_cnttype, len, memstat, timing);
//            CUBSelect(ascending_offsets, out_offset_temp, diff_bits_cnttype, len, memstat, timing);
            /*segmented version*/
            sum = CUBSegSelect(data_in[c], out_data_temp, diff_bits_cnttype, len, memstat, timing);
            CUBSegSelect(ascending_offsets, out_offset_temp, diff_bits_cnttype, len, memstat, timing);
        }
        else {
//            sum = CUBSelect(data_in[c], out_data_temp, diff_bits_arr[c], len, memstat, timing);
//            CUBSelect(ascending_offsets, out_offset_temp, diff_bits_arr[c], len, memstat, timing);
            /*segmented version*/
            sum = CUBSegSelect(data_in[c], out_data_temp, diff_bits_arr[c], len, memstat, timing);
            CUBSegSelect(ascending_offsets, out_offset_temp, diff_bits_arr[c], len, memstat, timing);
        }
        out_offset_temp[sum] = total;

        /*record to the output*/
        data_out[c] = out_data_temp;
        offset_out[c] = out_offset_temp;
        data_len_out[c] = sum;

        /*scan the bitmap for the next iteration if c != 0*/
        if(c != 0) {
            if (c == numCol - 2)
                total = CUBScanExclusive(diff_bits_cnttype, diff_bits_cnttype, len, memstat, timing); //todo: scan may also have problems!
            else
                total = CUBScanExclusive(diff_bits_arr[c], diff_bits_cnttype, len, memstat, timing);
        }
        std::swap(ascending_offsets, diff_bits_cnttype);
    }

    /*the last level of the Trie is the the last column of the sorted array */
    data_len_out[numCol-1] = len;

    /*hard copy*/
    CUDA_MALLOC(&data_out[numCol-1], sizeof(DataType)*len, memstat);
    checkCudaErrors(cudaMemcpy(data_out[numCol-1], data_in[numCol-1], sizeof(DataType)*len, cudaMemcpyDeviceToDevice));
//    data_out[numCol-1] = data_in[numCol-1];
    checkCudaErrors(cudaDeviceSynchronize());

#ifdef FREE_DATA
    CUDA_FREE(ascending_offsets, memstat);
    for(auto i = 0; i < numCol-2; i++) CUDA_FREE(diff_bits_arr[i], memstat);
    if (numCol > 2) CUDA_FREE(diff_bits_arr, memstat);
    if (numCol > 1) CUDA_FREE(diff_bits_cnttype, memstat);
#endif
}

