//
// Created by Bryan on 22/7/2019.
//
#pragma once

#ifdef __JETBRAINS_IDE__
#include "cuda/cuda_fake/fake.h"
#endif

#include "CUDAStat.cuh"
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <cooperative_groups.h>
using namespace std;
using namespace cooperative_groups;

#define GRID_SIZE_DEFAULT   (1024)
#define BLOCK_SIZE_DEFAULT  (256)
#define PRIM_WARP_SIZE           (32)

/*comparator for ThrustSortPairs
 * a_0, b_0: key1
 * a_1, b_1: key2
 * a_2, b_2: index*/
template<typename DataType1, typename DataType2, typename DataType3>
struct Thrust_ZipComparator_3
{
    __host__ __device__
    inline bool operator() (
            const thrust::tuple<DataType1, DataType2, DataType3> &a,
            const thrust::tuple<DataType1, DataType2, DataType3> &b)
    {
        auto a_0 = thrust::get<0>(a); auto b_0 = thrust::get<0>(b);
        auto a_1 = thrust::get<1>(a); auto b_1 = thrust::get<1>(b);
        auto a_2 = thrust::get<2>(a); auto b_2 = thrust::get<2>(b);

        return (a_0 < b_0) || ((a_0 == b_0) && (a_1 < b_1));
    }
};

template<typename DataType, typename CntType>
__device__
bool hash_probe(
        DataType needle,
        DataType *haystack, CntType &hay_start, CntType hay_end,
        char &msIdx, CntType *iterators,
        CntType &auc, int lane) {
    if (0 == lane) {
        auc = hay_end;
        msIdx = 0;
    }
    __syncwarp();

    /*update current iterator and point to the next matched batch of items*/
    for(auto i = hay_start + lane; i < auc; i += PRIM_WARP_SIZE) {
        auto active_come_in = coalesced_threads();
        if (needle == haystack[i]) {
            auto active = coalesced_threads();
            auto rank = active.thread_rank(); /*rank within the active group*/
            iterators[rank] = i; /*can write multiple iterators*/
            if (0 == rank) {
                msIdx = (char)(active.size()-1);
                hay_start = i + PRIM_WARP_SIZE - lane;
                auc = 0;
            }
        }
        active_come_in.sync();
    }
    __syncwarp();
    return (auc != hay_end); //a match is found
}

template<typename T>
__device__
uint32_t BinarySearchForGallopingSearch(const T *array, uint32_t offset_beg, uint32_t offset_end, int val) {
    while (offset_end - offset_beg >= 32) {
        auto mid = static_cast<uint32_t>((static_cast<unsigned long>(offset_beg) + offset_end) / 2);
        if (array[mid] == val) {
            return mid;
        } else if (array[mid] < val) {
            offset_beg = mid + 1;
        } else {
            offset_end = mid;
        }
    }

    // linear search fallback
    for (auto offset = offset_beg; offset < offset_end; offset++) {
        if (array[offset] >= val) {
            return offset;
        }
    }
    return offset_end;
}

template<typename T>
__device__
uint32_t GallopingSearch(T *array, uint32_t offset_beg, uint32_t offset_end, int val) {
    if (array[offset_end - 1] < val) {
        return offset_end;
    }
    // galloping
    if (array[offset_beg] >= val) {
        return offset_beg;
    }
    if (array[offset_beg + 1] >= val) {
        return offset_beg + 1;
    }
    if (array[offset_beg + 2] >= val) {
        return offset_beg + 2;
    }

    auto jump_idx = 4u;
    while (true) {
        auto peek_idx = offset_beg + jump_idx;
        if (peek_idx >= offset_end) {
            return BinarySearchForGallopingSearch(array, (jump_idx >> 1) + offset_beg + 1, offset_end, val);
        }
        if (array[peek_idx] < val) {
            jump_idx <<= 1;
        } else {
            return array[peek_idx] == val ? peek_idx :
                   BinarySearchForGallopingSearch(array, (jump_idx >> 1) + offset_beg + 1, peek_idx + 1, val);
        }
    }
}

/*CUDA kernels*/
template <typename DataType, typename IndexType, typename CntType>
__global__
void gather(DataType *input, DataType *output, IndexType *idxes, CntType cnt)
{
    CntType gtid = (CntType)(threadIdx.x + blockDim.x * blockIdx.x);
    CntType gnum = (CntType)(blockDim.x * gridDim.x);

    while (gtid < cnt)
    {
        output[gtid] = input[idxes[gtid]];
        gtid += gnum;
    }
}


/*
 * Single-thread device function.
 * Return the lower bound of the needle (the 1st item larger than or equal to needle)
 * in the haystacks[hay_start,hay_end).
 * Return false when no match is found.
 * Haystacks are sorted.
 * */
template<typename NeedleType, typename HaystackType, typename CntType>
__device__
bool dev_lower_bound( //faster than binary search
        NeedleType needle, HaystackType *haystacks,
        CntType hay_start, CntType hay_end, CntType &lower_bound)
{
    int middle, lo = hay_start, hi = hay_end - 1;
    while (lo <= hi) {
        middle = lo + (hi - lo)/2;
        if (needle > haystacks[middle])
            lo = middle + 1;
        else
            hi = middle - 1;
    }
    lower_bound = lo;

    if (lower_bound == hay_end) return false; //needle is greater than all the haystarts items
    return (haystacks[lo] == needle);
}

template<typename NeedleType, typename HaystackType, typename CntType>
__device__
bool dev_lower_bound_galloping( //faster than binary search
        NeedleType needle, HaystackType *haystacks,
        CntType hay_start, CntType hay_end, CntType &lower_bound) {
    /*todo: change int to CntType*/
    long long int lo = hay_start;
    long long int hi = hay_start;
    long long int scale = 8;
    while ((hi < hay_end) && (haystacks[hi] < needle)) {
        lo = hi;
        hi += scale;
        scale <<= 3;
    }
    if (hi > hay_end-1) hi = hay_end-1;

    while (lo <= hi) {
        scale = lo + (hi - lo)/2;
        if (needle > haystacks[scale])
            lo = scale + 1;
        else
            hi = scale - 1;
    }
    lower_bound = lo;
    if (lower_bound == hay_end) return false; //needle is greater than all the haystarts items
    return (haystacks[lower_bound] == needle);
}

/*
 * Single-thread device function.
 * Return the upper bound of the needle (the 1st item larger than needle) in the haystacks. Return false when no match is found
 * Haystacks are sorted.
 * */
template<typename NeedleType, typename HaystackType, typename CntType>
__device__
bool dev_upper_bound_galloping( //faster than binary search
        NeedleType needle, HaystackType *haystacks,
        CntType hay_start, CntType hay_end, CntType &upper_bound) {
    long long int lo = hay_start;
    long long int hi = hay_start;
    long long int scale = 8;

    //haystacks[hi] can be equal to needle in upper_bound
    while ((hi < hay_end) && (haystacks[hi] <= needle)) {
        lo = hi;
        hi += scale;
        scale <<= 3;
    }
    if (hi > hay_end-1) hi = hay_end-1;

    while (lo <= hi) {
        scale = lo + (hi - lo)/2;
        if (needle >= haystacks[scale])
            lo = scale + 1;
        else
            hi = scale - 1;
    }
    upper_bound = lo;
    return true;
}

/*
 * Single-thread device function.
 * Return the index of the match with binary search, return INVALID_VAL if the value is not found.
 * Haystacks are sorted.
 * */
template<typename NeedleType, typename HaystackType, typename CntType, CntType INVALID_VAL>
__device__
CntType dev_binary_search(NeedleType needle, HaystackType *haystacks,
                          CntType hay_start, CntType hay_end)
{
    int middle, begin = hay_start, end = hay_end - 1;
    while (begin <= end)
    {
        middle = begin + (end - begin)/2;
        if (needle > haystacks[middle])
            begin = middle + 1;
        else if (needle < haystacks[middle])
            end = middle - 1;
        else
            return (CntType)middle;
    }
    return INVALID_VAL;
}

/*wrapper of the CUB primitives*/
template<typename DataType, typename CntType>
DataType CUBMax(
        DataType *input,
        const CntType count,
        CUDAMemStat *memstat,
        CUDATimeStat *timer)
{
    DataType *maxVal = nullptr;
    CUDA_MALLOC(&maxVal, sizeof(DataType), memstat);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    timingKernel(
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, input, maxVal, count), timer);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, memstat);
    timingKernel(
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, input, maxVal, count), timer);

    DataType res = *maxVal;
    CUDA_FREE(d_temp_storage, memstat);
    CUDA_FREE(maxVal, memstat);

    return res;
}

template<typename InputType, typename OutputType, typename CntType>
OutputType CUBScanExclusive(InputType *input, OutputType *output, const CntType count,
                            CUDAMemStat *mem_stat, CUDATimeStat *timer) {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    InputType last_input;
    OutputType last_output;
    checkCudaErrors(cudaMemcpy(&last_input, input+(count-1), sizeof(InputType), cudaMemcpyDeviceToHost));

    timingKernel(
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input, output, count), timer);
    checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    timingKernel(
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, input, output, count), timer);
#ifdef FREE_DATA
    cudaFree(d_temp_storage);
#endif

    checkCudaErrors(cudaMemcpy(&last_output, output+(count-1), sizeof(OutputType), cudaMemcpyDeviceToHost));
    return last_output + (OutputType)last_input;
}

template<typename DataType, typename SumType, typename CntType>
SumType CUBSum(
        DataType *input,
        CntType count,
        CUDAMemStat *mem_stat,
        CUDATimeStat *timer)
{
    SumType *sum_value = nullptr;
    CUDA_MALLOC(&sum_value, sizeof(SumType), mem_stat);

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    timingKernel(
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input, sum_value, count), timer);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, mem_stat);
    timingKernel(
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input, sum_value, count), timer);

    SumType res = *sum_value;
    CUDA_FREE(d_temp_storage, mem_stat);
    CUDA_FREE(sum_value, mem_stat);

    return res;
}

/*Attention: cnt_input is int in CUB library!!! Please use our CUBSegmentSelect when
 * the range cnt_input is greater than the range of an int*/
template<typename DataType, typename CntType, typename FlagType>
CntType CUBSelect(DataType *input, DataType *output, FlagType *flags, const CntType cnt_input,
                  CUDAMemStat *mem_stat, CUDATimeStat *timer, cudaStream_t stream = 0) {
    CntType *cnt_output = nullptr;
    CntType res = 0;
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    checkCudaErrors(cudaMalloc((void**)&cnt_output, sizeof(CntType)));

    timingKernel(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, input,
                                            flags, output, cnt_output, cnt_input, stream), timer);
    checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    timingKernel(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, input,
                                            flags, output, cnt_output, cnt_input, stream), timer);

    checkCudaErrors(cudaFree(d_temp_storage));
    checkCudaErrors(cudaMemcpy(&res, cnt_output, sizeof(CntType), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(cnt_output));
    return res;
}

/*to handle cases where cnt_input has range greater than the range of an int*/
template<typename DataType, typename CntType, typename FlagType>
CntType CUBSegSelect(DataType *input, DataType *output, FlagType *flags, const CntType cnt_input,
                     CUDAMemStat *mem_stat, CUDATimeStat *timer, cudaStream_t stream = 0) {
    CntType items_per_seg = INT32_MAX/4;
    auto segments = (cnt_input + items_per_seg - 1)/items_per_seg;

    /*allocate output space*/
    CntType *cnt_output = nullptr;
    CntType output_off = 0;
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    checkCudaErrors(cudaMalloc((void**)&cnt_output, sizeof(CntType)));

    for(auto i = 0; i < segments; i++) {
        CntType seg_start = items_per_seg * i;
        CntType seg_len = items_per_seg;
        if (seg_start + seg_len > cnt_input) seg_len = cnt_input - seg_start;
        assert((seg_len > 0) && (seg_len < INT_MAX));

        /*CUB functions*/
        timingKernel(
                cub::DeviceSelect::Flagged(
                        d_temp_storage, temp_storage_bytes,
                        input, flags,
                        output, cnt_output, seg_len, stream), timer);
        checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        timingKernel(
                cub::DeviceSelect::Flagged(
                        d_temp_storage, temp_storage_bytes,
                        input+seg_start, flags+seg_start,
                        output+output_off, cnt_output, seg_len, stream), timer);
        CntType cur_output = 0;
        checkCudaErrors(cudaMemcpy(&cur_output, cnt_output, sizeof(CntType), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_temp_storage));
        d_temp_storage = nullptr; //must set to nullptr for next allocation
        output_off += cur_output;
    }
    checkCudaErrors(cudaFree(cnt_output));
    return output_off;
}

template<typename DataType, typename CntType, typename PredicateType>
CntType CUBIf(
        DataType *input,
        DataType *output,
        PredicateType predicate,
        const CntType cnt_input,
        CUDAMemStat *mem_stat,
        CUDATimeStat *timer)
{
    CntType *cnt_output = nullptr;
    CUDA_MALLOC(&cnt_output, sizeof(CntType), mem_stat);

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    timingKernel(
            cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, input, output, cnt_output, cnt_input, predicate), timer);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, mem_stat);
    timingKernel(
            cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, input, output, cnt_output, cnt_input, predicate), timer);

    CUDA_FREE(d_temp_storage, mem_stat);
    auto res = *cnt_output;
    CUDA_FREE(cnt_output, mem_stat);

    return res;
}

/*Sort pairs (keys,values) according to keys, then values, also provide with the offsets in the original tables*/
template<typename DataType, typename CntType, typename IndexType>
void CUBSortPairs(
        DataType *keysIn, DataType *keysOut,
        DataType *valuesIn, DataType *valuesOut,
        IndexType *idx_ascending, CntType cnt,
        CUDAMemStat *memstat, CUDATimeStat *timing)
{
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CntType *idx_ascending_temp;
    DataType *keys_temp;
    CUDA_MALLOC(&idx_ascending_temp, sizeof(CntType)*cnt, memstat);
    CUDA_MALLOC(&keys_temp, sizeof(DataType)*cnt, memstat);

    thrust::counting_iterator<CntType> iter(0);
    timingKernel(
            thrust::copy(iter, iter + cnt, idx_ascending), timing);

    /*Sort according to values*/
    timingKernel(
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, valuesIn, valuesOut, idx_ascending, idx_ascending_temp, cnt), timing);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, memstat);
    timingKernel(
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, valuesIn, valuesOut, idx_ascending, idx_ascending_temp, cnt), timing);

    CUDA_FREE(d_temp_storage, memstat);
    d_temp_storage = nullptr;

    /*rearrange the keys*/
    execKernel(gather,GRID_SIZE_DEFAULT,BLOCK_SIZE_DEFAULT,timing,false,keysIn, keys_temp, idx_ascending_temp, cnt);

    /* Sort according to keys, but have to use stable sort*/
    timingKernel(
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys_temp, keysOut, idx_ascending_temp, idx_ascending, cnt), timing);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, memstat);
    timingKernel(
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys_temp, keysOut, idx_ascending_temp, idx_ascending, cnt), timing);

    CUDA_FREE(d_temp_storage, memstat);
    CUDA_FREE(idx_ascending_temp,memstat);
    CUDA_FREE(keys_temp,memstat);

    /*rearrange the values according to the indexes*/
    execKernel(gather,GRID_SIZE_DEFAULT,BLOCK_SIZE_DEFAULT, timing, false, valuesIn, valuesOut, idx_ascending, cnt);
}

/*Simply sort pairs (keys,values), release inputs*/
template<typename DataType, typename CntType>
void CUBSortPairsInPlace(
        DataType *keys, DataType *values, CntType cnt,
        CUDAMemStat *memstat, CUDATimeStat *timing) {
    if (cnt > 2147435520) {
        log_error("CUB sort could not sort these many pairs correctly");
        exit(1);
    }

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    DataType *keys_temp = nullptr, *values_temp = nullptr;
    CUDA_MALLOC(&keys_temp, sizeof(DataType)*cnt, memstat);
    CUDA_MALLOC(&values_temp, sizeof(DataType)*cnt, memstat);

    /*Sort according to values*/
    timingKernel(
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, values, values_temp, keys, keys_temp, cnt), timing);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, memstat);
    timingKernel(
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, values, values_temp, keys, keys_temp, cnt), timing);
    CUDA_FREE(d_temp_storage, memstat);
    d_temp_storage = nullptr;

    /* Sort according to keys*/
    timingKernel(
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys_temp, keys, values_temp, values, cnt), timing);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, memstat);
    timingKernel(
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys_temp, keys, values_temp, values, cnt), timing);

    CUDA_FREE(d_temp_storage, memstat);
    CUDA_FREE(keys_temp,memstat);
    CUDA_FREE(values_temp,memstat);
}

/*Sort pairs (keys,values) according to keys, then values, also provide with the offsets in the original tables*/
template<typename DataType, typename CntType, typename IndexType>
void ThrustSortPairs(
        DataType *keysIn, DataType *keysOut,
        DataType *valuesIn, DataType *valuesOut,
        IndexType *idx_ascending, CntType cnt,
        CUDAMemStat *memstat, CUDATimeStat *timing)
{
    checkCudaErrors(cudaMemcpy(keysOut, keysIn, sizeof(DataType)*cnt, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(valuesOut, valuesIn, sizeof(DataType)*cnt, cudaMemcpyDeviceToDevice));

    thrust::counting_iterator<CntType> iter(0);
    timingKernel(thrust::copy(iter, iter + cnt, idx_ascending), timing);
    thrust::copy(iter, iter + cnt, idx_ascending);

    timingKernel(thrust::sort(thrust::device,
                 thrust::make_zip_iterator(thrust::make_tuple(keysOut, valuesOut, idx_ascending)),
                 thrust::make_zip_iterator(thrust::make_tuple(keysOut+cnt, valuesOut+cnt, idx_ascending+cnt)),
                 Thrust_ZipComparator_3<DataType,DataType,IndexType>()), timing);
}

template<typename DataType, typename CntType>
CntType CUBUnique(
        DataType *input, DataType *output, CntType count,
        CUDAMemStat *mem_stat, CUDATimeStat *timer)
{
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CntType *num_selected = nullptr;
    CUDA_MALLOC(&num_selected, sizeof(CntType), mem_stat);

    timingKernel(
            cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, input, output, num_selected, count), timer);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, mem_stat);
    timingKernel(
            cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, input, output, num_selected, count), timer);

    auto res = *num_selected;
    CUDA_FREE(num_selected, mem_stat);
    CUDA_FREE(d_temp_storage, mem_stat);

    return res;
}

template<typename DataType, typename CntType>
void zlai_load_balance_search(CntType count, CntType *exclusive_scans,
                         CntType num_scans, DataType *output) {
    CntType output_iter = 0;
    for(CntType i = 0; i < num_scans-1; i++) {
        auto this_num = exclusive_scans[i+1] - exclusive_scans[i];
        for(CntType j = 0; j < this_num; j++) {
            output[output_iter++] = i;
        }
    }
    auto this_num = count - exclusive_scans[num_scans-1];
    for(CntType j = 0; j < this_num; j++) {
        output[output_iter++] = num_scans-1;
    }
}

template<typename DataType, typename CntType>
void zlai_parallel_load_balance_search(CntType count, CntType *exclusive_scans,
                              CntType num_scans, DataType *output) {

#pragma omp parallel for schedule(static)
    for(CntType i = 0; i < num_scans-1; i++) {
        auto this_num = exclusive_scans[i+1] - exclusive_scans[i];
        for(CntType j = 0; j < this_num; j++) {
            output[exclusive_scans[i]+j] = i;
        }
    }
    auto this_num = count - exclusive_scans[num_scans-1];
#pragma omp parallel for
    for(CntType j = 0; j < this_num; j++) {
        output[exclusive_scans[num_scans-1]+j] = num_scans-1;
    }
}