//
// Created by Bryan on 30/11/2018.
//

#pragma once

#ifdef __JETBRAINS_IDE__
#include "cuda/cuda_fake/fake.h"
#include "openmp_fake.h"
#endif

#include <cmath>
#include <curand_kernel.h>
#include <iostream>
#include "cub/cub.cuh"

#include "conf.h"
#include "cuda/CUDAStat.cuh"
#include "log.h"
#include <omp.h>
using namespace std;

#define FIXED_SEED      (1234ull)

__global__ static void setup(curandState *state, unsigned long long seed) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    /* Each thread gets the same seed, a different sequence
       number, no offset */
    curand_init(seed, tid, 0, &state[tid]);
}

template<typename DataType, typename CntType, typename RangeTypeS, typename RangeTypeE>
__global__ void zipf_computation(
        DataType *data, CntType *mapping, CntType count, RangeTypeE range, RangeTypeS rangeStart,
        double *acc_sums, double totalSum, curandState *state) {
    CntType tid = threadIdx.x + blockIdx.x * blockDim.x;
    CntType tnum = blockDim.x * gridDim.x;
    curandState *myState = &state[tid];
    while (tid < count) {
        double z = curand_uniform_double(myState);

        /*binary search, find the first one that is larger than z*/
        int start = 0, end = range-1;
        while (start <= end) {
            int middle = start + (end - start)/2;
            if (acc_sums[middle]*totalSum < z) start = middle+1;
            else end = middle - 1;
        }
        if (end == range - 1)   data[tid] = mapping[range - 1] + rangeStart;
        else                    data[tid] = mapping[end + 1] + rangeStart;
        tid += tnum;
    }
}

template<typename CntType>
__global__ void power_computation(double *powers, CntType count, double alpha, double *totalSum) {
    auto tid = threadIdx.x;
    auto bid = blockIdx.x;
    CntType gtid = threadIdx.x + blockIdx.x * blockDim.x;
    CntType gtnum = blockDim.x * gridDim.x;
    __shared__ double sharedSum;
    __shared__ double localSums[BLOCK_SIZE];

    if (0 == tid) sharedSum = 0;
    localSums[tid] = 0;
    __syncthreads();

    while (gtid < count) {
        powers[gtid] = 1.0 / pow((double)(gtid+1), alpha);
        localSums[tid] += powers[gtid];
        gtid += gtnum;
    }
    __syncthreads();

    /*do not use atomicAdd because atomic funcs with double values
     * are not supported on sm_35*/
    if(0 == threadIdx.x) {
        for(int i = 0; i < BLOCK_SIZE; i++) sharedSum += localSums[i];
        totalSum[bid] = sharedSum;
    }
}

template<typename DataType, typename CntType, typename RangeTypeS, typename RangeTypeE>
__global__ void uniform_computation(
        DataType *data, CntType count, RangeTypeE range, RangeTypeS rangeStart,
        curandState *state) {
    auto tid = CntType(threadIdx.x + blockIdx.x * blockDim.x);
    auto tnum = CntType(blockDim.x * gridDim.x);
    curandState *myState = &state[tid];

    while (tid < count) {
        /*
         * curand_uniform_double() returns (0.0, 1.0]
         * Here curand_uniform_double() is neccessary since the precision of curand_uniform() is not enough
         * */
        double z = curand_uniform_double(myState);

        /* range of data[tid]: [rangeStart,rangeStart+range) */
        z *= (-0.000001 + range);
        z += rangeStart;
        data[tid] = (DataType)z;
        tid += tnum;
    }
}

/*------------ host code---------------*/

template<typename DataType, typename CntType, typename RangeTypeS, typename RangeTypeE>
void zipf_generator_GPU(
        DataType *data, CntType cnt, RangeTypeS range_start, RangeTypeE range_end,
        double alpha, CntType *mapping=nullptr) {
    unsigned long long seed = rand();
    static int delta = 0;  /*to ensure each time different sets of values are generated*/

    if (!data) {
        log_error("Memory object for data generator is not allocated.");
        exit(1);
    }
    auto range = range_end - (RangeTypeE)range_start;

    double *powers, *totalPower, *acc_sums;
    curandState *randStates;
    CUDA_MALLOC(&powers, sizeof(double)*range, nullptr);
    CUDA_MALLOC(&totalPower, sizeof(double)*GRID_SIZE, nullptr);
    CUDA_MALLOC(&randStates, sizeof(curandState)*GRID_SIZE*BLOCK_SIZE, nullptr);
    CUDA_MALLOC(&acc_sums, sizeof(double)*range, nullptr);

    setup<<<GRID_SIZE, BLOCK_SIZE>>>(randStates, seed+delta);
    power_computation <<<GRID_SIZE, BLOCK_SIZE>>>(powers, range, alpha, totalPower);
    checkCudaErrors(cudaDeviceSynchronize());

    /*compute the acc_sums according to powers*/
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, powers, acc_sums, range);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, nullptr);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, powers, acc_sums, range);
    checkCudaErrors(cudaDeviceSynchronize());

    CUDA_FREE(d_temp_storage, nullptr);
    CUDA_FREE(powers, nullptr);

    double sumTotal = 0;
    for(auto i = 0; i < GRID_SIZE; i++)    sumTotal += totalPower[i];
    sumTotal = 1.0 / sumTotal;

    /*construct the random permutation mapping using Knuth shuffle*/
    bool generate_mapping = false;
    if (!mapping) {
        log_info("Generate new mapping");
        generate_mapping = true;
        CUDA_MALLOC(&mapping, sizeof(CntType)*range, nullptr);
#pragma omp parallel for
        for(auto i = 0; i < range; i++) mapping[i] = i;

        for(auto i = range - 1; i >= 1; i--) {
            auto j = rand() % (i+1);
            std::swap(mapping[i], mapping[j]);
        }
    }

    zipf_computation <<<GRID_SIZE, BLOCK_SIZE>>>(data, mapping, cnt, range, range_start, acc_sums, sumTotal, randStates);
    cudaDeviceSynchronize();

    log_trace("Data init with GPU: num: %llu, size: %.2f GB, range: %llu-%llu, type: zipf(z=%.1f).",
              cnt, 1.0*cnt/1024/1024/1024*sizeof(DataType), range_start, range_end, alpha);

#ifdef DEBUG
    uint32_t *his = new uint32_t[range];
    for(auto i = 0; i < range; i++) his[i] = 0;
    for(auto i = 0; i < count; i++) {
        his[data[i]]++;
    }
    log_debug("Top 10 frequent values:");
    for(auto i = 0; i < 10; i++) {
        log_debug("val: %d, freq: %d.", mapping[i], his[mapping[i]]);
    }
#endif

    CUDA_FREE(acc_sums, nullptr);
    CUDA_FREE(totalPower, nullptr);
    CUDA_FREE(randStates, nullptr);
    if (generate_mapping) CUDA_FREE(mapping, nullptr);
    delta ++;
}

template<typename DataType, typename CntType, typename RangeTypeS, typename RangeTypeE>
void uniform_generator_GPU(DataType *data, CntType cnt, RangeTypeS range_start, RangeTypeE range_end, unsigned long long seed=FIXED_SEED) {
    static int delta = 0;  /*to ensure each time different sets of values are generated*/
    if (!data) {
        log_error("Memory object for data generator is not allocated.");
        exit(1);
    }
    auto range = range_end - (RangeTypeE)range_start;

    curandState *randStates;
    checkCudaErrors(cudaMalloc((void**)&randStates, sizeof(curandState)*GRID_SIZE*BLOCK_SIZE));

    setup<<<GRID_SIZE, BLOCK_SIZE>>>(randStates, seed+delta);
    uniform_computation <<<GRID_SIZE, BLOCK_SIZE>>>(data, cnt, range, range_start, randStates);
    cudaDeviceSynchronize();

    log_trace("Data init with GPU: num: %llu, size: %.2f GB, range: %llu-%llu, type: uniform.",
             cnt, 1.0*cnt/1024/1024/1024*sizeof(DataType), range_start, range_end);
#ifdef DEBUG
    uint32_t max_cnt = 0, min_cnt = 9999999;
    DataType max_val, min_val;
    uint32_t *his = new uint32_t[range];

    for(auto i = 0; i < range; i++) his[i] = 0;
    for(auto i = 0; i < count; i++) {
        his[data[i]]++;
    }
    for(auto i = 0; i < range; i++) {
        if (his[i] > max_cnt) {
            max_cnt = his[i];
            max_val = i;
        }
        if (his[i] < min_cnt) {
            min_cnt = his[i];
            min_val = i;
        }
    }
    log_debug("Max freq val: %lu, freq: %d.", max_val, max_cnt);
    log_debug("Min freq val: %lu, freq: %d.", min_val, min_cnt);
    delete[] his;
#endif
    checkCudaErrors(cudaFree(randStates));
    delta ++;
}

template<typename DataType, typename CntType, typename RangeTypeS, typename RangeTypeE>
void uniform_generator(DataType *data, CntType cnt, RangeTypeS range_start, RangeTypeE range_end) {
    static unsigned int seed = rand() % 1048576;
    if (!data) {
        log_error("Memory object for data generator is not allocated.");
        exit(1);
    }
    auto range = range_end - (RangeTypeE)range_start;
    auto count_percentage = cnt / 10;

#pragma omp parallel
    {
        unsigned int tid = omp_get_thread_num();
        unsigned my_seed = seed + tid;
#pragma omp for schedule(dynamic)
        for(CntType i = 0; i < cnt; i++) {
            data[i] = rand_r(&my_seed) % range + range_start;
            if((0 != count_percentage) && (i % count_percentage) == 0)
                    log_info("Initialized %.0f%% of the data, idx=%llu", 10.0*i/count_percentage,i);
        }
    }
    log_trace("Data init with CPU: num: %llu, size: %.2f GB, range: %llu-%llu, type: uniform.",
              cnt, 1.0*cnt/1024/1024/1024*sizeof(DataType), range_start, range_end);
    seed += 1024;
}