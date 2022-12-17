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
#include <set>
#include <thrust/sort.h>
#include <omp.h>

#include "config.h"
#include "cuda/CUDAStat.cuh"
#include "log.h"
#include "cub/cub.cuh"
#include "types.h"
#include "timer.h"
using namespace std;

#define GRID_SIZE_GEN   (1024)
#define BLOCK_SIZE_GEN  (1024)
#define MAX_NUM_CPU_THREADS (120)

template<typename DataType>
struct PairItem {
    DataType key0;
    DataType key1;
    bool operator <(const PairItem<DataType> tmp) const {
        return ((this->key0 < tmp.key0) || ((this->key0 == tmp.key0) && (this->key1 < tmp.key1)));
    }
};

__global__
static void setup(curandState *state, unsigned long long seed)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    /* Each thread gets the same seed, a different sequence
       number, no offset */
    curand_init(seed, tid, 0, &state[tid]);
}

template<typename DataType, typename CntType, typename RangeTypeS, typename RangeTypeE>
__global__
void zipf_computation(
        DataType *data, CntType *mapping, CntType count, RangeTypeE range, RangeTypeS rangeStart,
        double *acc_sums, double totalSum, curandState *state)
{
    CntType tid = threadIdx.x + blockIdx.x * blockDim.x;
    CntType tnum = blockDim.x * gridDim.x;

    curandState *myState = &state[tid];

    while (tid < count)
    {
        double z = curand_uniform_double(myState);

        /*binary search, find the first one that is larger than z/totalSum*/
        int start = 0, end = range-1;
        while (start <= end)
        {
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
__global__
void power_computation(double *powers, CntType count, double alpha, double *totalSum)
{
    auto tid = threadIdx.x;
    auto bid = blockIdx.x;
    CntType gtid = threadIdx.x + blockIdx.x * blockDim.x;
    CntType gtnum = blockDim.x * gridDim.x;

    __shared__ double sharedSum;
    __shared__ double localSums[BLOCK_SIZE_GEN];

    if (0 == tid) sharedSum = 0;
    localSums[tid] = 0;
    __syncthreads();

    while (gtid < count)
    {
        powers[gtid] = 1.0 / pow((double)(gtid+1), alpha);
        localSums[tid] += powers[gtid];
        gtid += gtnum;
    }
    __syncthreads();

    /*do not use atomicAdd because atomic funcs with double values
     * are not supported on sm_35*/
    if(0 == threadIdx.x)
    {
        for(int i = 0; i < BLOCK_SIZE_GEN; i++)
        {
            sharedSum += localSums[i];
        }
        totalSum[bid] = sharedSum;
    }
}

template<typename DataType, typename CntType, typename RangeTypeS, typename RangeTypeE>
__global__
void uniform_computation(
        DataType *data, CntType count, RangeTypeE range, RangeTypeS rangeStart,
        curandState *state)
{
    auto tid = CntType(threadIdx.x + blockIdx.x * blockDim.x);
    auto tnum = CntType(blockDim.x * gridDim.x);
    curandState *myState = &state[tid];

    while (tid < count)
    {
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
        DataType *data, CntType count,
        RangeTypeS rangeStart, RangeTypeE rangeEnd,
        double alpha, unsigned long long seed,
        CUDAMemStat *memstat)
{
    static int delta = 0;  /*to ensure each time different sets of values are generated*/
    CntType *mapping = nullptr;

    if (!data)
    {
        log_error("Memory object for data generator is not allocated.");
        exit(1);
    }
    auto range = rangeEnd - (RangeTypeE)rangeStart;

    double *powers, *totalPower, *acc_sums;
    curandState *randStates;
    CUDA_MALLOC(&powers, sizeof(double)*range, memstat);
    CUDA_MALLOC(&totalPower, sizeof(double)*GRID_SIZE_GEN, memstat);
    CUDA_MALLOC(&randStates, sizeof(curandState)*GRID_SIZE_GEN*BLOCK_SIZE_GEN, memstat);
    CUDA_MALLOC(&acc_sums, sizeof(double)*range, memstat);

    setup<<<GRID_SIZE_GEN, BLOCK_SIZE_GEN>>>(randStates, seed+delta);
    power_computation <<<GRID_SIZE_GEN, BLOCK_SIZE_GEN>>>(powers, range, alpha, totalPower);
    checkCudaErrors(cudaDeviceSynchronize());

    /*compute the acc_sums according to powers*/
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, powers, acc_sums, range);
    CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, memstat);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, powers, acc_sums, range);
    checkCudaErrors(cudaDeviceSynchronize());

    CUDA_FREE(d_temp_storage, memstat);
    CUDA_FREE(powers, memstat);

    double sumTotal = 0;
    for(auto i = 0; i < GRID_SIZE_GEN; i++)    sumTotal += totalPower[i];
    sumTotal = 1.0 / sumTotal;

    /*construct the random permutation mapping using Knuth shuffle*/
    CUDA_MALLOC(&mapping, sizeof(CntType)*range, memstat);
#pragma omp parallel for
    for(auto i = 0; i < range; i++) mapping[i] = i;

    for(auto i = range - 1; i >= 1; i--)
    {
        auto j = rand() % (i+1);
        std::swap(mapping[i], mapping[j]);
    }

    zipf_computation <<<GRID_SIZE_GEN, BLOCK_SIZE_GEN>>>(data, mapping, count, range, rangeStart, acc_sums, sumTotal, randStates);
    cudaDeviceSynchronize();

    log_info("Data init finished using GPU. Num: %u, size: %.2f GB, range: %u--%u, type: zipf(z=%f).", count, 1.0*count/1024/1024/1024*sizeof(DataType), rangeStart, rangeEnd, alpha);

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

    CUDA_FREE(acc_sums, memstat);
    CUDA_FREE(totalPower, memstat);
    CUDA_FREE(randStates, memstat);
    CUDA_FREE(mapping, memstat);

    delta ++;
}

template<typename DataType, typename CntType, typename RangeTypeS, typename RangeTypeE>
void uniform_generator_GPU(
        DataType *data, CntType count,
        RangeTypeS rangeStart, RangeTypeE rangeEnd,
        unsigned long long seed, CUDAMemStat *memstat)
{
    static int delta = 0;  /*to ensure each time different sets of values are generated*/

    if (!data)
    {
        log_error("Memory object for data generator is not allocated.");
        exit(1);
    }
    auto range = rangeEnd - (RangeTypeE)rangeStart;

    curandState *randStates;
    CUDA_MALLOC(&randStates, sizeof(curandState)*GRID_SIZE_GEN*BLOCK_SIZE_GEN, memstat);

    setup<<<GRID_SIZE_GEN, BLOCK_SIZE_GEN>>>(randStates, seed+delta);
    uniform_computation <<<GRID_SIZE_GEN, BLOCK_SIZE_GEN>>>(data, count, range, rangeStart, randStates);
    cudaDeviceSynchronize();

    log_info("Data init finished using GPU. Num: %u, size: %.2f GB, range: %u-%u, type: uniform.", count, 1.0*count/1024/1024/1024*sizeof(DataType), rangeStart, rangeEnd);

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

    CUDA_FREE(randStates, memstat);
    delta ++;
}

template<typename DataType, typename CntType, typename RangeTypeS, typename RangeTypeE>
void uniformUniquePairGeneratorCUDA(
        DataType *key0, DataType *key1, CntType count,
        RangeTypeS range_start_key0, RangeTypeE range_end_key0,
        RangeTypeS range_start_key1, RangeTypeE range_end_key1,
        unsigned long long seed, CUDAMemStat *memstat) {
    static int delta = 0;  /*to ensure each time different sets of values are generated*/

    if ((!key0) || (!key1)) {
        log_error("Memory object for data generator is not allocated.");
        exit(1);
    }
    auto range_key0 = range_end_key0 - (RangeTypeE)range_start_key0;
    auto range_key1 = range_end_key1 - (RangeTypeE)range_start_key1;
    assert(1.0 * range_key0 * range_key1 > 1.0 * count);

    set<PairItem<DataType>> item_set;

    /*init the data using CPU*/
    srand((unsigned)(seed+delta));
    for(auto i = 0; i < count; i++) {
        auto prev_cnt = item_set.size();
        do {
            DataType temp_key_0 = (rand() % range_key0) + range_start_key0;
            DataType temp_key_1 = (rand() % range_key1) + range_start_key1;
            item_set.insert({temp_key_0,temp_key_1});
        } while (item_set.size() == prev_cnt);
    }
#ifdef UM
    CntType idx = 0;
    for(auto it = item_set.begin(); it != item_set.end(); it++, idx++) {
        key0[idx] = it->key0;
        key1[idx] = it->key1;
    }
    /*shuffle the elements*/
    for(auto i = count - 1; i >= 1; i--) {
        auto j = rand() % (i+1);
        std::swap(key0[i], key0[j]);
        std::swap(key1[i], key1[j]);
    }
    checkCudaErrors(cudaMemPrefetchAsync(key0, sizeof(DataType)*count, DEVICE_ID));
    checkCudaErrors(cudaMemPrefetchAsync(key1, sizeof(DataType)*count, DEVICE_ID));
#else
    DataType *key0_cpu = (DataType*)malloc(sizeof(DataType)*count);
    DataType *key1_cpu = (DataType*)malloc(sizeof(DataType)*count);
    CntType idx = 0;
    for(auto it = item_set.begin(); it != item_set.end(); it++, idx++) {
        key0_cpu[idx] = it->key0;
        key1_cpu[idx] = it->key1;
    }
    /*shuffle the elements*/
    for(auto i = count - 1; i >= 1; i--) {
        auto j = rand() % (i+1);
        std::swap(key0_cpu[i], key0_cpu[j]);
        std::swap(key1_cpu[i], key1_cpu[j]);
    }
    checkCudaErrors(cudaMemcpy(key0, key0_cpu, sizeof(DataType)*count, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(key1, key1_cpu, sizeof(DataType)*count, cudaMemcpyHostToDevice));
#endif
    delta++;

    log_info("Data init finished using CPU. Num: %u, size: %.2f GB, range(key0): %u-%u, range(key1): %u-%u, type: uniform.", count, 1.0*count/1024/1024/1024*sizeof(DataType), range_start_key0, range_end_key0, range_start_key1, range_end_key1);
}

template<typename DataType, typename CntType, typename RangeTypeS, typename RangeTypeE>
void uniformUniquePairGenerator(
        DataType *key0, DataType *key1, CntType count,
        RangeTypeS range_start_key0, RangeTypeE range_end_key0,
        RangeTypeS range_start_key1, RangeTypeE range_end_key1) {
    Timer t;
    static int delta = 0;  /*to ensure each time different sets of values are generated*/
    if ((!key0) || (!key1)) {
        log_error("Memory object for data generator is not allocated.");
        exit(1);
    }

    int max_num_threads = omp_get_max_threads();
    auto range_key0 = range_end_key0 - (RangeTypeE)range_start_key0;
    auto range_key1 = range_end_key1 - (RangeTypeE)range_start_key1;
    assert(1.0 * range_key0 * range_key1 > 1.0 * count);
    auto range_key0_per_thread = range_key0 / max_num_threads;
    auto count_percentage = count / 10;

    /*init the data using CPU using multi-thread*/
    set<PairItem<DataType>> item_set[MAX_NUM_CPU_THREADS];
#pragma omp parallel
{
    auto tid = omp_get_thread_num();
    auto l_range_start_key0 = range_key0_per_thread * tid;
    unsigned seed = (unsigned)(delta*max_num_threads+tid);

    DataType temp_key_0 = l_range_start_key0 + (rand_r(&seed) % range_key0_per_thread);
    DataType temp_key_1 = range_start_key1 + (rand_r(&seed) % range_key1);
#pragma omp for schedule(dynamic) nowait
    for(CntType i = 0; i < count; i++) {
        auto prev_cnt = item_set[tid].size();
        item_set[tid].insert({temp_key_0,temp_key_1}); //try to insert
        while (item_set[tid].size() == prev_cnt) { //should use rand_r() for parallel random generation instead of rand()
            temp_key_0 = (temp_key_0 - l_range_start_key0 + (rand_r(&seed) % range_key0_per_thread)) % range_key0_per_thread + l_range_start_key0;
            temp_key_1 = (temp_key_1 - range_start_key1 + (rand_r(&seed) % range_key1)) % range_key1 + range_start_key1;
            item_set[tid].insert({temp_key_0,temp_key_1}); //try to insert until success
        }
        temp_key_0 = (temp_key_0 - l_range_start_key0 + (rand_r(&seed) % range_key0_per_thread)) % range_key0_per_thread + l_range_start_key0;
        temp_key_1 = (temp_key_1 - range_start_key1 + (rand_r(&seed) % range_key1)) % range_key1 + range_start_key1;
#ifdef VERBOSE
        if((i % count_percentage) == 0)
            log_info("Initialized %.0f%% of the data, idx=%llu", 10.0*i/count_percentage,i);
#endif
    }
}
    CntType idx = 0;
    for(auto s = 0; s < max_num_threads; s++) {
        for(auto it = item_set[s].begin(); it != item_set[s].end(); it++, idx++) {
            key0[idx] = it->key0;
            key1[idx] = it->key1;
        }
    }
#ifdef UNSORTED_DATA
    /*shuffle the elements*/
    log_info("Begin shuffling");
    for(CntType i = count - 1; i >= 1; i--) {
        auto j = rand() % (i+1);
        std::swap(key0[i], key0[j]);
        std::swap(key1[i], key1[j]);
    }
    log_info("Finish shuffling");
#else
    log_info("Generated sorted data");
#endif

    delta++;

    log_info("Data init finished. Num: %llu, size: %.2f GB, range(key0): %llu-%llu, range(key1): %llu-%llu, type: uniform.", count, 1.0*count/1024/1024/1024*sizeof(DataType)*2, range_start_key0, range_end_key0, range_start_key1, range_end_key1);
    log_info("Generating data time: %.2f s", t.elapsed());
}

/*generating (0,0),(0,1),(0,2),...,(1,0),(1,1),... for debugging*/
template<typename DataType, typename CntType, typename RangeTypeS, typename RangeTypeE>
void uniformUniqueSeqPairGenerator(
        DataType *key0, DataType *key1, CntType count,
        RangeTypeS range_start_key0, RangeTypeE range_end_key0,
        RangeTypeS range_start_key1, RangeTypeE range_end_key1) {
    Timer t;
    static int delta = 0;  /*to ensure each time different sets of values are generated*/
    if ((!key0) || (!key1)) {
        log_error("Memory object for data generator is not allocated.");
        exit(1);
    }

    auto vals_per_key0 = count / (range_end_key0-range_start_key0);

#pragma omp parallel for
    for(CntType idx = 0; idx < count; idx++) {
        key0[idx] = range_start_key0 + DataType(idx/vals_per_key0);
        key1[idx] = DataType(idx%vals_per_key0);
    }

    printf("First 10 items: ");
    for(CntType i = 0; i < 10; i++) {
        printf("(%d,%d), ", key0[i], key1[i]);
    }
    printf("\n");
    printf("Last 10 items: ");
    for(CntType i = count-10; i < count; i++) {
        printf("(%d,%d), ", key0[i], key1[i]);
    }
    printf("\n");

#ifdef UNSORTED_DATA
    /*shuffle the elements*/
    log_info("Begin shuffling");
    for(CntType i = count - 1; i >= 1; i--) {
        auto j = rand() % (i+1);
        std::swap(key0[i], key0[j]);
        std::swap(key1[i], key1[j]);
    }
    log_info("Finish shuffling");
#else
    log_info("Generated sorted data");
#endif

    delta++;

    log_info("Data init finished. Num: %llu, size: %.2f GB, range(key0): %llu-%llu, range(key1): %llu-%llu, type: uniform.", count, 1.0*count/1024/1024/1024*sizeof(DataType)*2, range_start_key0, range_end_key0, range_start_key1, range_end_key1);
    log_info("Generating data time: %.2f s", t.elapsed());
}

/*generate skewed data for a single column*/
template<typename DataType, typename CntType, typename RangeTypeS, typename RangeTypeE>
void zipfUniquePairGenerator(
        DataType *key0, DataType *key1, CntType count,
        RangeTypeS range_start_key0, RangeTypeE range_end_key0,
        RangeTypeS range_start_key1, RangeTypeE range_end_key1,
        unsigned long long seed, char skew_col, double z) {
    assert(skew_col < 2);
    static int delta = 0;  /*to ensure each time different sets of values are generated*/

    if ((!key0) || (!key1)) {
        log_error("Memory object for data generator is not allocated.");
        exit(1);
    }

    auto skewed_col = (skew_col == 0) ? key0 : key1;
    auto uniform_col = (skew_col == 1) ? key0 : key1;
    auto skewed_range_start = (skew_col == 0) ? range_start_key0 : range_start_key1;
    auto skewed_range_end = (skew_col == 0) ? range_end_key0 : range_end_key1;
    auto uniform_range_start = (skew_col == 1) ? range_start_key0 : range_start_key1;
    auto uniform_range_end = (skew_col == 1) ? range_end_key0 : range_end_key1;
    auto skew_range = skewed_range_end - skewed_range_start;
    auto uniform_range = uniform_range_end - uniform_range_start;
    assert(1.0 * skew_range * uniform_range > 1.0 * count);

    /*1.generate the skewed column in key0*/
    set<PairItem<DataType>> item_set;
    zipf_generator_GPU(skewed_col, count, skewed_range_start, skewed_range_end, z, seed + delta, nullptr);
    thrust::sort(thrust::device, skewed_col, skewed_col+count);
    cudaDeviceSynchronize();

    /*init the uniform column*/
    srand((unsigned)(seed+delta));
    for(CntType i = 0; i < count; i++) {
        auto prev_cnt = item_set.size();
        DataType temp_uniform_key;
        do {
            temp_uniform_key = (rand() % uniform_range) + uniform_range_start;
            item_set.insert({skewed_col[i],temp_uniform_key});
        } while (item_set.size() == prev_cnt);
#ifdef VERBOSE
        log_info("Insert item %llu, key1=%d, val=%d", i, skewed_col[i], temp_uniform_key);
#endif
    }
    CntType idx = 0;
    for(auto it = item_set.begin(); it != item_set.end(); it++, idx++) {
        skewed_col[idx] = it->key0;
        uniform_col[idx] = it->key1;
    }

#ifdef UNSORTED_DATA
    /*shuffle the elements*/
    log_info("Begin shuffling");
    for(CntType i = count - 1; i >= 1; i--) {
        auto j = rand() % (i+1);
        std::swap(skewed_col[i], skewed_col[j]);
        std::swap(uniform_col[i], uniform_col[j]);
    }
    log_info("Finish shuffling");
#else
    log_info("Generated sorted data");
#endif
    delta++;

    log_info("Data init finished using CPU. Num: %u, size: %.2f GB, range(key0): %u-%u, range(key1): %u-%u, type: zipf, skewed col: %d.", count, 1.0*count/1024/1024/1024*sizeof(DataType), range_start_key0, range_end_key0, range_start_key1, range_end_key1, skew_col);
}


#undef GRID_SIZE_GEN
#undef BLOCK_SIZE_GEN