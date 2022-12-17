/* A simple GPU hash table implemented in CUDA using lock free techniques
 * github link: https://github.com/nosferalatu/SimpleGPUHashTable.git
 * */
#pragma once

#include "../types.h"
#include "../Relation.cuh"
#include "helper.h"
#include "../common_kernels.cuh"
#include "timer.h"

#define OA_HT_RATIO        (0.5)

/*Insert the keys and values into the hash table*/
template<typename DataType, typename CntType>
__global__ void gpu_hashtable_insert_KI(
        DataType *hash_keys, CntType *hash_values,
        const DataType *input_keys, CntType num, CntType hash_capacity) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num) {
        auto key = input_keys[tid];
        auto val = tid;
        auto slot = murmur_hash(key, hash_capacity);
        while (true) {
            auto prev = atomicCAS(&hash_keys[slot], EMPTY_HASH_SLOT, key);
            if (prev == EMPTY_HASH_SLOT) {
                hash_values[slot]= val;
                return;
            }
            slot = (slot + 1) % hash_capacity;
        }
    }
}

/*Insert the keys and values into the hash table*/
template<typename KType, typename VType, typename CntType>
__global__ void gpu_hashtable_insert_KV(
        KType *hash_keys, CntType *hash_values,
        const KType *input_keys, const VType *input_values,
        CntType num, CntType hash_capacity) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num) {
        auto key = input_keys[tid];
        auto slot = murmur_hash(key, hash_capacity);
        while (true) {
            auto prev = atomicCAS(&hash_keys[slot], EMPTY_HASH_SLOT, key);
            if (prev == EMPTY_HASH_SLOT) {
                hash_values[slot]= input_values[tid];
                return;
            }
            slot = (slot + 1) % hash_capacity;
        }
    }
}

template<typename KType, typename VType, typename CntType>
__global__ void gpu_hashtable_lookup_KV(
        KType *hash_keys, VType *hash_values,
        KType *lookup_keys, VType *lookup_values,
        CntType num, CntType hash_capacity) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num) {
        auto key = lookup_keys[tid];
        auto slot = murmur_hash(key, hash_capacity);
        while (hash_keys[slot] != EMPTY_HASH_SLOT) {
            if (hash_keys[slot] == key) {
                lookup_values[tid] = hash_values[slot];
                break;
            }
            slot = (slot + 1) % hash_capacity;
        }
    }
}

/*---------------- Host functions ------------------*/

/* temporary use: build hash table ht for relation rel*/
template<typename DataType, typename CntType>
bool build_hash_openaddr_KI(
        Relation<DataType, CntType> *rel, HashTable<DataType, CntType> &ht,
        CUDAMemStat *memstat, CUDATimeStat *timing) {

    auto capacity = ceilPowerOf2(rel->length / OA_HT_RATIO);
    assert(isPowerOf2<CntType>(capacity)); //capacity should be a power of 2
    ht.capacity = capacity;
    ht.length = rel->length;
    ht.hash_attr = rel->attr_list[0]; //first attr is the hash attr

    CUDA_MALLOC(&ht.hash_keys, sizeof(DataType)*capacity, memstat);
    CUDA_MALLOC(&ht.idx_in_origin, sizeof(CntType)*capacity, memstat);
    checkCudaErrors(cudaMemset(ht.hash_keys, 0xff, sizeof(DataType) * capacity)); //init to 0xffffffff

    int mingridsize, block_size;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &block_size, gpu_hashtable_insert_KI<DataType, CntType>, 0, 0);
    auto grid_size = (ht.length + block_size - 1) / block_size;

    cudaDeviceSynchronize();
    execKernel((gpu_hashtable_insert_KI<DataType,CntType>), grid_size, block_size, timing, false, ht.hash_keys, ht.idx_in_origin, rel->data[0], ht.length, ht.capacity);

    return true;
}

template<typename KType, typename VType, typename CntType>
class LinearHashTable {
public:
    CntType _table_card;            //cardinality of the hash table

    /* Actual key-value data*/
    KType *_keys;
    VType *_values;

    CUDAMemStat *_memstat;
    CUDATimeStat *_timing;

    LinearHashTable(const CntType ht_card, CUDAMemStat *memstat, CUDATimeStat *timing)
            : _table_card(ht_card), _memstat(memstat), _timing(timing) {

        /*Allocate space for data table */
        CUDA_MALLOC(&_keys, sizeof(KType)*_table_card, _memstat);
        CUDA_MALLOC(&_values, sizeof(VType)*_table_card, _memstat);
        checkCudaErrors(cudaMemset(_keys, 0xff, sizeof(KType) * _table_card)); //init to 0xffffffff
    }

    void insert_vals(KType *keys, VType *values, CntType n);
    void lookup_vals(KType *keys, VType *values, CntType n);
    void show_content();
};

template<typename KType, typename VType, typename CntType>
void
LinearHashTable<KType,VType,CntType>::insert_vals(KType *keys, VType *values, CntType n) {
    int mingridsize, block_size;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &block_size,
                                       gpu_hashtable_insert_KV<KType,VType,CntType>, 0, 0);
    auto grid_size = (n + block_size - 1) / block_size;
    cudaDeviceSynchronize();
    execKernel((gpu_hashtable_insert_KV<KType,VType,CntType>), grid_size, block_size, _timing, false,
               _keys, _values, keys, values, n, _table_card);
};

template<typename KType, typename VType, typename CntType>
void
LinearHashTable<KType,VType,CntType>::lookup_vals(KType *keys, VType *values, CntType n) {
    int mingridsize, block_size;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &block_size,
                                       gpu_hashtable_lookup_KV<KType,VType,CntType>, 0, 0);
    auto grid_size = (n + block_size - 1) / block_size;
    cudaDeviceSynchronize();
    execKernel((gpu_hashtable_lookup_KV<KType,VType,CntType>), grid_size, block_size, _timing, false,
               _keys, _values, keys, values, n, _table_card);
};

/** LinearHashTable: print content out. */
template<typename KType, typename VType, typename CntType>
void
LinearHashTable<KType,VType,CntType>::show_content() {
    for (int j = 0; j < _table_card; ++j) {
        if (_keys[j] != EMPTY_HASH_SLOT) {
            std::cout << "(" << _keys[j] <<" "<< _values[j] <<" "<<j<<"),";
        }
    }
    std::cout << std::endl;
}