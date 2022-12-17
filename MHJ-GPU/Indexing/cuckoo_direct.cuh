//
// Created by Bryan on 1/6/2020.
//
#pragma once

#include "cuda/primitives.cuh"
#include "radix_partitioning.cuh"
#include "cuda/sharedMem.cuh"
#include "timer.h"
#include "../Relation.cuh"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdlib>

/** Max rehashing depth, and error depth. */
#define MAX_DEPTH (100)
#define ERR_DEPTH (-1)
#define MY_PRIME    (334214459)

/** Struct of a hash function config. */
struct FuncConfigDirect {
    int a;
    int b;
};

template <typename DataType, typename CntType>
inline __host__ __device__ DataType
hash_direct(const DataType val, const FuncConfigDirect *const hash_func_configs,
            const int func_idx, const CntType size) {
    FuncConfigDirect fc = hash_func_configs[func_idx];
    return (DataType)(((fc.a*val + fc.b) % MY_PRIME) % size);
}

template <typename DataType>
inline __device__ DataType
make_data_direct(const DataType val, const int func, const int pos_width) {
    return (val << pos_width) ^ func;   // CANNOT handle signed values currently!
}

template <typename DataType>
inline __device__ DataType
fetch_val_direct(const DataType data, const int pos_width) {
    return data >> pos_width;
}

template <typename DataType>
inline __device__ int
fetch_func_direct(const DataType data, const int pos_width) {
    return data & ((0x1 << pos_width) - 1);
}

/*
 * Inserting key-value pairs, with unique keys
 * */
template<typename KType, typename VType, typename CntType>
__global__ void
cuckoo_direct_insert_kernel(KType *keys_in, VType *values_in, CntType num,
                     KType *keys_out, VType *values_out, CntType ht_size,
                     FuncConfigDirect *hash_func_configs, int num_funcs,
                     int evict_bound, int pos_width,
                     int *rehash_requests) {
    KType old_key;
    VType old_val;
    CntType gtid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gtid < num) {
        auto cur_key = keys_in[gtid];
        auto cur_val = values_in[gtid];
        int cur_func = 0;
        int evict_count = 0;

        do { //the test-kick-and-reinsert loops
            auto pos = hash_direct(cur_key, hash_func_configs, cur_func, ht_size);
            while ((old_key = atomicExch(&keys_out[pos], INVALID_KEY)) == INVALID_KEY);
            if (old_key != EMPTY_CELL) { //have data previously
                volatile auto temp_key = fetch_val_direct(old_key, pos_width);
                old_val = values_out[pos];
                values_out[pos] = cur_val; //exchange the value
                __threadfence();
                keys_out[pos] = make_data_direct(cur_key, cur_func, pos_width); //exchange the key
                cur_val = old_val;
                cur_key = temp_key;
                evict_count++;
                cur_func = (fetch_func_direct(old_key, pos_width) + 1) % num_funcs;
            } else { //insert to an empty slot
                values_out[pos] = cur_val;
                keys_out[pos] = make_data_direct(cur_key, cur_func, pos_width);
                break;
            }
        } while (evict_count < evict_bound);

        /*If exceeds eviction bound, then needs rehashing*/
        if (evict_count >= evict_bound) {
            atomicAdd(rehash_requests, 1);
        }
    }
}

template<typename KType, typename VType, typename CntType>
__global__ void
cuckoo_direct_lookup_kernel(KType *lookup_keys, VType *lookup_values, CntType num,
                     KType *ht_keys, VType *ht_values, CntType ht_size,
                     FuncConfigDirect *hash_func_configs, int num_funcs, int pos_width) {
    CntType gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < num) {
        auto cur_key = lookup_keys[gtid];
        for(auto i = 0; i < num_funcs; i++) {
            auto pos = hash_direct(cur_key, hash_func_configs, i, ht_size);
            if (fetch_val_direct(ht_keys[pos], pos_width) == cur_key) {
                lookup_values[gtid] = ht_values[pos];
                return;
            }
        }
        lookup_values[gtid] = INT32_MAX; //not found
    }
}

template<typename KType, typename VType, typename CntType>
struct CuckooHashTableDirect : CuckooHashTable<KType,VType,CntType> {
    /*using base class's members*/
    using CuckooHashTable<KType,VType,CntType>::_cuckoo_cardinality;
    using CuckooHashTable<KType,VType,CntType>::_evict_bound;
    using CuckooHashTable<KType,VType,CntType>::_num_funcs;
    using CuckooHashTable<KType,VType,CntType>::_pos_width;
    using CuckooHashTable<KType,VType,CntType>::_keys;
    using CuckooHashTable<KType,VType,CntType>::_values;
    using CuckooHashTable<KType,VType,CntType>::_memstat;
    using CuckooHashTable<KType,VType,CntType>::_timing;

    /*my own members*/
    Relation<KType,CntType> *_origin;   //pointer to the original Relation
    int _hash_attr;                     //hash attr in the original relation

    /* Cuckoo hash function set */
    FuncConfigDirect *_hash_func_configs;

    /*my own functions*/
    void gen_hash_funcs();

    /** Inline helper functions. */
    inline KType fetch_val_direct(const KType data) {
        return data >> _pos_width;
    }
    inline int fetch_func_direct(const KType data) {
        return data & ((0x1 << _pos_width) - 1);
    }

    void init(const CntType cuckoo_card, int evict_bound, const int num_funcs,
              Relation<KType,CntType> *origin, int hash_attr,
              CUDAMemStat *memstat, CUDATimeStat *timing);
    CuckooHashTableDirect(const CntType cuckoo_card, const int evict_bound, const int num_funcs,
                         Relation<KType,CntType> *origin, int hash_attr,
                         CUDAMemStat *memstat, CUDATimeStat *timing);
    ~CuckooHashTableDirect();

    FuncConfigDirect *get_hash_func_configs() {
        return _hash_func_configs;
    }

    /*insert the lookup functions*/
    int insert_vals(KType *keys, VType *values, CntType n);
    void lookup_vals(KType *keys, VType *values, CntType n);
    void show_content();
};

/* generate hash functions */
template<typename KType, typename VType, typename CntType>
void CuckooHashTableDirect<KType,VType,CntType>::
gen_hash_funcs() {
    for (int i = 0; i < _num_funcs; ++i) { // Generate randomized configurations.
        _hash_func_configs[i] = {rand(), rand()};
    }
}

/*for objs generated through CudaMallocManaged which will not invoke the constructor*/
template<typename KType, typename VType, typename CntType>
void CuckooHashTableDirect<KType,VType,CntType>::
init(const CntType cuckoo_card, int evict_bound, const int num_funcs,
          Relation<KType,CntType> *origin, int hash_attr,
          CUDAMemStat *memstat, CUDATimeStat *timing) {
    _cuckoo_cardinality = cuckoo_card;
    _num_funcs = num_funcs;
    _pos_width = ceil(log2((double) _num_funcs));
    _origin = origin;
    _hash_attr = hash_attr;
    _memstat = memstat;
    _timing = timing;
    _evict_bound = evict_bound;

    /*Allocate space for data table and hash function configs*/
    log_info("Cuckoo card: %llu, num_func: %d", _cuckoo_cardinality, _num_funcs);
    CUDA_MALLOC(&_keys, sizeof(KType)*cuckoo_card, _memstat);
    checkCudaErrors(cudaMemset(_keys, 0xff, sizeof(KType)*cuckoo_card)); //important: init to EMPTY_CELL
    CUDA_MALLOC(&_values, sizeof(VType)*cuckoo_card, _memstat);
    CUDA_MALLOC(&_hash_func_configs, sizeof(FuncConfigDirect)*_num_funcs, _memstat);

    /*Generate initial hash function configs*/
    gen_hash_funcs();
}

/* Constructor */
template<typename KType, typename VType, typename CntType>
CuckooHashTableDirect<KType,VType,CntType>::
CuckooHashTableDirect(const CntType cuckoo_card, const int evict_bound, const int num_funcs,
                     Relation<KType,CntType> *origin, int hash_attr,
                     CUDAMemStat *memstat, CUDATimeStat *timing) {
    init(cuckoo_card, evict_bound, num_funcs, origin, hash_attr, memstat, timing);
}

template<typename KType, typename VType, typename CntType>
CuckooHashTableDirect<KType,VType,CntType>::
~CuckooHashTableDirect() {
    CUDA_FREE(_keys, _memstat);
    CUDA_FREE(_values, _memstat);
    CUDA_FREE(_hash_func_configs, _memstat);
}

template<typename KType, typename VType, typename CntType>
int CuckooHashTableDirect<KType,VType,CntType>::
insert_vals(KType *keys, VType *values, CntType n) {
    Timer t;
    int *rehash_requests = nullptr;
    CUDA_MALLOC(&rehash_requests, sizeof(int), _memstat);

    // Loops until no rehashing needed.
    int rehash_count = 0;
    do {
        checkCudaErrors(cudaMemset(rehash_requests, 0, sizeof(int)));
        int block_size = 128;
        auto grid_size = (n + block_size - 1) / block_size;
        log_info("Insert kernel, grid size: %llu, block size: %llu", grid_size, block_size);
        execKernel(cuckoo_direct_insert_kernel, grid_size, block_size, _timing, false,
                   keys, values, n, _keys, _values, _cuckoo_cardinality,
                   _hash_func_configs, _num_funcs,
                   _evict_bound, _pos_width, rehash_requests);
        if ((*rehash_requests) == 0)
            break;
        else {
            rehash_count++;
            log_info("Rehash");
            gen_hash_funcs();
        }
    } while (rehash_count < MAX_DEPTH);
    if (rehash_count >= MAX_DEPTH) {
        log_error("Too many rehash, exit");
    }
    log_info("Inside cuckoo hashing: %.1f ms", t.elapsed()*1000);
    CUDA_FREE(rehash_requests, _memstat);

    return (rehash_count < MAX_DEPTH) ? rehash_count : ERR_DEPTH;
}

template<typename KType, typename VType, typename CntType>
void CuckooHashTableDirect<KType,VType,CntType>::
lookup_vals(KType *keys, VType *values, CntType n) {
    int block_size = 128;
    auto grid_size = (n + block_size - 1) / block_size;
    execKernel(cuckoo_direct_lookup_kernel, grid_size, block_size, _timing, false,
               keys, values, n, _keys, _values, _cuckoo_cardinality,
               _hash_func_configs, _num_funcs, _pos_width);
}

/** Cuckoo: print content out. */
template<typename KType, typename VType, typename CntType>
void CuckooHashTableDirect<KType,VType,CntType>::
show_content() {
    std::cout << "Funcs: ";
    for (int i = 0; i < _num_funcs; ++i) {
        FuncConfigDirect fc = _hash_func_configs[i];
        std::cout << "(" << fc.a << ", " << fc.b << ") ";
    }
    std::cout << std::endl;
    for(CntType i = 0; i < _cuckoo_cardinality; i++) {
        if (fetch_val_direct(_keys[i]) != -1) {
            std::cout << "(" << fetch_val_direct(_keys[i])
                      <<" "<< _values[i] <<"),";
        }
    }
    std::cout << std::endl;
}