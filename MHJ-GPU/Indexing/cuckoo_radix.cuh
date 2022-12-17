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

#define CUCKOO_HT_RATIO    (0.95)

/** Max rehashing depth, and error depth. */
#define MAX_DEPTH (100)
#define ERR_DEPTH (-1)

/** Size of each bucket in the cuckoo hash table. */
#define BUCKET_SIZE     (512)
#define NUM_HASH_FUNC   (4)
#define ITEMS_PER_FUNC  (BUCKET_SIZE/NUM_HASH_FUNC)

/** Struct of a hash function config. */
struct FuncConfig {
    int rv;
    int ss;
};

template <typename DataType>
inline __host__ __device__ DataType
do_2nd_hash(const DataType val, const FuncConfig * const hash_func_configs, const int func_idx) {
    FuncConfig fc = hash_func_configs[func_idx];
    return (DataType)(ITEMS_PER_FUNC*func_idx + ((val ^ fc.rv) >> fc.ss) % ITEMS_PER_FUNC);
}

template <typename DataType>
inline __device__ DataType
make_data(const DataType val, const int func, const int pos_width) {
    return (val << pos_width) ^ func;   // CANNOT handle signed values currently!
}

template <typename DataType>
inline __device__ DataType
fetch_val(const DataType data, const int pos_width) {
    return data >> pos_width;
}

template <typename DataType>
inline __device__ int
fetch_func(const DataType data, const int pos_width) {
    return data & ((0x1 << pos_width) - 1);
}

/*
 * Inserting key-value pairs, with unique keys
 * */
template<typename KType, typename VType, typename CntType>
__global__ void
cuckoo_insert_kernel(KType *keys_in, VType *values_in,
                     KType *keys_out, VType *values_out,
                     FuncConfig *hash_func_configs, int num_funcs,
                     CntType *buc_ptrs, int evict_bound, int pos_width,
                     int *rehash_requests) {
    //todo: if this bucket has more than BUCKET_SIZE items, exit and process in another pass
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    CntType gtid = tid + bid * blockDim.x;
    KType old_key;
    VType old_val;

    SharedMemory<KType> smem; //use SharedMemory because extern shared memory does not support template
    KType *local_data = smem.getPointer();

    KType *local_keys = local_data;
    VType *local_vals = (VType *)&local_data[BUCKET_SIZE];
    auto bucket_size = buc_ptrs[bid+1] - buc_ptrs[bid];
    assert(bucket_size <= BUCKET_SIZE);

    local_keys[tid] = EMPTY_CELL; //init ALL slots for keys to EMPTY_CELL
    __syncthreads();

    if(tid < bucket_size) {  //todo: skewing handling when bucket_size is very large
        auto cur_key = keys_in[buc_ptrs[bid]+tid];
        auto cur_val = values_in[buc_ptrs[bid]+tid];
        int cur_func = 0;
        int evict_count = 0;

        do { // The test-kick-and-reinsert loops.
            auto pos = do_2nd_hash(cur_key, hash_func_configs, cur_func);
            while ((old_key = atomicExch(&local_keys[pos], INVALID_KEY)) == INVALID_KEY);
            if (old_key != EMPTY_CELL) { //have data previously
                volatile auto temp_key = fetch_val(old_key, pos_width);
                old_val = local_vals[pos];
                local_vals[pos] = cur_val; //exchange the value
                __threadfence();
                local_keys[pos] = make_data(cur_key, cur_func, pos_width); //exchange the key
                cur_val = old_val;
                cur_key = temp_key;
                evict_count++;
                cur_func = (fetch_func(old_key, pos_width) + 1) % num_funcs;
            } else { //insert to an empty slot
                local_vals[pos] = cur_val;
                local_keys[pos] = make_data(cur_key, cur_func, pos_width);
                break;
            }
        } while (evict_count < num_funcs * evict_bound);

        /*If exceeds eviction bound, then needs rehashing*/
        if (evict_count >= num_funcs * evict_bound) {
            atomicAdd(rehash_requests, 1);
        }
    }

    /*Every thread write its responsible local slot into the global data table*/
    __syncthreads();

    keys_out[gtid] = local_keys[tid];
    values_out[gtid] = local_vals[tid];
}

template<typename KType, typename VType, typename CntType>
__global__ void
cuckoo_lookup_kernel(KType *lookup_keys, VType *lookup_values, CntType card,
                     KType *cuckoo_table_keys, VType *cuckoo_table_values,
                     FuncConfig *hash_func_configs, int num_funcs,
                     CntType num_buckets, int pos_width) {
    CntType gtid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gtid < card) {
        auto cur_key = lookup_keys[gtid];
        auto bucket_id = cur_key % num_buckets;
        for(auto i = 0; i < num_funcs; i++) {
            auto pos = bucket_id * BUCKET_SIZE + do_2nd_hash(cur_key, hash_func_configs, i);
            if (fetch_val(cuckoo_table_keys[pos], pos_width) == cur_key) {
                lookup_values[gtid] = cuckoo_table_values[pos];
                return;
            }
        }
        lookup_values[gtid] = INT32_MAX;
    }
}

template<typename KType, typename VType, typename CntType>
struct CuckooHashTableRadix : CuckooHashTable<KType,VType,CntType> {
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
    CntType _num_buckets;               //number of buckets used for partitioning in phase 1
    Relation<KType,CntType> *_origin;   //pointer to the original Relation
    int _hash_attr;                     //hash attr in the original relation

    /* Cuckoo hash function set */
    FuncConfig *_hash_func_configs;

    /*my own functions*/
    void gen_hash_funcs();

    /** Inline helper functions. */
    inline KType fetch_val(const KType data) {
        return data >> _pos_width;
    }
    inline int fetch_func(const KType data) {
        return data & ((0x1 << _pos_width) - 1);
    }

    void init(CntType input_card, float cuckoo_ratio, const int evict_bound,
              Relation<KType,CntType> *origin, int hash_attr,
              CUDAMemStat *memstat, CUDATimeStat *timing);
    CuckooHashTableRadix(CntType input_card, float cuckoo_ratio, const int evict_bound,
                         Relation<KType,CntType> *origin, int hash_attr,
                         CUDAMemStat *memstat, CUDATimeStat *timing);
    ~CuckooHashTableRadix();

    CntType get_num_buckets() {
        return _num_buckets;
    }
    FuncConfig *get_hash_func_configs() {
        return _hash_func_configs;
    }

    /*insert the lookup functions*/
    int insert_kvs(KType *keys, VType *values, CntType n);
    int insert_ki(KType *keys, CntType n);
    void lookup_vals(KType *keys, VType *values, CntType n);
    void show_content();
};

/* generate hash functions */
template<typename KType, typename VType, typename CntType>
void CuckooHashTableRadix<KType,VType,CntType>::
gen_hash_funcs() {
    // Calculate bit width of value range and table size.
    int key_width = 8 * sizeof(KType) - ceil(log2((double) _num_funcs));
    int bucket_width = ceil(log2((double) _num_buckets));
    int size_width = ceil(log2((double) BUCKET_SIZE));

    /*Generate randomized configurations*/
    for (int i = 0; i < _num_funcs; ++i) {  // At index 0 is a dummy function.
        if (key_width - bucket_width <= size_width) {
            _hash_func_configs[i] = {rand(), 0};
        }
        else {
            _hash_func_configs[i] = {rand(),
                                     rand() % (key_width - bucket_width - size_width + 1) + bucket_width};
        }
    }
}

/*for objs generated through CudaMallocManaged which will not invoke the constructor*/
template<typename KType, typename VType, typename CntType>
void CuckooHashTableRadix<KType,VType,CntType>::
init(CntType input_card, float cuckoo_ratio, const int evict_bound,
     Relation<KType,CntType> *origin, int hash_attr,
     CUDAMemStat *memstat, CUDATimeStat *timing) {
    auto partition_size_exp = (CntType)(BUCKET_SIZE * cuckoo_ratio);
    _num_buckets = ceilPowerOf2((input_card + partition_size_exp - 1) / partition_size_exp);
    _num_funcs = NUM_HASH_FUNC;
    _cuckoo_cardinality = _num_buckets * BUCKET_SIZE;
    _evict_bound = evict_bound;
    _pos_width = (int)ceil(log2((double) _num_funcs));
    _origin = origin;
    _hash_attr = hash_attr;
    _memstat = memstat;
    _timing = timing;

    /*Allocate space for data table and hash function configs*/
    log_info("Input card=%llu, Cuckoo card=%llu, num_func=%d, buckets=%llu, evict_bound=%d",
             input_card, _cuckoo_cardinality, _num_funcs, _num_buckets, _evict_bound);
    CUDA_MALLOC(&_keys, sizeof(KType)*_cuckoo_cardinality, _memstat);
    CUDA_MALLOC(&_values, sizeof(VType)*_cuckoo_cardinality, _memstat);
    CUDA_MALLOC(&_hash_func_configs, sizeof(FuncConfig)*_num_funcs, _memstat);

    /*Generate initial hash function configs*/
    gen_hash_funcs();
}

/* Constructor */
template<typename KType, typename VType, typename CntType>
CuckooHashTableRadix<KType,VType,CntType>::
CuckooHashTableRadix(CntType input_card, float cuckoo_ratio, const int evict_bound,
                     Relation<KType,CntType> *origin, int hash_attr,
                     CUDAMemStat *memstat, CUDATimeStat *timing) {
    init(input_card, cuckoo_ratio, evict_bound, origin, hash_attr, memstat, timing);
}

template<typename KType, typename VType, typename CntType>
CuckooHashTableRadix<KType,VType,CntType>::
~CuckooHashTableRadix() {
    CUDA_FREE(_keys, _memstat);
    CUDA_FREE(_values, _memstat);
    CUDA_FREE(_hash_func_configs, _memstat);
}

template<typename KType, typename VType, typename CntType>
int CuckooHashTableRadix<KType,VType,CntType>::
insert_ki(KType *keys, CntType n) {
    Timer t;
    /* Phase 1: Distribute keys into buckets */
    KType *keys_after_part = nullptr;       //keys after partitioning
    VType *vals_after_part = nullptr;       //values after partitioning
    CntType *buc_ptrs = nullptr;            //bucket ptrs
    CUDA_MALLOC(&buc_ptrs, sizeof(CntType)*(_num_buckets+1), _memstat);

    CUDA_MALLOC(&keys_after_part, sizeof(KType)*n, _memstat);
    CUDA_MALLOC(&vals_after_part, sizeof(VType)*n, _memstat);

    t.reset();

    /* Partitioning */
    RadixPartitioner<KType,VType,CntType> rp(n, _num_buckets, _memstat, _timing);
    rp.splitKI(keys, keys_after_part, vals_after_part, buc_ptrs);
    log_info("partition finished");

    /* Phase 2: Local cuckoo hashing */
    int *rehash_requests = nullptr;
    CUDA_MALLOC(&rehash_requests, sizeof(int), _memstat);

    // Loops until no rehashing needed.
    int rehash_count = 0;
    do {
        checkCudaErrors(cudaMemset(rehash_requests, 0, sizeof(int)));
        int grid_size = _num_buckets;   //each block processes a bucket
        int block_size = BUCKET_SIZE;   //each bucket is at most BUCKET_SIZE items
        log_info("Insert kernel, grid size: %llu, block size: %llu", grid_size, block_size);
        execKernelDynamicAllocation(cuckoo_insert_kernel, grid_size, block_size, BUCKET_SIZE * (sizeof(KType)+ sizeof(VType)), _timing, false, keys_after_part, vals_after_part, _keys, _values, _hash_func_configs, _num_funcs, buc_ptrs, _evict_bound, _pos_width, rehash_requests);
        if ((*rehash_requests) == 0)
            break;
        else {
            rehash_count++;
            log_info("Rehash");
            gen_hash_funcs();

            std::cout << "Funcs: ";
            for (int i = 0; i < _num_funcs; ++i) {
                FuncConfig fc = _hash_func_configs[i];
                std::cout << "(" << fc.rv << ", " << fc.ss << ") ";
            }
            std::cout << std::endl;
        }
    } while (rehash_count < MAX_DEPTH);
    if (rehash_count >= MAX_DEPTH) {
        log_error("Too many rehash, exit");
    }
    log_info("Inside cuckoo hashing: %.1f ms", t.elapsed()*1000);
    CUDA_FREE(keys_after_part, _memstat);
    CUDA_FREE(vals_after_part, _memstat);
    CUDA_FREE(buc_ptrs, _memstat);
    CUDA_FREE(rehash_requests, _memstat);

    return (rehash_count < MAX_DEPTH) ? rehash_count : ERR_DEPTH;
}

template<typename KType, typename VType, typename CntType>
int CuckooHashTableRadix<KType,VType,CntType>::
insert_kvs(KType *keys, VType *values, CntType n) {
    Timer t;
    /* Phase 1: Distribute keys into buckets */
    KType *keys_after_part = nullptr;       //keys after partitioning
    VType *vals_after_part = nullptr;       //values after partitioning
    CntType *buc_ptrs = nullptr;            //bucket ptrs
    CUDA_MALLOC(&buc_ptrs, sizeof(CntType)*(_num_buckets+1), _memstat);

    CUDA_MALLOC(&keys_after_part, sizeof(KType)*n, _memstat);
    CUDA_MALLOC(&vals_after_part, sizeof(VType)*n, _memstat);

    t.reset();
    if (_num_buckets > 1) {
        /* Partitioning */
        RadixPartitioner<KType,VType,CntType> rp(n, _num_buckets, _memstat, _timing);
        rp.splitKV(keys, keys_after_part, values, vals_after_part, buc_ptrs);
        log_info("partition finished");
    }
    else { //only a single bucket, no partitioning
        buc_ptrs[0] = 0;
        buc_ptrs[1] = n;
        keys_after_part = keys;
        vals_after_part = values;
        log_info("No partitioning");
    }

    /* Phase 2: Local cuckoo hashing */
    int *rehash_requests = nullptr;
    CUDA_MALLOC(&rehash_requests, sizeof(int), _memstat);

    // Loops until no rehashing needed.
    int rehash_count = 0;
    do {
        checkCudaErrors(cudaMemset(rehash_requests, 0, sizeof(int)));
        int grid_size = _num_buckets;   //each block processes a bucket
        int block_size = BUCKET_SIZE;   //each bucket is at most BUCKET_SIZE items
        log_info("Insert kernel, grid size: %llu, block size: %llu", grid_size, block_size);
        execKernelDynamicAllocation(cuckoo_insert_kernel, grid_size, block_size, BUCKET_SIZE * (sizeof(KType)+ sizeof(VType)), _timing, false, keys_after_part, vals_after_part, _keys, _values, _hash_func_configs, _num_funcs, buc_ptrs, _evict_bound, _pos_width, rehash_requests);
        if ((*rehash_requests) == 0)
            break;
        else {
            rehash_count++;
            log_info("Rehash");
            gen_hash_funcs();

            std::cout << "Funcs: ";
            for (int i = 0; i < _num_funcs; ++i) {
                FuncConfig fc = _hash_func_configs[i];
                std::cout << "(" << fc.rv << ", " << fc.ss << ") ";
            }
            std::cout << std::endl;
        }
    } while (rehash_count < MAX_DEPTH);
    if (rehash_count >= MAX_DEPTH) {
        log_error("Too many rehash, exit");
    }
    log_info("Inside cuckoo hashing: %.1f ms", t.elapsed()*1000);
    CUDA_FREE(keys_after_part, _memstat);
    CUDA_FREE(vals_after_part, _memstat);
    CUDA_FREE(buc_ptrs, _memstat);
    CUDA_FREE(rehash_requests, _memstat);

//    return (rehash_count < MAX_DEPTH) ? rehash_count : ERR_DEPTH;
    return rehash_count;
}

template<typename KType, typename VType, typename CntType>
void CuckooHashTableRadix<KType,VType,CntType>::
lookup_vals(KType *keys, VType *values, CntType n) {
    execKernel(cuckoo_lookup_kernel, ceil((double) n / BUCKET_SIZE), BUCKET_SIZE, _timing, false,
               keys, values, n, _keys, _values, _hash_func_configs, _num_funcs, _num_buckets, _pos_width);
};

/** Cuckoo: print content out. */
template<typename KType, typename VType, typename CntType>
void CuckooHashTableRadix<KType,VType,CntType>::
show_content() {
    std::cout << "Buckets: " << _num_buckets << std::endl;
    std::cout << "Funcs: ";
    for (int i = 0; i < _num_funcs; ++i) {
        FuncConfig fc = _hash_func_configs[i];
        std::cout << "(" << fc.rv << ", " << fc.ss << ") ";
    }
    std::cout << std::endl;
    for (int j = 0; j < _cuckoo_cardinality; ++j) {
        if (fetch_val(_keys[j]) != EMPTY_CELL) {
            std::cout << "(" << fetch_val(_keys[j]) <<" "<< _values[j] <<" "<<j<<"),";
        }
    }
    std::cout << std::endl;
}