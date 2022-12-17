#pragma once

#ifdef __JETBRAINS_IDE__
#include "cuda/cuda_fake/fake.h"
#include "openmp_fake.h"
#endif

#ifndef DEVICE_ID
#define DEVICE_ID (0)
#endif

#include <iostream>
#include <string>
#include "cuda/CUDAStat.cuh"
#include "conf.h"

#define INVALID_REL_VAL                 (99)
#define EMPTY_HASH_SLOT (0xffffffff)
#define INVALID_BUC_START   (INT32_MAX)

#define INVALID_FUNC_IDX        (-1)

using KeyType = int;
using CarType = unsigned long long;
using bsize_t = unsigned long long;
using AttrType = uint32_t;
using RelType = char;

enum AlgoType {
    TYPE_MHJ, TYPE_AMHJ,
    TYPE_AMHJ_DRO,
    TYPE_AMHJ_WS,
    TYPE_AMHJ_WS_DRO,
    TYPE_AMHJ_FIB,
    TYPE_AMHJ_OPEN_ADDR,
    TYPE_MHJ_OPEN_ADDR,
    TYPE_AMHJ_DRO_OPEN_ADDR
};

std::string rel_prefix = ".db";

/* Cuckoo hashing structures*/
/** Reserved value for indicating "empty". */
#define EMPTY_CELL (0xffffffff)
#define INVALID_KEY (INT32_MAX) //occupy the slot when a key is fetched but the value is not

template<typename KType, typename VType, typename CntType>
struct CuckooHashTable {
    CntType _cuckoo_cardinality;    //cardinality of A SINGLE CuckooHashTable
    int _evict_bound; //upper bound of eviction before rehash
    int _num_funcs;   //number of hash functions
    int _pos_width;   //offset for storing the hash function id in each key

    /* Actual key-value data*/
    KType *_keys;
    VType *_values;

    CUDAMemStat *_memstat;
    CUDATimeStat *_timing;

    CuckooHashTable() {}
    CuckooHashTable(const CntType cuckoo_card, const int evict_bound,
                    const int num_funcs, CUDAMemStat *memstat, CUDATimeStat *timing)
            : _cuckoo_cardinality(cuckoo_card), _num_funcs(num_funcs),
              _pos_width(ceil(log2((double) _num_funcs))), _memstat(memstat), _timing(timing) {}
};