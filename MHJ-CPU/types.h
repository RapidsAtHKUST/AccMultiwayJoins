#pragma once

#ifdef __JETBRAINS_IDE__
#include "openmp_fake.h"
#include "pthread_fake.h"
#endif

#include <string>
#include <cstdint>

#define MAX_NUM_BUILD_TABLES     (7)     //maximum number of build tables
#define MAX_NUM_RES_ATTRS        (8)     //maximum number of attrs in results
#define BUC_RATIO                (1)     //ratio between build table cardinality and hash buckets
#define RES_PER_CHUNK            (65536)    //number of result tuples per chunk
#define CACHE_LINE_SIZE          (64)    //cache line size

using KeyType = int;
using CarType = unsigned long long;
using bsize_t = unsigned long long;
using AttrType = uint32_t;
using RelType = char;

std::string rel_prefix = ".db";

template <typename DataType, typename CntType>
struct hash_tuple_t {
    DataType hash_key;
    CntType idx_in_origin;
};

template <typename DataType, typename CntType>
struct Relation {
    uint32_t num_attrs;     //number of attrs
    CntType length;         //table cardinality
    DataType **data;        //[0:num_attrs)[0:cardinality), data of the rel, column store
    AttrType *attr_list;    //[0:num_attrs), the attribute list

    void init(uint32_t num_attrs, CntType length) {
        this->num_attrs = num_attrs;
        this->length = length;
        this->data = (DataType**)malloc(sizeof(DataType*)*num_attrs);
        this->attr_list = (AttrType*)malloc(sizeof(AttrType)*num_attrs);
    }
    void destroy(){
        free(this->data);
        free(this->attr_list);
    }
};

template <typename DataType, typename CntType>
struct HashTable {
    hash_tuple_t<DataType,CntType> *hash_tuples;

    CntType length;             //length of the hash table
    AttrType hash_attr;         //build hash index on this attr of the original table
    Relation<DataType,CntType> *origin; //pointer to the original Relation
    uint32_t buckets;           //number of buckets
    CntType *buc_ptrs;              //bucket pointers, (buckets+1) pointers

    void init(CntType length, uint32_t buckets) {
        this->length = length;
        this->buckets = buckets;
        hash_tuples = (hash_tuple_t<DataType,CntType>*)malloc(sizeof(hash_tuple_t<DataType,CntType>)*length);
        buc_ptrs = (CntType*)malloc(sizeof(CntType)*(buckets+1));
    }
};

/*chunk of result tables*/
template <typename DataType, typename CntType>
struct ResChunk {
    DataType **data;
    CntType num_res;
    uint32_t num_res_attrs;
    ResChunk<DataType,CntType> *next;
    ResChunk(uint32_t num_res_attrs) {
        this->num_res_attrs = num_res_attrs;
        data = (DataType**)malloc(sizeof(DataType*)*num_res_attrs);
        num_res = 0;
        next = nullptr;
        for(auto i = 0; i < num_res_attrs; i++) {
            data[i] = (DataType*)malloc(sizeof(DataType)*RES_PER_CHUNK);
        }
    }
};


