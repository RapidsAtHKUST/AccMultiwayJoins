//
// Created by Bryan on 25/3/2020.
//
#pragma once

#include "types.h"

template <typename DataType, typename CntType>
struct Relation {
    uint32_t num_attrs;     //number of attrs
    CntType length;         //table cardinality
    DataType **data;        //[0:num_attrs)[0:cardinality), data of the rel, column store
    AttrType *attr_list;    //[0:num_attrs), the attribute list
    CUDAMemStat *memstat;

    void init(uint32_t num_attrs, CntType length, CUDAMemStat *memstat) {
        this->num_attrs = num_attrs;
        this->length = length;
        this->memstat = memstat;
        CUDA_MALLOC(&data, sizeof(DataType*)*num_attrs, memstat);
        CUDA_MALLOC(&attr_list, sizeof(AttrType)*num_attrs, memstat);
    }
    void destroy(){
        CUDA_FREE(data, memstat);
        CUDA_FREE(attr_list, memstat);
    }
    bsize_t get_rel_size() {
        bsize_t total_size = 0;
        total_size += sizeof(DataType)*length*num_attrs; //the data
        total_size += sizeof(AttrType)*num_attrs; //ths attr_list
        total_size += sizeof(CntType); //the length field
        total_size += sizeof(uint32_t); //the num_attr field
        return total_size;
    }
    void print() {
        for(auto i = 0; i < length; i++) {
            std::cout<<"item "<<i<<": (";
            for(auto j = 0; j < num_attrs; j++) {
                std::cout<<data[j][i];
                if (j != num_attrs-1) std::cout<<",";
                else std::cout<<")";
            }
            std::cout<<std::endl;
        }
    }

};

template <typename DataType, typename CntType>
struct HashTable {
    DataType *hash_keys;        //[0,length), hash table
    CntType *idx_in_origin;     //[0,length), row indexes in the original table

    CntType length;             //length of the hash table
    CntType capacity;           //capacity of the hash table, used in open addressing

    AttrType *attr_list;        //attr_list of the original table
    uint32_t num_attrs;         //number of attrs
    AttrType hash_attr;         //build hash index on this attr of the original table
    DataType **data;            //the data of the original table

    CntType *buc_ptrs;          //bucket pointers, (buckets+1) pointers, used in bucket hashing
    uint32_t buckets;           //number of buckets, used in bucket hashing

    bsize_t get_ht_size() {
        bsize_t total_size = 0;
        total_size += sizeof(CntType); //the length field
        total_size += sizeof(DataType)*length;// the hash keys
        total_size += sizeof(AttrType); //the hash_attr field
        total_size += sizeof(AttrType)*num_attrs; //the attr_list field
        total_size += sizeof(uint32_t); //the num_attrs field
        total_size += sizeof(Relation<DataType,CntType>*); //the origin field
        total_size += sizeof(CntType)*length;// the idx_in_origin
        total_size += sizeof(uint32_t); //the bucket field
        total_size += sizeof(CntType)*(buckets+1);// the buc_ptrs
        return total_size;
    }
};