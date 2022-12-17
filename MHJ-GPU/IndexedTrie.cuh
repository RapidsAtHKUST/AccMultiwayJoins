//
// Created by Bryan on 4/2/2020.
//

#pragma once

#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#include "cuda/CUDAStat.cuh"
#include "pretty_print.h"
#include "timer.h"
#include "types.h"
using namespace std;

template<typename DataType, typename CntType>
struct IndexedTrie {
    uint32_t num_attrs;     //number of attributes
    DataType **data;        //[0:num_attrs][0:data_len[i]], i = 0,1,...,num_attrs, Trie data
    CntType *data_len;      //[0:num_attrs], number of values in each level of data
    AttrType *attr_list;    //[0:num_attrs], the attribute index of each level
    CntType **offsets;      //[0:num_attrs-1][0:data_len[i]+1], i=0,1,..., offsets for the next level
    CUDAMemStat *memstat;

    void init(uint32_t num_attrs, CUDAMemStat *memstat, cudaStream_t stream=0) {
        this->num_attrs = num_attrs;
        this->memstat = memstat;
        CUDA_MALLOC(&data, sizeof(DataType*)*num_attrs, memstat, stream);
        if (num_attrs > 1) {
            CUDA_MALLOC(&offsets, sizeof(CntType*)*(num_attrs-1), memstat, stream);
        }
        else {
            offsets = nullptr;
        }
        CUDA_MALLOC(&data_len, sizeof(CntType)*num_attrs, memstat, stream);
        CUDA_MALLOC(&attr_list, sizeof(uint32_t)*num_attrs, memstat, stream);
    }

    void clear(bool delete_data=true) { //currently not support deconstruction function
        if (delete_data) { //delete the data they point to
            for(auto i = 0; i < num_attrs; i++) CUDA_FREE(data[i], memstat);
            for(auto i = 0; i < num_attrs-1; i++) CUDA_FREE(offsets[i], memstat);
        }
        //free buc_ptrs for hash-based method
        CUDA_FREE(data, memstat);
        if (this->num_attrs > 1) {
            CUDA_FREE(offsets, memstat);
        }
        CUDA_FREE(data_len, memstat);
        CUDA_FREE(attr_list, memstat);
    }

    void print() { //print the Trie
        for(auto a = 0; a < num_attrs; a++) {
            cout<<"("<<attr_list[a]<<")"<<"val: "
                <<pretty_print_array(data[a], data_len[a])
                <<" len:"<<data_len[a]<<endl;
            if (a < num_attrs-1)
                cout<<"("<<attr_list[a]<<")"<<"off: "
                    <<pretty_print_array(offsets[a], data_len[a]+1)<<endl;
        }
    }
};