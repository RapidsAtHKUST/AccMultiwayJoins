//
// Created by Bryan on 4/2/2020.
//

#pragma once

#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#include "../common-utils/cuda/CUDAStat.cuh"
#include "types.h"
#include "../common-utils/pretty_print.h"
#include "timer.h"
using namespace std;

#define INVALID_ATTR    (9999)

template<typename DataType, typename CntType>
struct IndexedTrie {
    uint32_t num_attrs;     //number of attributes
    CntType *data_len;      //[0:num_attrs), number of values in each level of data
    uint32_t *attr_list;    //[0:num_attrs), the attribute index of each level
    CntType *trie_offsets; //[0:num_attrs-1], offsets of the attribute arrays in the global Trie. offsets[a] should be deducted trie_offsets[a+1] to derive correct number

    DataType **data;        //[0:num_attrs)[0:data_len[i]), i = 0,1,...,num_attrs, Trie data
    CntType **offsets;      //[0:num_attrs-1)[0:data_len[i]+1), i=0,1,..., offsets for the next level, for sort-based LFTJ

    CUDAMemStat *memstat;
    bool validity;          //whether the Trie is a valid Trie

    void init(uint32_t num_attrs, CUDAMemStat *memstat, cudaStream_t stream=0) {
        this->num_attrs = num_attrs;
        this->memstat = memstat;
        CUDA_MALLOC(&data, sizeof(DataType*)*num_attrs, memstat, stream);
        if (num_attrs > 1) {
            CUDA_MALLOC(&offsets, sizeof(CntType *) * (num_attrs - 1), memstat, stream);
        }
        CUDA_MALLOC(&data_len, sizeof(CntType)*num_attrs, memstat, stream);
        CUDA_MALLOC(&attr_list, sizeof(uint32_t)*num_attrs, memstat, stream);
        CUDA_MALLOC(&trie_offsets, sizeof(CntType)*num_attrs, memstat, stream);
        cudaMemsetAsync(trie_offsets, 0, sizeof(CntType)*num_attrs, stream);
        validity = true;

        for(auto i = 0; i < num_attrs; i++) {
            data[i] = nullptr;
            data_len[i] = 0;
            attr_list[i] = INVALID_ATTR;
        }
        for(auto i = 0; i < num_attrs-1; i++) {
            offsets[i] = nullptr;
        }
    }

    void clear(bool delete_data=true) { //currently not support deconstruction function
        if (delete_data) { //delete the data they point to
            for(auto i = 0; i < num_attrs; i++) CUDA_FREE(data[i], memstat);
            for(auto i = 0; i < num_attrs-1; i++) CUDA_FREE(offsets[i], memstat);
        }
        //free buc_ptrs for hash-based method
        CUDA_FREE(data, memstat);
        CUDA_FREE(offsets, memstat);
        CUDA_FREE(data_len, memstat);
        CUDA_FREE(attr_list, memstat);
        CUDA_FREE(trie_offsets, memstat);
        validity = false;
    }

    bool check() { //check whether all the Trie structures are ready
        for(auto i = 0; i < num_attrs; i++) {
            if(data[i] == nullptr) {
                log_error("Trie data not set");
                return false;
            }
            if (data_len[i] == 0)  {
                log_error("Trie data_len not set");
                return false;
            }
            if (attr_list[i] == INVALID_ATTR) {
                log_error("Trie attr_list not set");
                return false;
            }
        }
        for(auto i = 0; i < num_attrs-1; i++) {
            if (offsets[i] == nullptr) {
                log_error("Trie offset not set");
                return false;
            }
        }
        return true;
    }

    bsize_t get_disk_size() { //return the size for writting to disk
        auto main_size = get_size_from_1st(); //data and offset size
        main_size += sizeof(uint32_t); //num_attrs
        main_size += sizeof(CntType)*num_attrs; //data_len
        main_size += sizeof(uint32_t)*num_attrs; //attr_list
        return main_size;
    }

    /*get the data size(in bytes) of the trie from the first level
     *no parameter means from 0 to data_len[0]*/
    bsize_t get_size_from_1st(CntType head=0, CntType tail=UINT32_MAX) {
        bsize_t res = 0;
        CntType moving_head = head;
        CntType moving_tail = (tail == UINT32_MAX) ? data_len[0]:tail;

        assert(moving_tail <= data_len[0]);
        for(auto l = 0; l < num_attrs; l++) { //add up all the attrs except the last one
            res += sizeof(DataType)*(moving_tail-moving_head); //val l
            if (l < num_attrs-1) {
                res += sizeof(CntType)*(moving_tail-moving_head+1); // off l
                moving_head = offsets[l][moving_head]; //update head and tail
                moving_tail = offsets[l][moving_tail];
            }
        }
        return res;
    }

    /*serialize Trie data to disk*/
    void serialization(const char *file_name) {
        Timer serialization_time;
        auto fd = open(file_name, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        auto total_size = this->get_disk_size();
        log_info("total serialization size: %llu bytes", total_size);
        auto ret = ftruncate(fd, total_size);
        auto *buf = (char*)mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

        if (buf == MAP_FAILED) {
            log_error("mmap (for write) gets wrong.");
            exit(1);
        }
        assert(ret == 0);

        /*writing data*/
        bsize_t off = 0;
        memcpy(buf+off, &num_attrs, sizeof(uint32_t)); //write num_attr
        off += sizeof(uint32_t);
        memcpy(buf+off, data_len, sizeof(CntType)*num_attrs); //write data_len
        off += sizeof(CntType)*num_attrs;
        memcpy(buf+off, attr_list, sizeof(uint32_t)*num_attrs); //write attr_list
        off += sizeof(uint32_t)*num_attrs;
        for(auto i = 0; i < num_attrs; i++) { //write data
            memcpy(buf+off, data[i], sizeof(DataType)*data_len[i]);
            off += sizeof(DataType)*data_len[i];
        }
        for(auto i = 0; i < num_attrs-1; i++) { //write offset
            memcpy(buf+off, offsets[i], sizeof(CntType)*(data_len[i]+1));
            off += sizeof(CntType)*(data_len[i]+1);
        }
        assert(off == total_size);
        close(fd);
        munmap(buf, total_size);
        log_info("Serialization time: %.2f s.", serialization_time.elapsed());
    }

    /*deserialize disk data and recover the Trie*/
    void init_with_deserialization(const char *file_name, CUDAMemStat *memstat) {
        Timer read_time;
        auto fd = open(file_name, O_RDONLY, S_IRUSR | S_IWUSR);
        if (fd == -1) {
            log_error("Faile to open file: %s", file_name);
            exit(1);
        }
        size_t total_size = lseek64(fd,0,SEEK_END);
        log_info("total deserialization size: %llu bytes", total_size);

        auto *buf = (char*)mmap(NULL, total_size, PROT_READ, MAP_SHARED, fd, 0);
        if (buf == MAP_FAILED) {
            log_error("mmap (for read) gets wrong.");
            exit(1);
        }
        bsize_t off = 0;
        memcpy(&num_attrs, buf+off, sizeof(uint32_t)); //read num_attrs
        off += sizeof(uint32_t);
        init(num_attrs, memstat); //allocate memory

        /*recovering data*/
        memcpy(data_len, buf+off, sizeof(CntType)*num_attrs); //read data_len
        off += sizeof(CntType)*num_attrs;
        memcpy(attr_list, buf+off, sizeof(uint32_t)*num_attrs); //read attr_list
        off += sizeof(uint32_t)*num_attrs;
        for(auto i = 0; i < num_attrs; i++) { //read data
            CUDA_MALLOC_NO_PREFETCH(&data[i], sizeof(DataType)*data_len[i], memstat); //do not prefetch to GPUs
            memcpy(data[i], buf+off, sizeof(DataType)*data_len[i]);
            off += sizeof(DataType)*data_len[i];
        }
        for(auto i = 0; i < num_attrs-1; i++) { //read offset
            CUDA_MALLOC_NO_PREFETCH(&offsets[i], sizeof(CntType)*(data_len[i]+1), memstat); //do not prefetch to GPUs
            memcpy(offsets[i], buf+off, sizeof(CntType)*(data_len[i]+1));
            off += sizeof(CntType)*(data_len[i]+1);
        }
        close(fd);
        munmap(buf, total_size);
        log_info("Deserialization time: %.2f s.", read_time.elapsed());
    }

    void print() { //print the Trie
        for(auto a = 0; a < num_attrs; a++) {
            cout<<"("<<attr_list[a]<<")"<<"val: "
                <<pretty_print_array(data[a], data_len[a])
                <<" tri_off:"<<trie_offsets[a]<<' '<<"len:"<<data_len[a]<<endl;
            if (a < num_attrs-1)
                cout<<"("<<attr_list[a]<<")"<<"off: "
                    <<pretty_print_array(offsets[a], data_len[a]+1)<<endl;
        }
    }
};

#undef INVALID_ATTR