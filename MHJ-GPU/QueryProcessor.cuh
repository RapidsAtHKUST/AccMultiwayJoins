//
// Created by Bryan on 18/4/2020.
//

#pragma once

#include "file_io.h"
#include "Relation.cuh"
#include "helper.h"
#include "Indexing/radix_partitioning.cuh"
#include "types.h"

using namespace std;

template<typename DataType, typename CntType>
class QueryProcessor {
public:
    /*
     * Load a single relation file (with two columns, for graph data)
     * */
    void load_single_rel(string full_file_name, Relation<DataType, CntType> *&relation, CUDAMemStat *memstat) {
        CUDA_MALLOC(&relation, sizeof(Relation<DataType,CntType>), memstat);
        int num_columns = 2;
        vector<string> properties;
        auto file_name = get_file_name_from_addr(full_file_name);
        split_string(file_name, properties, "_");
        auto file_rows = stoull(properties[0]);
        assert(file_rows > 0);

        auto data_from_file = read_rel_cols_mmap<DataType,CntType>(
                full_file_name.c_str(), num_columns, file_rows);
        relation[0].init(num_columns, file_rows, memstat);
        for(auto a = 0; a < num_columns; a++) {
            relation[0].data[a] = data_from_file[a];
        }
    }

    /*
     * Load all the relation files in the directory
     * */
    void load_multi_rels(
            string dir_name, Relation<DataType, CntType> *&relations,
            uint32_t &num_tables, CUDAMemStat *memstat) {
        auto file_list = list_files(dir_name.c_str());
        if (0 == file_list.size()) {
            log_error("The directory is empty");
            exit(1);
        }
        num_tables = (uint32_t)file_list.size();

        /*read the relations from files*/
        CUDA_MALLOC(&relations, sizeof(Relation<DataType,CntType>)*num_tables, memstat);
        for(auto i = 0; i < num_tables; i++) {
            string cur_file_name = file_list[i].substr(0,file_list[i].length()-rel_prefix.size());
            vector<string> properties;
            split_string(cur_file_name, properties, "_");
            auto data_from_file = read_rel_cols_mmap<DataType,CntType>(
                    (dir_name+"/"+file_list[i]).c_str(),
                    (int)properties.size() - 1, stoull(properties[0]));

            int cur_idx;
            log_no_newline("Enter the table order: ");
            cin>>cur_idx; //input table order
//            cur_idx = i;
            assert(cur_idx < num_tables);

            relations[cur_idx].init((uint32_t)properties.size() - 1, stoull(properties[0]), memstat);
            for(auto a = 0; a < relations[cur_idx].num_attrs; a++) {
                relations[cur_idx].attr_list[a] = (AttrType)stoi(properties[a+1]);
                relations[cur_idx].data[a] = data_from_file[a];
            }
        }
    }

    /*build hash tables, todo: each relation is hashed on the first attr by default*/
    void build_hash_multiway(
            Relation<DataType, CntType> *build_tables, HashTable<DataType, CntType> *&hash_tables,
            CntType *&bucket_vec, uint32_t num_hash_tables, uint32_t buc_ratio,
            CUDAMemStat *memstat, CUDATimeStat *timing) {
        Timer t;
        bsize_t ht_total_size = 0;
        CUDA_MALLOC(&hash_tables, sizeof(HashTable<DataType,CntType>)*num_hash_tables, memstat);
        CUDA_MALLOC(&bucket_vec, sizeof(CntType)*num_hash_tables, memstat);

        for(auto i = 0; i < num_hash_tables; i++) {
            uint32_t cur_buckets = floorPowerOf2(build_tables[i].length / buc_ratio);
            log_info("Buckets for hash table %d: %llu", i, cur_buckets);
            t.reset();
            CUDA_MALLOC(&hash_tables[i].hash_keys, sizeof(DataType)*build_tables[i].length, memstat);
            CUDA_MALLOC(&hash_tables[i].idx_in_origin, sizeof(CntType)*build_tables[i].length, memstat);
            CUDA_MALLOC(&hash_tables[i].buc_ptrs, sizeof(CntType)*(cur_buckets+1), memstat);
            CUDA_MALLOC(&hash_tables[i].attr_list, sizeof(AttrType)*build_tables[i].num_attrs, memstat);
            hash_tables[i].hash_attr = build_tables[i].attr_list[0];
            hash_tables[i].length = build_tables[i].length;
            hash_tables[i].buckets = cur_buckets;
            hash_tables[i].num_attrs = build_tables[i].num_attrs;
            hash_tables[i].data = build_tables[i].data;
            checkCudaErrors(cudaMemcpy(hash_tables[i].attr_list, build_tables[i].attr_list, sizeof(AttrType)*hash_tables[i].num_attrs, cudaMemcpyDeviceToDevice));

            bucket_vec[i] = cur_buckets;
            RadixPartitioner<DataType,CntType,CntType> rp(hash_tables[i].length, hash_tables[i].buckets,
                                                          memstat, timing);
            auto gpu_time_idx = timing->get_idx();
            rp.splitKI(build_tables[i].data[0], hash_tables[i].hash_keys,
                       hash_tables[i].idx_in_origin, hash_tables[i].buc_ptrs);
            log_info("Build hash table %d: GPU time: %.2f ms, CPU time: %.2f ms", i, timing->diff_time(gpu_time_idx), t.elapsed()*1000);
            ht_total_size += hash_tables[i].get_ht_size();
        }
        log_info("Total hash table size: %.1f MB", 1.0*ht_total_size/1024/1024);
    }

    void build_hash_single(
            Relation<DataType, CntType> &build_table, HashTable<DataType, CntType> *&hash_tables,
            CntType *&bucket_vec, uint32_t num_hash_tables, uint32_t buc_ratio, vector<pair<AttrType,AttrType>> input_attrs,
            CUDAMemStat *memstat, CUDATimeStat *timing) {
        Timer t;
        bsize_t ht_total_size = 0;
        CUDA_MALLOC(&hash_tables, sizeof(HashTable<DataType,CntType>)*num_hash_tables, memstat);
        CUDA_MALLOC(&bucket_vec, sizeof(CntType)*num_hash_tables, memstat);

        uint32_t cur_buckets = floorPowerOf2(build_table.length / buc_ratio);
        log_info("Buckets for hash tables: %llu", cur_buckets);

        /*set attr_list for the probe (build) table*/
        build_table.attr_list[0] = input_attrs[0].first;
        build_table.attr_list[1] = input_attrs[0].second;

        /*init for hash_tables[0]*/
        t.reset();
        CUDA_MALLOC(&hash_tables[0].hash_keys, sizeof(DataType)*build_table.length, memstat);
        CUDA_MALLOC(&hash_tables[0].idx_in_origin, sizeof(CntType)*build_table.length, memstat);
        CUDA_MALLOC(&hash_tables[0].buc_ptrs, sizeof(CntType)*(cur_buckets+1), memstat);
        CUDA_MALLOC(&hash_tables[0].attr_list, sizeof(AttrType)*build_table.num_attrs, memstat);
        hash_tables[0].length = build_table.length;
        hash_tables[0].buckets = cur_buckets;
        hash_tables[0].num_attrs = build_table.num_attrs;
        hash_tables[0].data = build_table.data;
        bucket_vec[0] = cur_buckets;

        hash_tables[0].attr_list[0] = input_attrs[1].first;
        hash_tables[0].attr_list[1] = input_attrs[1].second;
        hash_tables[0].hash_attr = hash_tables[0].attr_list[0];

        RadixPartitioner<DataType,CntType,CntType> rp(hash_tables[0].length, hash_tables[0].buckets,
                                                      memstat, timing);
        auto gpu_time_idx = timing->get_idx();
        rp.splitKI(build_table.data[0], hash_tables[0].hash_keys,
                   hash_tables[0].idx_in_origin, hash_tables[0].buc_ptrs);
        log_info("Build hash table: GPU time: %.2f ms, CPU time: %.2f ms", timing->diff_time(gpu_time_idx), t.elapsed()*1000);
        ht_total_size += hash_tables[0].get_ht_size();

        /*copy to other hash_tables*/
        for(auto i = 1; i < num_hash_tables; i++) {
            CUDA_MALLOC(&hash_tables[i].attr_list, sizeof(AttrType)*hash_tables[0].num_attrs, memstat);
            hash_tables[i].data = hash_tables[0].data;
            hash_tables[i].hash_keys = hash_tables[0].hash_keys;
            hash_tables[i].idx_in_origin = hash_tables[0].idx_in_origin;
            hash_tables[i].buc_ptrs = hash_tables[0].buc_ptrs;
            hash_tables[i].length = hash_tables[0].length;
            hash_tables[i].buckets = hash_tables[0].buckets;
            hash_tables[i].num_attrs = hash_tables[0].num_attrs;
            bucket_vec[i] = bucket_vec[0];

            hash_tables[i].attr_list[0] = input_attrs[i+1].first;
            hash_tables[i].attr_list[1] = input_attrs[i+1].second;
            hash_tables[i].hash_attr = hash_tables[i].attr_list[0];
            log_info("Copy properties of hash table 0 to hash table %d", i);
        }
    }
};
