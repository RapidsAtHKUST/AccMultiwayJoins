//
// Created by Bryan on 27/3/2020.
//
#pragma once

#ifdef __JETBRAINS_IDE__
#include "../../common-utils/cuda/cuda_fake/fake.h"
#endif

#include <iostream>
#include "cuda/CUDAStat.cuh"
#include "Relation.cuh"
#include "Joins/AMHJ.cuh"
#include "Joins/AMHJ_WS.cuh"
#include "Joins/AMHJ_DRO.cuh"
#include "Joins/AMHJ_WS_FIB.cuh"
#include "Joins/AMHJ_WS_DRO.cuh"
#include "Joins/MHJ.cuh"
#include "Joins/MHJ_open_addr.cuh"
#include "Joins/AMHJ_open_addr.cuh"
#include "Joins/AMHJ_DRO_open_addr.cuh"
#include "HashJoinEPFL/hashJoinEPFL.cuh"

using bsize_t = unsigned long long;

template<typename DataType, typename OffsetType>
__global__ void set_offset(DataType *&output, DataType *input, OffsetType offset) {
    output = input + offset;
}

template<typename DataType, typename CntType, bool single_join_key, AlgoType algo_type>
class OOC {
private:
    float input_budget_MB; //in MB
    float output_budget_MB; //in MB
    float total_budget_MB; //total memory budgets, in MB
public:
    OOC(float total_budget_MB): total_budget_MB(total_budget_MB) {
        input_budget_MB = 1024;
        output_budget_MB = 4096;
        /*check the budget setting*/
        if (output_budget_MB * 1024 * 1024 > MAX_PINNED_MEMORY_SIZE) {
            log_error("Output budget could not exceed pinned memory size");
            exit(1);
        }
        if ((input_budget_MB + output_budget_MB)*2*1024*1024 + OOC_RESERVE_SIZE > total_budget_MB*1024*1024) {
            log_error("Sum of input and output budget exceeds the total budget");
            exit(1);
        }
    }
    OOC(float total_budget_MB, float input_budget_MB, float output_budget_MB):
            total_budget_MB(total_budget_MB),
            input_budget_MB(input_budget_MB),
            output_budget_MB(output_budget_MB) {
        /*check the budget setting*/
        if (output_budget_MB * 1024 * 1024 > MAX_PINNED_MEMORY_SIZE) {
            log_error("Output budget could not exceed pinned memory size");
            exit(1);
        }
        if ((input_budget_MB + output_budget_MB)*2*1024*1024 + OOC_RESERVE_SIZE > total_budget_MB*1024*1024) {
            log_error("Sum of input and output budget exceeds the total budget");
            exit(1);
        }
    }
    void prefetch(Relation<DataType, CntType> input_rel, Relation<DataType, CntType> &output_rel,
                  CntType offset, CntType length, cudaStream_t stream) {
        output_rel.length = length;
        /*separate the set_offset and prefetch*/
        for(auto i = 0; i < output_rel.num_attrs; i++) { //set offset with GPU
            set_offset<<<1,1,0,stream>>>(output_rel.data[i], input_rel.data[i], offset);
        }
        for(auto i = 0; i < output_rel.num_attrs; i++) { //prefetch
            checkCudaErrors(cudaMemPrefetchAsync(input_rel.data[i]+offset, sizeof(DataType)*length, DEVICE_ID, stream));
        }
    }

    CntType execute(
            const Relation<DataType, CntType> probe_table,
            HashTable<DataType, CntType> *hash_tables, uint32_t num_hash_tables,
            bool **used_for_compare, CntType *bucketVec,
            int num_res_attrs, AttrType *attr_idxes_in_iRes, int num_attr_idxes_in_iRes, bool ooc,
            DataType **&res, CUDAMemStat *memstat, CUDATimeStat *timing) {
        Timer t;
        CntType res_cnt = 0;
        cudaStream_t streams[3];
        bsize_t input_budget = (bsize_t)(input_budget_MB*1024*1024);
        bsize_t output_budget = (bsize_t)(output_budget_MB*1024*1024);
        CntType max_items_in_output_buffer = output_budget / num_res_attrs / sizeof(DataType);
        log_info("input_budget: %llu bytes, output_budget: %llu bytes", input_budget, output_budget);
        log_info("Maximum num of outputs: %llu", max_items_in_output_buffer);

        if (!ooc) {
            log_info("ooc is disabled");
            switch (algo_type) {
                case TYPE_MHJ: {
                    MHJ<DataType,CntType,single_join_key> MHJ_instance(num_res_attrs, probe_table.length, memstat, timing);
                    res_cnt += MHJ_instance.MHJ_evaluate
                            (probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, attr_idxes_in_iRes,
                             num_attr_idxes_in_iRes, res, max_items_in_output_buffer);
                    break;
                }
                case TYPE_AMHJ: {
                    AMHJ<DataType,CntType,single_join_key> AMHJ_instance(num_res_attrs, probe_table.length, memstat, timing);
                    res_cnt += AMHJ_instance.AMHJ_evaluate
                            (probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, attr_idxes_in_iRes,
                             num_attr_idxes_in_iRes, res, max_items_in_output_buffer);
                    break;
                }
                case TYPE_AMHJ_DRO: {
                    AMHJ_DRO<DataType,CntType,single_join_key> AMHJ_DRO_instance
                            (num_hash_tables, num_res_attrs, probe_table.length, probe_table.length, memstat, timing);
                    res_cnt += AMHJ_DRO_instance.AMHJ_DRO_evaluate
                            (probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, attr_idxes_in_iRes,
                             num_attr_idxes_in_iRes, res, max_items_in_output_buffer);
                    break;
                }
                case TYPE_AMHJ_WS: {
                    AMHJ_WS<DataType,CntType,single_join_key,true> AMHJ_WS_instance(num_res_attrs, memstat, timing);
                    res_cnt += AMHJ_WS_instance.AMHJ_WS_evaluate
                            (probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec,
                             attr_idxes_in_iRes, num_attr_idxes_in_iRes, res, max_items_in_output_buffer);
                    break;
                }
                case TYPE_AMHJ_FIB: {
                    AMHJ_WS_FIB<DataType,CntType,single_join_key,true,true> AMHJ_WS_FIB_instance(num_res_attrs, memstat, timing);
                    res_cnt += AMHJ_WS_FIB_instance.AMHJ_WS_FIB_evaluate
                            (probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec,
                             attr_idxes_in_iRes, num_attr_idxes_in_iRes, res, max_items_in_output_buffer);
                    break;
                }
                case TYPE_MHJ_OPEN_ADDR: {
                    res_cnt += MHJ_open_addr<DataType, CntType, single_join_key>
                            (probe_table, hash_tables, num_hash_tables, used_for_compare, attr_idxes_in_iRes,
                             num_attr_idxes_in_iRes, res, num_res_attrs, max_items_in_output_buffer, memstat, timing);
                    break;
                }
                case TYPE_AMHJ_OPEN_ADDR: {
                    res_cnt += AMHJ_open_addr<DataType, CntType, single_join_key>
                            (probe_table, hash_tables, num_hash_tables, used_for_compare, attr_idxes_in_iRes,
                             num_attr_idxes_in_iRes, res, num_res_attrs, max_items_in_output_buffer, memstat, timing);
                    break;
                }
                case TYPE_AMHJ_DRO_OPEN_ADDR: {
                    res_cnt += AMHJ_DRO_open_addr<DataType, CntType, single_join_key>
                            (probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, attr_idxes_in_iRes,
                             num_attr_idxes_in_iRes, res, num_res_attrs, max_items_in_output_buffer, memstat, timing);
                    break;
                }
                case TYPE_AMHJ_WS_DRO: {
                    AMHJ_WS_DRO<DataType,CntType,single_join_key,true> AMHJ_WS_DRO_instance
                            (num_hash_tables, num_res_attrs, probe_table.length, probe_table.length, memstat, timing);
                    res_cnt += AMHJ_WS_DRO_instance.AMHJ_WS_DRO_evaluate
                            (probe_table, hash_tables, num_hash_tables, used_for_compare, bucketVec, attr_idxes_in_iRes,
                             num_attr_idxes_in_iRes, res, max_items_in_output_buffer);
                    break;
                }
                default: {
                    log_error("Unsupported algorithm");
                    exit(1);
                }
            }
            cudaDeviceSynchronize();
            log_info("Total number of outputs: %llu", res_cnt);
            return res_cnt;
        }
        log_info("ooc is enabled");

        /*compute the probe items processed per iteration*/
        CntType intervals[MAX_OOC_ITERATIONS];
        CntType probe_interval = (input_budget - 30) / probe_table.num_attrs / sizeof(DataType);
        auto num_iterations = (probe_table.length + probe_interval - 1) / probe_interval;
        assert(num_iterations <= MAX_OOC_ITERATIONS);

        for(auto i = 0; i < num_iterations-1; i++) {
            intervals[i] = probe_interval;
        }
        if (num_iterations > 1) {
            intervals[num_iterations-1] = probe_table.length % probe_interval;
            if (intervals[num_iterations-1] < 0.1 * probe_interval) { //adjust last two iterations
                intervals[num_iterations-1] = intervals[num_iterations-2] =
                        ((probe_table.length % probe_interval)+probe_interval)/2;
            }
        }
        auto min_probe_length = intervals[num_iterations-1];
        auto max_probe_length = intervals[0];
        log_info("Buffer size: input: %.1f GB, output: %.1f GB", 1.0 * input_budget/1024/1024/1024, 1.0*output_budget/1024/1024/1024);
        log_info("probe table cardinality: %llu", probe_table.length);
        log_info("Number of join iterations: %llu", num_iterations);
        log_info("Min probe length: %llu", min_probe_length);
        log_info("Max probe length: %llu", max_probe_length);

        Relation<DataType,CntType> *probe_table_slices = nullptr;   /*input double buffer*/
        DataType **res_ooc[2];                                      /*output double buffer*/
        CUDA_MALLOC(&probe_table_slices, sizeof(Relation<DataType,CntType>)*2, memstat);

        AMHJ_WS<DataType,CntType,single_join_key,true> AMHJ_WS_instance(num_res_attrs, memstat, timing);
        AMHJ<DataType,CntType,single_join_key> AMHJ_instance(num_res_attrs, max_probe_length, memstat, timing);
        MHJ<DataType,CntType,single_join_key> MHJ_instance(num_res_attrs, max_probe_length, memstat, timing);
        AMHJ_DRO<DataType,CntType,single_join_key> AMHJ_DRO_instance
                (num_hash_tables, num_res_attrs, min_probe_length, max_probe_length, memstat, timing); //SBP class

        int output_buffer_idx = 0;
        int input_buffer_idx = 0;
        CntType probe_offset = 0; //advance this offset val during iteration

        for(auto i = 0; i < 3; i++) { //streams[i]=0 if disabling pipelining
//            streams[i] = 0;
            cudaStreamCreate(&streams[i]);
        }
        for(auto i = 0; i < 2; i++) {
            /*init input buffers*/
            probe_table_slices[i].init(probe_table.num_attrs, 0, probe_table.memstat);
            checkCudaErrors(cudaMemcpy(probe_table_slices[i].attr_list, probe_table.attr_list,
                                       sizeof(AttrType)*probe_table.num_attrs, cudaMemcpyDeviceToDevice));

            /*init output buffers, using cudaMalloc*/
            CUDA_MALLOC(&res_ooc[i], sizeof(DataType*)*num_res_attrs, memstat);
            for(auto j = 0; j < num_res_attrs; j++) {
                cudaMalloc((void**)&res_ooc[i][j], sizeof(DataType)*max_items_in_output_buffer);
            }
        }

        /* allocate the output space on CPUs, only of size sizeof(DataType)*max_items_in_output_buffer*num_res_attrs
         * using cudaHostAlloc, system malloc and cudaMallocManaged do not work*/
        DataType **res_cpu = nullptr;
        checkCudaErrors(cudaHostAlloc((void**)&res_cpu, sizeof(DataType*)*num_res_attrs, cudaHostAllocDefault));
        for(auto i = 0; i < num_res_attrs; i++) {
            checkCudaErrors(cudaHostAlloc((void**)&res_cpu[i], sizeof(DataType)*max_items_in_output_buffer, cudaHostAllocDefault));
        }

        /*prefetch the frist batch*/
        prefetch(probe_table, probe_table_slices[input_buffer_idx],
                 probe_offset, intervals[0], streams[1]);

        cudaDeviceSynchronize();
        for(auto cur_iteration = 0; cur_iteration < num_iterations; cur_iteration++) {
            if (cur_iteration > 0) { /*copy previous results back to CPU*/
                for(auto i = 0; i < num_res_attrs; i++) {
                    checkCudaErrors(cudaMemcpyAsync(res_cpu[i], res_ooc[1-output_buffer_idx][i],
                                                    sizeof(DataType)*max_items_in_output_buffer,
                                                    cudaMemcpyDeviceToHost, streams[0]));
                }
            }
            if (cur_iteration < num_iterations-1) { //prefetch the next batch
                auto new_probe_offset = probe_offset + intervals[cur_iteration];
                prefetch(probe_table, probe_table_slices[1 - input_buffer_idx],
                         new_probe_offset, intervals[cur_iteration+1], streams[1]);
            }

            log_info("------------- Join iteration %d -------------", cur_iteration);
            log_info("Probe table offset=%llu, len=%llu", probe_offset, intervals[cur_iteration]);

            switch (algo_type) {
                case TYPE_MHJ: {
                    res_cnt += MHJ_instance.MHJ_evaluate
                            (probe_table_slices[input_buffer_idx], hash_tables,
                             num_hash_tables, used_for_compare, bucketVec,
                             attr_idxes_in_iRes, num_attr_idxes_in_iRes,
                             res_ooc[output_buffer_idx], max_items_in_output_buffer, streams[2]);
                    break;
                }
                case TYPE_AMHJ: {
                    res_cnt += AMHJ_instance.AMHJ_evaluate
                            (probe_table_slices[input_buffer_idx], hash_tables,
                             num_hash_tables, used_for_compare, bucketVec,
                             attr_idxes_in_iRes, num_attr_idxes_in_iRes,
                             res_ooc[output_buffer_idx], max_items_in_output_buffer, streams[2]);
                    break;
                }
                case TYPE_AMHJ_WS: {
                    res_cnt += AMHJ_WS_instance.AMHJ_WS_evaluate
                            (probe_table_slices[input_buffer_idx], hash_tables,
                             num_hash_tables, used_for_compare, bucketVec,
                             attr_idxes_in_iRes, num_attr_idxes_in_iRes,
                             res_ooc[output_buffer_idx], max_items_in_output_buffer, streams[2]);
                    break;
                }
                case TYPE_AMHJ_DRO: {
                    res_cnt += AMHJ_DRO_instance.AMHJ_DRO_evaluate
                            (probe_table_slices[input_buffer_idx], hash_tables,
                             num_hash_tables, used_for_compare, bucketVec,
                             attr_idxes_in_iRes, num_attr_idxes_in_iRes,
                             res_ooc[output_buffer_idx], max_items_in_output_buffer, streams[2]);
                    break;
                }
                default: {
                    log_error("Unsupported algorithm");
                    exit(1);
                }
            }
            cudaDeviceSynchronize();

            probe_offset += intervals[cur_iteration];
            output_buffer_idx = 1 - output_buffer_idx; //switch output buffer
            input_buffer_idx = 1 - input_buffer_idx;   //switch input buffer
            cudaDeviceSynchronize();
        }
        /*copy back the last batch*/
        for(auto i = 0; i < num_res_attrs; i++) {
            checkCudaErrors(cudaMemcpyAsync(res_cpu[i], res_ooc[1 - output_buffer_idx][i],
                                            sizeof(DataType) * max_items_in_output_buffer,
                                            cudaMemcpyDeviceToHost, streams[0]));
        }
        cudaDeviceSynchronize();
        log_info("Total number of outputs: %llu", res_cnt);
        return res_cnt;
    }
};
