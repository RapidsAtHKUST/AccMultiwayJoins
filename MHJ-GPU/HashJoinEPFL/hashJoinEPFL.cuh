/*Copyright (c) 2018 Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

#pragma once

#include <algorithm>
#include <cassert>
#include <cinttypes>

#include "common.h"
#include "join-primitives.cuh"

#include <vector>
#include <iostream>
#include <unistd.h>
#include <list>
#include "../Relation.cuh"
#include "cuda/primitives.cuh"
using namespace std;

template<typename CntType>
CntType hashJoinEPFL(
        int* R, uint32_t RelsNum, int* S, uint32_t SelsNum, int *match_indexes[2],
        CUDAMemStat *memstat, CUDATimeStat *timing) {
    log_trace("Function: %s", __FUNCTION__);
    log_info("probe table length: %llu, build table length: %llu", RelsNum, SelsNum);


    Timer t;
    uint32_t first_bit = 0;
    uint32_t parts2 = 1 << (log_parts1 + log_parts2);

    size_t buckets_num_max_R = 2*((((RelsNum + parts2 - 1)/parts2) + bucket_size - 1)/bucket_size)*parts2 + 1024;
    size_t buckets_num_max_S = 2*((((SelsNum + parts2 - 1)/parts2) + bucket_size - 1)/bucket_size)*parts2 + 1024;

    int *R_gpu_final, *S_gpu_final;
    int *R_gpu_temp, *S_gpu_temp;
    int *Pr_gpu_final, *Ps_gpu_final;
    int *Pr_gpu_temp, *Ps_gpu_temp;

    CUDA_MALLOC(&R_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t), memstat);
    CUDA_MALLOC(&Pr_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t), memstat);
    CUDA_MALLOC(&R_gpu_final, buckets_num_max_R * bucket_size * sizeof(int32_t), memstat);
    CUDA_MALLOC(&Pr_gpu_final, buckets_num_max_R * bucket_size * sizeof(int32_t), memstat);
    log_debug("R_gpu_final size: %llu", buckets_num_max_R * bucket_size);

    CUDA_MALLOC(&S_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t), memstat);
    CUDA_MALLOC(&Ps_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t), memstat);
    CUDA_MALLOC(&S_gpu_final, buckets_num_max_S * bucket_size * sizeof(int32_t), memstat);
    CUDA_MALLOC(&Ps_gpu_final, buckets_num_max_S * bucket_size * sizeof(int32_t), memstat);
    log_debug("S_gpu_final size: %llu", buckets_num_max_S * bucket_size);

    int *Pr, *Ps; //indexes
    CUDA_MALLOC(&Pr, sizeof(int)*RelsNum, memstat);
    CUDA_MALLOC(&Ps, sizeof(int)*SelsNum, memstat);

    cudaMemset(R_gpu_temp, 0, buckets_num_max_R * bucket_size * sizeof(int32_t));
    cudaMemset(Pr_gpu_temp, 0, buckets_num_max_R * bucket_size * sizeof(int32_t));

    uint32_t *chains_R[2], *chains_S[2];
    uint32_t *cnts_R[2], *cnts_S[2];
    uint64_t *heads_R[2], *heads_S[2];
    uint32_t *buckets_used_R[2], *buckets_used_S[2];

    int* aggr_cnt;
    CUDA_MALLOC(&aggr_cnt, 64*sizeof(int), memstat);
    cudaMemset(aggr_cnt, 0, 64 * sizeof(int));

    for (int i = 0; i < 2; i++) {
        CUDA_MALLOC(&chains_R[i], buckets_num_max_R * sizeof(uint32_t), memstat);
        CUDA_MALLOC(&cnts_R[i], parts2 * sizeof(uint32_t), memstat);
        CUDA_MALLOC(&heads_R[i], parts2 * sizeof(uint64_t), memstat);
        CUDA_MALLOC(&buckets_used_R[i], sizeof(uint32_t), memstat);

        CUDA_MALLOC(&chains_S[i], buckets_num_max_S * sizeof(uint32_t), memstat);
        CUDA_MALLOC(&cnts_S[i], parts2 * sizeof(uint32_t), memstat);
        CUDA_MALLOC(&heads_S[i], parts2 * sizeof(uint64_t), memstat);
        CUDA_MALLOC(&buckets_used_S[i], sizeof(uint32_t), memstat);
    }

    checkCudaErrors(cudaDeviceSynchronize());

    thrust::counting_iterator<int> iter((int)0);
    auto gpu_time_stamp = timing->get_idx();
    timingKernel(thrust::copy(iter, iter + RelsNum, Pr), timing);
    timingKernel(thrust::copy(iter, iter + SelsNum, Ps), timing);
    log_info("Prepare CPU time: %.2f s.", t.elapsed());
    log_info("Prepare GPU time: %.2f ms.", timing->diff_time(gpu_time_stamp));

    auto timestamp = timing->get_idx();
    prepare_Relation_payload_triple (  R, R_gpu_temp, R_gpu_final,
                                       Pr, Pr_gpu_temp, Pr_gpu_final,
                                       RelsNum,
                                       buckets_num_max_R, heads_R, cnts_R, chains_R, buckets_used_R,
                                       log_parts1, log_parts2, first_bit, 0, NULL, OMP_PARALLELISM1, timing);
    prepare_Relation_payload_triple (  S, S_gpu_temp, S_gpu_final,
                                       Ps, Ps_gpu_temp, Ps_gpu_final,
                                       SelsNum,
                                       buckets_num_max_S, heads_S, cnts_S, chains_S, buckets_used_S,
                                       log_parts1, log_parts2, first_bit, 0, NULL, OMP_PARALLELISM1, timing);

    uint32_t* bucket_info_R = (uint32_t*) Pr_gpu_temp;
    execKernelDynamicAllocation(decompose_chains, (1 << log_parts1), 1024, 0, timing, false, bucket_info_R, chains_R[1], cnts_R[1], log_parts1 + log_parts2, 2*bucket_size);
    log_info("Total Build kernel time: %.2f ms.", timing->diff_time(timestamp));

    /*probe count*/
    timestamp = timing->get_idx();
    execKernelDynamicAllocation(join_partitioned_aggregate,(1 << log_parts1), 512, 0, timing, false,R_gpu_final, Pr_gpu_final, chains_R[1], bucket_info_R,S_gpu_final, Ps_gpu_final, cnts_S[1], chains_S[1],log_parts1 + log_parts2, buckets_used_R[1], &aggr_cnt[0]);
    log_info("Probe-count kernel time: %.2f ms.", timing->diff_time(timestamp));

    auto num_matches = aggr_cnt[0];
    log_info("Output count: %llu", num_matches);
    for(auto i = 0; i < 2; i++) CUDA_MALLOC(&match_indexes[i], sizeof(CntType)*num_matches, memstat);
    cudaMemset(aggr_cnt, 0, 64 * sizeof(int)); // still necessary because probe-write also updates it

    /*probe write*/
    timestamp = timing->get_idx();
    execKernelDynamicAllocation(join_partitioned_results, (1 << log_parts1), 512, 0, timing, false, R_gpu_final, Pr_gpu_final, chains_R[1], bucket_info_R, S_gpu_final, Ps_gpu_final, cnts_S[1], chains_S[1], log_parts1 + log_parts2, buckets_used_R[1], &aggr_cnt[0], match_indexes[0], match_indexes[1]);
    log_info("Probe-write kernel time: %.2f ms.", timing->diff_time(timestamp));
    assert(aggr_cnt[0] == num_matches);

    /*Free the memory*/
    CUDA_FREE(R_gpu_temp, memstat);
    CUDA_FREE(Pr_gpu_temp, memstat);
    CUDA_FREE(R_gpu_final, memstat);
    CUDA_FREE(Pr_gpu_final, memstat);

    CUDA_FREE(S_gpu_temp, memstat);
    CUDA_FREE(Ps_gpu_temp, memstat);
    CUDA_FREE(S_gpu_final, memstat);
    CUDA_FREE(Ps_gpu_final, memstat);

    CUDA_FREE(Pr, memstat);
    CUDA_FREE(Ps, memstat);
    CUDA_FREE(aggr_cnt, memstat);

    for (int i = 0; i < 2; i++) {
        CUDA_FREE(chains_R[i], memstat);
        CUDA_FREE(cnts_R[i], memstat);
        CUDA_FREE(heads_R[i], memstat);
        CUDA_FREE(buckets_used_R[i], memstat);

        CUDA_FREE(chains_S[i], memstat);
        CUDA_FREE(cnts_S[i], memstat);
        CUDA_FREE(heads_S[i], memstat);
        CUDA_FREE(buckets_used_S[i], memstat);
    }
    return num_matches;
}

/* pairwise hash join (right-deep join)
 * join order: input_tables[0] Join input_tables[1] Join input_tables[2] Join...
 * the 1st attr of each build table is its join attr
 * */
template<typename DataType, typename CntType>
CntType PW(Relation<DataType, CntType> *input_tables, uint32_t num_tables, DataType **&res, int num_res_attrs,
           CUDAMemStat *memstat, CUDATimeStat *timing) {
    log_trace("Function: %s", __FUNCTION__);

    /*materialized_indexes and materialized_table store materialized indexes and data */
    vector<int*> materialized_indexes(num_tables, nullptr);//join match indexes of the joined tables
    Relation<DataType,CntType> materialized_table;//join match data of the joined tables for next join

    /*init materialized_indexes and materialized_table*/
    thrust::counting_iterator<CntType> iter(0);
    materialized_table.init(1, input_tables[0].length, input_tables[0].memstat);//todo: only a single attr
    CUDA_MALLOC(&materialized_indexes[0], sizeof(int)*materialized_table.length, memstat);
    timingKernel(thrust::copy(iter, iter + materialized_table.length, materialized_indexes[0]), timing);
    for(auto xp = 0; xp < input_tables[0].num_attrs; xp++) { /*find join attr*/
        auto join_attr = input_tables[1].attr_list[0];
        if (input_tables[0].attr_list[xp] == join_attr) {
            materialized_table.data[0] = input_tables[0].data[xp];
            materialized_table.attr_list[0] = input_tables[0].attr_list[xp];
            break;
        }
    }
    vector<RelType> attr_found_in_rel(num_res_attrs, INVALID_REL_VAL);//in which table can the attr be found
    vector<AttrType> attr_idx_in_rel(num_res_attrs);//join attr index of this table
    for(auto i = 0; i < input_tables[0].num_attrs; i++) { //set attr_found_in_rel w.r.t. the probe table
        attr_found_in_rel[input_tables[0].attr_list[i]] = (RelType)0;
        attr_idx_in_rel[input_tables[0].attr_list[i]] = (AttrType)i;
    }
    CntType num_matches = 0;

    for(auto join_idx = 1; join_idx < num_tables; join_idx++) {
        int* match_indexes[2];
        num_matches = hashJoinEPFL<CntType>(materialized_table.data[0], (uint32_t)materialized_table.length,
                                   input_tables[join_idx].data[0], (uint32_t)input_tables[join_idx].length,
                                   match_indexes, memstat, timing);
        /*update materialized_indexes*/
        auto timestamp = timing->get_idx();
        for(auto i = 0; i < materialized_indexes.size(); i++) {
            if (materialized_indexes[i] != nullptr) {
                int *cur_indexes = nullptr;
                CUDA_MALLOC(&cur_indexes, sizeof(int)*num_matches, memstat);
                execKernel(gather, GRID_SIZE, BLOCK_SIZE, timing, false,
                           materialized_indexes[i], cur_indexes, match_indexes[0], num_matches);
                CUDA_FREE(materialized_indexes[i], memstat);
                materialized_indexes[i] = cur_indexes;
            }
        }
        materialized_indexes[join_idx] = match_indexes[1];//append the index of the build table
        CUDA_FREE(match_indexes[0], memstat);

        /*update attr_found_in_rel w.r.t. the build table*/
        for(auto i = 0; i < input_tables[join_idx].num_attrs; i++) {
            if (attr_found_in_rel[input_tables[join_idx].attr_list[i]] == INVALID_REL_VAL) {
                attr_found_in_rel[input_tables[join_idx].attr_list[i]] = (RelType)join_idx;
                attr_idx_in_rel[input_tables[join_idx].attr_list[i]] = (AttrType)i;
            }
        }
        if (join_idx != num_tables-1) { //not reaching the last join, update materialized_table
            auto rel_with_join_attr = attr_found_in_rel[input_tables[join_idx+1].attr_list[0]];
            auto idx_in_rel = attr_idx_in_rel[input_tables[join_idx+1].attr_list[0]];
            assert(rel_with_join_attr != INVALID_REL_VAL);

            DataType *new_materialized_col = nullptr;
            CUDA_MALLOC(&new_materialized_col, sizeof(DataType)*num_matches, memstat);
            execKernel(gather, GRID_SIZE, BLOCK_SIZE, timing, false,
                       input_tables[rel_with_join_attr].data[idx_in_rel],
                       new_materialized_col, materialized_indexes[rel_with_join_attr], num_matches);
            materialized_table.data[0] = new_materialized_col;
            materialized_table.length = num_matches;
            materialized_table.attr_list[0] = input_tables[join_idx+1].attr_list[0];
        }
        else { //materialize all the columns
            /*check whether each attr is attached to an input rel*/
            CUDA_MALLOC(&res, sizeof(DataType*)*num_res_attrs, memstat);
            for(auto i = 0; i < num_res_attrs; i++) {
                assert(attr_found_in_rel[i] != INVALID_REL_VAL);
                CUDA_MALLOC(&res[i], sizeof(DataType)*num_matches, memstat);
                execKernel(gather, GRID_SIZE, BLOCK_SIZE, timing, false,
                           input_tables[attr_found_in_rel[i]].data[attr_idx_in_rel[i]],
                           res[i], materialized_indexes[attr_found_in_rel[i]], num_matches);
            }
        }
        log_info("Materialization kernel time: %.2f ms.", timing->diff_time(timestamp));
    }
    return num_matches;
}