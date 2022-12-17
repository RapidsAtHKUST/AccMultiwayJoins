//
// Created by Bryan on 4/2/2020.
//

#pragma once

#include "config.h"
#include "cuda/primitives.cuh"
#include "helper.h"
#include "timer.h"
#include "types.h"
#include "helper_cuda.h"
#include "IndexedTrie.cuh"
#include "LFTJ_base.h"

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <moderngpu/kernel_sortedsearch.hxx>
#include <moderngpu/kernel_load_balance.hxx>
#include <cooperative_groups.h>

//#define HASH_PROBE

using namespace std;
using namespace cooperative_groups;
using namespace mgpu;

template<typename NeedleType, typename HaystackType, typename CntType>
__global__
void binarySearchForMatchAndIndexes(
        NeedleType *needles, CntType num_needles,        /*needles are unsorted*/
        HaystackType *haystacks, CntType num_haystacks,    /*haystacks are sorted*/
        bool *bitmaps, CntType *matchIdx) {
    auto gtid = (CntType)(threadIdx.x + blockIdx.x * blockDim.x);
    auto gtnum = (CntType)(blockDim.x * gridDim.x);

    while (gtid < num_needles) {
        bool found = false;
        NeedleType needle = needles[gtid];
        int middle, begin = 0, end = num_haystacks;
        while (begin <= end) {
            middle = begin + (end - begin)/2;
            if (needle > haystacks[middle])
                begin = middle + 1;
            else if (needle < haystacks[middle])
                end = middle - 1;
            else { /*found the match*/
                found = true;
                break;
            }
        }
        bitmaps[gtid] = found;
        matchIdx[gtid] = (CntType)middle;
        gtid += gtnum;
    }
}

/*
 * matchNeedles[i] is 1 if needles[i] finds a match in haystack
 * Both the needles and haystack arrays should be sorted.
 * */
template<typename DataType, typename CntType>
void MGPUFindMatchBoolean(
        DataType *needles, CntType needle_num,
        DataType *haystack, CntType haystack_num,
        bool *matchNeedles, standard_context_t *context,
        CUDAMemStat *memstat, CUDATimeStat *timing) {
    auto asc_begin = thrust::make_counting_iterator((CntType)0);
    auto asc_end = thrust::make_counting_iterator(needle_num);
    CntType *matchIdx = nullptr;
    CUDA_MALLOC(&matchIdx, sizeof(CntType)*needle_num, memstat);

    /*
     * Using MGPU sorted search to compute the lower bound index of each needle in haystack
     * */
    timingKernel(
            sorted_search<bounds_lower>(needles, needle_num, haystack, haystack_num, matchIdx, less_t<DataType>(), *context), timing);
    /*
     * Check whether the value in the lower bound indexed slot is equal to needle
     * */
    timingKernel(
            thrust::transform(thrust::device, asc_begin, asc_end, matchNeedles, [=] __device__(CntType idx) {
            return (matchIdx[idx] < haystack_num) && (needles[idx] == haystack[matchIdx[idx]]);
    }), timing); //matchIdx[idx] < haystack_num is necessary
    CUDA_FREE(matchIdx, memstat);
}

/*
 * return the number of matches between two sorted arrays of pairs
 * Both the needles and haystack arrays should be sorted according to key, then value.
 * */
template<typename KeyType, typename ValueType, typename CntType>
CntType ThrustCountMatchPair(
        KeyType *needles_keys, ValueType *needles_values, CntType needle_num,
        KeyType *haystack_keys, ValueType *haystack_values, CntType haystack_num,
        CUDAMemStat *memstat, CUDATimeStat *timing) {
    Timer t;
    CntType *output = nullptr;
    bool *is_match = nullptr;
    CUDA_MALLOC(&output, sizeof(CntType)*needle_num, memstat);
    CUDA_MALLOC(&is_match, sizeof(bool)*needle_num, memstat);

    timingKernel(thrust::lower_bound(
            thrust::device,
            thrust::make_zip_iterator(thrust::make_tuple(haystack_keys, haystack_values)),
            thrust::make_zip_iterator(thrust::make_tuple(haystack_keys+haystack_num, haystack_values+haystack_num)),
            thrust::make_zip_iterator(thrust::make_tuple(needles_keys, needles_values)),
            thrust::make_zip_iterator(thrust::make_tuple(needles_keys+needle_num, needles_values+needle_num)),
            output), timing);
    cudaDeviceSynchronize();

    auto asc_begin = thrust::make_counting_iterator((CntType)0);
    auto asc_end = thrust::make_counting_iterator(needle_num);
    timingKernel(
        thrust::transform(thrust::device, asc_begin, asc_end, is_match, [=] __device__ (CntType idx) {
            return (output[idx] < haystack_num) && (needles_keys[idx] == haystack_keys[output[idx]]) && (needles_values[idx] == haystack_values[output[idx]]);
    }), timing);
    CntType cnt = CUBSum<bool,CntType,CntType>(is_match, needle_num, memstat, timing);

    CUDA_FREE(output, memstat);
    CUDA_FREE(is_match, memstat);
    return cnt;
}

template<typename DataType, typename CntType>
class LFTJ_BFS : public LFTJ_Base<DataType,CntType> {
public:

    LFTJ_BFS(uint32_t num_tables, uint32_t num_attrs, uint32_t *attr_order, CUDAMemStat *memstat, CUDATimeStat *timing){
        this->num_tables = num_tables;
        this->num_attrs = num_attrs;
        this->attr_order = attr_order;
        this->memstat = memstat;
        this->timing = timing;
    };

    /*
     * Two unprocessed tables, on the first attr
     * CSR parent join parent, items in each table are unique
     *
     * The order and order+1 column of res will be populated
     * */
    void np_join_np(
            IndexedTrie<DataType,CntType> rel_np0, uint32_t rel_np0_index,
            IndexedTrie<DataType,CntType> rel_np1, uint32_t rel_np1_index,
            CntType **res, CntType &res_num,
            standard_context_t *context) {
        Timer t;
        log_trace("In function: %s", __FUNCTION__);
        auto gpu_time_idx = this->timing->get_idx();
        bool *match_A = nullptr, *match_B = nullptr;
        CUDA_MALLOC(&match_A, sizeof(bool)*rel_np0.data_len[0], this->memstat);
        CUDA_MALLOC(&match_B, sizeof(bool)*rel_np1.data_len[0], this->memstat);

        MGPUFindMatchBoolean(rel_np0.data[0], rel_np0.data_len[0], rel_np1.data[0], rel_np1.data_len[0], match_A, context, this->memstat, this->timing);
        MGPUFindMatchBoolean(rel_np1.data[0], rel_np1.data_len[0], rel_np0.data[0], rel_np0.data_len[0], match_B, context, this->memstat, this->timing);

        /*construct ascending arrays using Thrust*/
        CntType *asc_A = nullptr, *asc_B = nullptr;
        CUDA_MALLOC(&asc_A, sizeof(CntType)*rel_np0.data_len[0], this->memstat);
        CUDA_MALLOC(&asc_B, sizeof(CntType)*rel_np1.data_len[0], this->memstat);

        thrust::counting_iterator<CntType> iter(0);
        timingKernel(thrust::copy(iter, iter + rel_np0.data_len[0], asc_A), this->timing);
        timingKernel(thrust::copy(iter, iter + rel_np1.data_len[0], asc_B), this->timing);

        CntType *res_A = nullptr, *res_B = nullptr;
        CUDA_MALLOC(&res_A, sizeof(CntType)*rel_np0.data_len[0], this->memstat);
        CUDA_MALLOC(&res_B, sizeof(CntType)*rel_np1.data_len[0], this->memstat);

        auto resNum_A = CUBSelect(asc_A, res_A, match_A, rel_np0.data_len[0], this->memstat, this->timing);
        auto resNum_B = CUBSelect(asc_B, res_B, match_B, rel_np1.data_len[0], this->memstat, this->timing);
        assert(resNum_A == resNum_B);

#ifdef FREE_DATA
        CUDA_FREE(match_A, this->memstat);
        CUDA_FREE(match_B, this->memstat);
        CUDA_FREE(asc_A, this->memstat);
        CUDA_FREE(asc_B, this->memstat);
#endif
        res_num= resNum_A;
        res[rel_np0_index] = res_A;
        res[rel_np1_index] = res_B;
        log_info("%s kernel time: %.2f ms.", __FUNCTION__, this->timing->diff_time(gpu_time_idx));
        log_info("%s CPU time: %.2f ms.", __FUNCTION__, t.elapsed()*1000);
    }

    /*
     * Joining one processed table and an unprocessed table
     * CSR parent join children, items in the parent CSR are unique
     *
     * The order and order+1 column of res will be populated
     * rel_p: the table whose child column will be joined
     * rel_np: the table whose parent column will be joined (keys are unique)
     * */
    void p_join_np(
            IndexedTrie<DataType,CntType> rel_p, uint32_t rel_p_index,
            IndexedTrie<DataType,CntType> rel_np, uint32_t rel_np_index,
            vector<uint32_t> processed_rel_list,
            CntType **res, CntType &res_num,
            standard_context_t *context) {
        Timer t;
        log_trace("In function: %s", __FUNCTION__);
        auto gpu_time_idx = this->timing->get_idx();
        CntType *col_parent_p = res[rel_p_index];
        auto num_pars_in_rel_p = res_num; //number of parent items in rel_p

        /*1st: get children count of rel_p*/
        int *children_cnts_in_rel_p;
        CUDA_MALLOC(&children_cnts_in_rel_p, sizeof(int)*num_pars_in_rel_p, this->memstat);

        auto asc_begin = thrust::make_counting_iterator((CntType)0);
        auto asc_end = thrust::make_counting_iterator(num_pars_in_rel_p);

        /*2nd: extract the number of children for each parent item in rel_p*/
        timingKernel(
                thrust::transform(thrust::device, asc_begin, asc_end, children_cnts_in_rel_p, [=] __device__(CntType idx) {
                return rel_p.offsets[0][col_parent_p[idx]+1] - rel_p.offsets[0][col_parent_p[idx]];
        }), this->timing);

        unsigned long long sum_p = 0;
        float sum_p_in_GB;
        sum_p = CUBSum<int,unsigned long long,CntType>(children_cnts_in_rel_p, num_pars_in_rel_p, this->memstat, this->timing);
        sum_p_in_GB = sum_p * sizeof(int) * 1.0f / 1024/1024/1024;
        log_info("p expanded children num: %lu (%.2f GB)", sum_p, sum_p_in_GB);
        if (sum_p_in_GB > 32) {
            log_error("Counter overflow. Exit");
            exit(1);
        }
        CntType num_children_in_rel_p = CUBScanExclusive(children_cnts_in_rel_p, children_cnts_in_rel_p, num_pars_in_rel_p, this->memstat, this->timing);

        int *lbs = nullptr; //load-balancing search res, to store the expanded children
        CUDA_MALLOC(&lbs, sizeof(int)*num_children_in_rel_p, this->memstat);

        /*3rd: load balance search*/
        timingKernel(load_balance_search(num_children_in_rel_p, children_cnts_in_rel_p, num_pars_in_rel_p, lbs, *context), this->timing);
        log_trace("finish load_balance_search");

        /*4th: compute the filtered children values of relation a when intersecting with b*/
        DataType *expanded_children_val = nullptr;
        CntType *expanded_children_ptr = nullptr;
        CUDA_MALLOC(&expanded_children_val, sizeof(DataType)*num_children_in_rel_p, this->memstat);
        CUDA_MALLOC(&expanded_children_ptr, sizeof(CntType)*num_children_in_rel_p, this->memstat);

        /*populate the expanded children ptrs*/
        asc_end = thrust::make_counting_iterator(num_children_in_rel_p);
        timingKernel(
                thrust::transform(thrust::device, asc_begin, asc_end, expanded_children_ptr, [=] __device__(CntType idx) {
                auto rank = idx - children_cnts_in_rel_p[lbs[idx]];
                auto startPos = rel_p.offsets[0][col_parent_p[lbs[idx]]] - rel_p.trie_offsets[1];
                return rank + startPos;
        }), this->timing);

        /*populate the expanded children vals*/
        timingKernel(
                thrust::transform(thrust::device, asc_begin, asc_end, expanded_children_val, [=] __device__(CntType idx) {
                return rel_p.data[1][expanded_children_ptr[idx]];
        }), this->timing);
#ifdef FREE_DATA
        CUDA_FREE(children_cnts_in_rel_p, this->memstat);
#endif
        /*5th: compute the bitmap and matching indexes using binary searches*/
        bool *bitmap = nullptr;
        CntType *match_idx = nullptr;
        CUDA_MALLOC(&bitmap, sizeof(bool)*num_children_in_rel_p, this->memstat);
        CUDA_MALLOC(&match_idx, sizeof(CntType)*num_children_in_rel_p, this->memstat);

        /*binary search solution*/
        execKernel(binarySearchForMatchAndIndexes, GRID_SIZE, BLOCK_SIZE, this->timing, false, expanded_children_val, num_children_in_rel_p, rel_np.data[0], rel_np.data_len[0], bitmap, match_idx);
#ifdef FREE_DATA
        CUDA_FREE(expanded_children_val, this->memstat);
#endif
        /*compute the number of matches*/
        auto num_matches = CUBSum<bool,CntType,CntType>(bitmap, num_children_in_rel_p, this->memstat, this->timing);
        if (0 == num_matches)    {
            res_num = num_matches;
            log_info("No matches!");
            return;
        }

        /*6th: update the res of each relevant relation*/
        /*1)update all the previous relations that are joined with the processed relation before
         * Approach: get the filtered parents (lbs+bitmap) of the processed relation and update the prev_res of those previous relations
         * */
        int *selected_lbs = nullptr;
        CUDA_MALLOC(&selected_lbs, sizeof(int)*num_matches, this->memstat);
        CUBSelect(lbs, selected_lbs, bitmap, num_children_in_rel_p, this->memstat, this->timing);
#ifdef FREE_DATA
        CUDA_FREE(lbs, this->memstat);
#endif
        /*construct new result and replace the old one for each previously processed rel*/
        for(auto i = 0; i < processed_rel_list.size(); i++) {
            auto processed_rel_index = processed_rel_list[i];
            CntType *new_prev = nullptr;
            CUDA_MALLOC(&new_prev, sizeof(CntType)*num_matches, this->memstat);
            execKernel(gather, GRID_SIZE, BLOCK_SIZE, this->timing, false, res[processed_rel_index], new_prev, selected_lbs, num_matches);
#ifdef FREE_DATA
            CUDA_FREE(res[processed_rel_index], this->memstat);
#endif
            res[processed_rel_index] = new_prev;
        }
#ifdef FREE_DATA
        CUDA_FREE(selected_lbs, this->memstat);
#endif
        /*2)update the rel_p relation in this function (res indicates its children)
         * Approach: select the children ptr, flag = bitmap
         * */
        CntType *new_col_for_C = nullptr;
        CUDA_MALLOC(&new_col_for_C, sizeof(CntType)*num_matches, this->memstat);
        CUBSelect(expanded_children_ptr, new_col_for_C, bitmap, num_children_in_rel_p, this->memstat, this->timing);
#ifdef FREE_DATA
        CUDA_FREE(expanded_children_ptr, this->memstat);
        CUDA_FREE(res[rel_p_index], this->memstat);
#endif
        res[rel_p_index] = new_col_for_C;

        /*3)update the rel_np relation in this function (res indicates its parents) and append it to the final result list
         * Approach: select the match indexes, flag = bitmap
         * */
        CntType *new_col_for_P = nullptr;
        CUDA_MALLOC(&new_col_for_P, sizeof(CntType)*num_matches, this->memstat);
        CUBSelect(match_idx, new_col_for_P, bitmap, num_children_in_rel_p, this->memstat, this->timing);
#ifdef FREE_DATA
        CUDA_FREE(match_idx, this->memstat);
        CUDA_FREE(bitmap, this->memstat);
#endif
        res[rel_np_index] = new_col_for_P;
        res_num = num_matches; //update the result count

        log_info("%s kernel time: %.2f ms.", __FUNCTION__, this->timing->diff_time(gpu_time_idx));
        log_info("%s CPU time: %.2f ms.", __FUNCTION__, t.elapsed()*1000);
    }

    /*
     * Joining two processed table
     * CSR children join children
     *
     * The order and order+1 column of res will be populated
     * */
    void p_join_p(
            IndexedTrie<DataType,CntType> rel_p0, uint32_t rel_p0_index,
            IndexedTrie<DataType,CntType> rel_p1, uint32_t rel_p1_index,
            CntType **res, CntType &res_num,
            standard_context_t *context) {
        Timer t;
        log_trace("In function: %s", __FUNCTION__);
        auto gpu_time_idx = this->timing->get_idx();
        CntType *col_parent_p0 = res[rel_p0_index];
        CntType *col_parent_p1 = res[rel_p1_index];
        auto num_pars = res_num; //number of parent items in rel_p0 and rel_p1

        /*1st: get children count of rel_p0 and rel_p1*/
        CntType *children_cnts_in_p0, *children_cnts_in_p1;
        CUDA_MALLOC(&children_cnts_in_p0, sizeof(CntType)*num_pars, this->memstat);
        CUDA_MALLOC(&children_cnts_in_p1, sizeof(CntType)*num_pars, this->memstat);

        auto asc_begin = thrust::make_counting_iterator((CntType)0);
        auto asc_end = thrust::make_counting_iterator(num_pars);

        /*2nd: extract the number of children for each parent item in rel_p0 and rel_p1*/
        timingKernel(
                thrust::transform(thrust::device, asc_begin, asc_end, children_cnts_in_p0, [=] __device__(CntType idx) {
                return rel_p0.offsets[0][col_parent_p0[idx]+1] - rel_p0.offsets[0][col_parent_p0[idx]];
        }), this->timing);
        timingKernel(
                thrust::transform(thrust::device, asc_begin, asc_end, children_cnts_in_p1, [=] __device__(CntType idx) {
                return rel_p1.offsets[0][col_parent_p1[idx]+1] - rel_p1.offsets[0][col_parent_p1[idx]];
        }), this->timing);

        unsigned long long sum_p0 = 0, sum_p1 = 0;
        float sum_p0_in_GB, sum_p1_in_GB;
        sum_p0 = CUBSum<CntType,unsigned long long,CntType>(children_cnts_in_p0, num_pars, this->memstat, this->timing);
        sum_p1 = CUBSum<CntType,unsigned long long,CntType>(children_cnts_in_p1, num_pars, this->memstat, this->timing);
        sum_p0_in_GB = sum_p0 * sizeof(int) * 1.0f / 1024/1024/1024;
        sum_p1_in_GB = sum_p1 * sizeof(int) * 1.0f / 1024/1024/1024;
        log_info("p0 expanded children num: %lu (%.2f GB)", sum_p0, sum_p0_in_GB);
        log_info("p1 expanded children num: %lu (%.2f GB)", sum_p1, sum_p1_in_GB);
        if (sum_p0_in_GB > 32 || sum_p1_in_GB > 32) {
            log_error("Counter overflow. Exit");
            exit(1);
        }

        CntType num_children_in_p0 = CUBScanExclusive(children_cnts_in_p0, children_cnts_in_p0, num_pars, this->memstat, this->timing);
        CntType num_children_in_p1 = CUBScanExclusive(children_cnts_in_p1, children_cnts_in_p1, num_pars, this->memstat, this->timing);

        int *lbs_p0 = nullptr, *lbs_p1 = nullptr; //load-balancing search res, to store the expanded children
        CUDA_MALLOC(&lbs_p0, sizeof(int)*num_children_in_p0, this->memstat);
        CUDA_MALLOC(&lbs_p1, sizeof(int)*num_children_in_p1, this->memstat);

        /*3rd: load balance search*/
        /*todo: the load balancing search of ModernGPU cannot handle large datasets (e.g. Livejournal), now change to my own CPU load balance search*/
        log_debug("begin lbs");
//        timingKernel(load_balance_search(num_children_in_p0, children_cnts_in_p0, num_pars, lbs_p0, *context), this->timing);
//        timingKernel(load_balance_search(num_children_in_p1, children_cnts_in_p1, num_pars, lbs_p1, *context), this->timing);
        Timer tt;
        zlai_parallel_load_balance_search(num_children_in_p0, children_cnts_in_p0, num_pars, lbs_p0);
        zlai_parallel_load_balance_search(num_children_in_p1, children_cnts_in_p1, num_pars, lbs_p1);
        log_debug("lbs CPU time: %.1f s", tt.elapsed());
        log_debug("end lbs");

        /*4th: compute the filtered children values of relation a when intersecting with b*/
        DataType *expanded_children_val_p0 = nullptr, *expanded_children_val_p1 = nullptr;
        CntType *expanded_children_ptr_p0 = nullptr, *expanded_children_ptr_p1 = nullptr;
        CUDA_MALLOC(&expanded_children_val_p0, sizeof(DataType)*num_children_in_p0, this->memstat);
        CUDA_MALLOC(&expanded_children_ptr_p0, sizeof(CntType)*num_children_in_p0, this->memstat);
        CUDA_MALLOC(&expanded_children_val_p1, sizeof(DataType)*num_children_in_p1, this->memstat);
        CUDA_MALLOC(&expanded_children_ptr_p1, sizeof(CntType)*num_children_in_p1, this->memstat);

        /*populate the expanded children ptrs and vals*/
        asc_end = thrust::make_counting_iterator(num_children_in_p0);
        timingKernel(
                thrust::transform(thrust::device, asc_begin, asc_end, expanded_children_ptr_p0, [=] __device__(CntType idx) {
                auto rank = idx - children_cnts_in_p0[lbs_p0[idx]];
                auto startPos = rel_p0.offsets[0][col_parent_p0[lbs_p0[idx]]] - rel_p0.trie_offsets[1];
                return rank + startPos;
        }), this->timing);
        timingKernel(
                thrust::transform(thrust::device, asc_begin, asc_end, expanded_children_val_p0, [=] __device__(CntType idx) {
                return rel_p0.data[1][expanded_children_ptr_p0[idx]];
        }), this->timing);

        asc_end = thrust::make_counting_iterator(num_children_in_p1);
        timingKernel(
                thrust::transform(thrust::device, asc_begin, asc_end, expanded_children_ptr_p1, [=] __device__(CntType idx) {
                auto rank = idx - children_cnts_in_p1[lbs_p1[idx]];
                auto startPos = rel_p1.offsets[0][col_parent_p1[lbs_p1[idx]]] - rel_p1.trie_offsets[1];
                return rank + startPos;
        }), this->timing);
        timingKernel(
                thrust::transform(thrust::device, asc_begin, asc_end, expanded_children_val_p1, [=] __device__(CntType idx) {
                return rel_p1.data[1][expanded_children_ptr_p1[idx]];
        }), this->timing);
#ifdef FREE_DATA
        CUDA_FREE(children_cnts_in_p0, this->memstat);
        CUDA_FREE(children_cnts_in_p1, this->memstat);
#endif
        /*todo: currently only count the result, no materialization*/
        res_num = ThrustCountMatchPair(
                lbs_p0, expanded_children_val_p0, num_children_in_p0,
                lbs_p1, expanded_children_val_p1, num_children_in_p1,
                this->memstat, this->timing);

        /*the free functions really time-consuming: e.g., 292.38ms, total time: 577ms*/
#ifdef FREE_DATA
        CUDA_FREE(expanded_children_val_p0, this->memstat);
        CUDA_FREE(expanded_children_ptr_p0, this->memstat);
        CUDA_FREE(expanded_children_val_p1, this->memstat);
        CUDA_FREE(expanded_children_ptr_p1, this->memstat);
        CUDA_FREE(lbs_p0, this->memstat);
        CUDA_FREE(lbs_p1, this->memstat);
#endif
        log_info("%s kernel time: %.2f ms.", __FUNCTION__, this->timing->diff_time(gpu_time_idx));
        log_info("%s CPU time: %.2f ms.", __FUNCTION__, t.elapsed()*1000);
    }

    /*todo: add streams for BFS and adjust the return value*/
    CntType evaluate(IndexedTrie<DataType,CntType> *Tries, DataType **res_tuples, CntType *num_res_acc,
                  bool ooc, bool work_sharing, cudaStream_t stream) override {
        log_trace("In BFS-LFTJ evaluate");
        standard_context_t context(false); //ModernGPU context, do not print device info

        /*res stores the indexes of the input tables*/
        CntType **res = nullptr;
        CntType res_num = 0;
        CUDA_MALLOC(&res, sizeof(CntType*)*this->num_tables, this->memstat);

        /*group the input rels according to their first attr
         * todo: need to rewrite when handling arbitrary queries
         * */
        IndexedTrie<DataType,CntType> Trie_0_1, Trie_1_2, Trie_0_2;
        uint32_t idx_0_1, idx_1_2, idx_0_2;
        for(auto i = 0; i < this->num_tables; i++) {
            auto cur_trie = Tries[i];
            if ((cur_trie.attr_list[0] == 0) && (cur_trie.attr_list[1] == 1)) {
                Trie_0_1 = cur_trie;
                idx_0_1 = i;
            }
            if ((cur_trie.attr_list[0] == 1) && (cur_trie.attr_list[1] == 2)) {
                Trie_1_2 = cur_trie;
                idx_1_2 = i;
            }
            if ((cur_trie.attr_list[0] == 0) && (cur_trie.attr_list[1] == 2)) {
                Trie_0_2 = cur_trie;
                idx_0_2 = i;
            }
        }

        /*BFS-LFTJ evaluation*/
        np_join_np(Trie_0_1, idx_0_1, Trie_0_2, idx_0_2, res, res_num, &context);
        if (0 == res_num) return 0;
        p_join_np(Trie_0_1, idx_0_1, Trie_1_2, idx_1_2, {idx_0_2}, res, res_num, &context);
        if (0 == res_num) return 0;
        p_join_p(Trie_1_2, idx_1_2, Trie_0_2, idx_0_2, res, res_num, &context);
        if (0 == res_num) return 0;

        auto total_cnt = num_res_acc[0];
        CUDA_FREE(res, this->memstat);

        return total_cnt;
    }
};