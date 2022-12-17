//
// Created by Bryan on 7/3/2020.
//

#pragma once

#include "config.h"
#include "cuda/primitives.cuh"
#include "helper.h"
#include "timer.h"
#include "types.h"
#include "cuda/GCQueue.cuh"
#include "IndexedTrie.cuh"
#include "LFTJ_base.h"
#include "CBarrier.cuh"

#include <moderngpu/kernel_segsort.hxx>
#include <moderngpu/kernel_load_balance.hxx>
#include <cooperative_groups.h>

#define SMALL_BUC  (512)
#define BUC_THRESHOLD  (1024) //todo: need to revise

/*todo: this is the max value, do not rely on this value, e.g, set to 3 causes errors with triangle counting*/
#define MAX_REL_PER_LEVEL (2)   //each level has at most 2 tables joined (triangle), 3 for 4-clique
#define MAX_NUM_ARR     (6)    //max number of arrays

/*query specific macro*/
#define MAX_NUM_THREADS_PER_BLOCK   (256)
#define MIN_BLOCKS_PERS_SM          (8)

using namespace std;
using namespace cooperative_groups;

//#define ALFTJ_DEBUG

/* Each thread processes one item in input_array[smallest_row]
 * num_output is set to 0 if no match is found
 * */
template<typename DataType, typename CntType>
__device__
void warp_intersection(
        DataType **input_array, uint32_t rows,
        CntType *starts, CntType *ends,
        CntType output_indexes[MAX_REL_PER_LEVEL][WARP_SIZE],
        CntType *valid_output_idx,
        CntType &end_fake,  //must be reference or pointer for shared mem object
        CntType &num_output,
        uint32_t lane)
{
    /*get the index of the smallest array*/
    uint32_t idx_smallest_arr = 0;
    if (0 == lane) {
        CntType smallest_cnt = ends[0] - starts[0];
        for(auto i = 1; i < rows; i++) {
            if (ends[i] - starts[i] < smallest_cnt) {
                smallest_cnt = ends[i] - starts[i];
                idx_smallest_arr = (uint32_t)i;
            }
        }
        end_fake = ends[idx_smallest_arr];
        num_output = 0;
    }
    idx_smallest_arr = __shfl_sync(0xffffffff, idx_smallest_arr, 0); //pass to other lanes, sync necessary for the following coalesed_threads();
    CntType *l_loop_row = output_indexes[idx_smallest_arr];

    /*loop over the smallest array*/
    for(auto idx = starts[idx_smallest_arr] + lane; idx < end_fake; idx += WARP_SIZE) {
        auto group_A = coalesced_threads();
        auto val_comp = input_array[idx_smallest_arr][idx];
        l_loop_row[lane] = idx;         //write the loop row indexes
        bool has_match = true;
        for(auto i = 0; i < rows; i++) {
            if (i == idx_smallest_arr) {
                if (0 == lane) starts[i] += WARP_SIZE;
                group_A.sync();
                continue;
            }
            /*need to write the output_indexes even when it does not have a match*/
            auto has_res = dev_lower_bound<DataType,DataType,DataType>(
                    val_comp, input_array[i],
                    starts[i], ends[i],
                    output_indexes[i][lane]);
            /*last thread going into the loop updates the starts*/
            if (group_A.thread_rank() == group_A.size()-1) {
                if (has_res) starts[i] = output_indexes[i][lane]+1;
                else         starts[i] = output_indexes[i][lane];
            }
            if (!has_res) has_match = false;
        }
        group_A.sync();

        if (has_match) {
            auto group_B = coalesced_threads();
            auto rank = group_B.thread_rank();
            valid_output_idx[rank] = lane;

            if (0 == rank) {
                end_fake = 0;
                num_output = group_B.size();
            }
        }
        group_A.sync();
    }
    __syncwarp();
}

/*
 * prerequisite:
 *  end1 - start1 <= end2 - start2, i.e., input_array1 is the shortest array
 * */
template<typename DataType, typename CntType, typename NumOutputType>
__device__
bool warp_intersection_binary(
        DataType *input_array1, DataType *input_array2,
        CntType &start1, CntType end1,
        CntType &start2, CntType end2,
        CntType output_indexes[MAX_REL_PER_LEVEL][WARP_SIZE/2],
//        CntType *valid_output_idx,
        CntType &end_fake,  //must be reference or pointer for shared mem object
        NumOutputType &num_output,
        uint32_t lane,
        int comp_off)       //indicating which one (0th or 1st) is the smaller array
{
    CntType temp;
    if (0 == lane) {
        end_fake = end1;
        num_output = 0;
    }
    __syncwarp();

    /*loop over the smallest array*/
    for(auto idx = start1 + lane; idx < end_fake; idx += WARP_SIZE) {
        auto group_A = coalesced_threads();
        auto val_comp = input_array1[idx];
        auto res = dev_lower_bound_galloping<DataType, DataType, CntType>(
                val_comp, input_array2, start2, end2, temp); //temp can be equal to end2
        group_A.sync(); //this sync is important to make the following group_B correct

        if (res) {
            auto group_B = coalesced_threads();
            auto rank = group_B.thread_rank();
            output_indexes[1-comp_off][rank] = temp;
            output_indexes[comp_off][rank] = idx;
            if (0 == rank) {
                end_fake = 0;
                num_output = group_B.size() - 1;
            }
        }
        group_A.sync();

        /*update start1 and start2*/
        if ((0 == end_fake) && (group_A.thread_rank() == group_A.size()-1)) {
            if (res) {
                start1 = idx+1;
                start2 = temp+1;
            }
            else {
                start1 = idx;
                start2 = temp;
            }
        }
        group_A.sync();
    }
    __syncwarp();
    return (end_fake != end1);
}


/*------the innermost count-only intersection device functions------*/
//template<typename DataType, typename CntType>
//__device__
//CntType warp_intersection_count_only(
//        DataType **input_array, uint32_t rows,
//        CntType *starts, CntType *ends,
//        uint32_t lane) {
//    CntType dummy, l_cnt = 0;
//    uint32_t idx_smallest_arr = 0;
//    if (0 == lane) {
//        CntType smallest_cnt = ends[0] - starts[0];
//        for(auto i = 1; i < rows; i++) {
//            if (ends[i] - starts[i] < smallest_cnt) {
//                smallest_cnt = ends[i] - starts[i];
//                idx_smallest_arr = (uint32_t)i;
//            }
//        }
//    }
//    idx_smallest_arr = __shfl_sync(0xffffffff, idx_smallest_arr, 0); //pass to other lanes
//    /*loop over the smallest array*/
//    for(auto idx = starts[idx_smallest_arr] + lane; idx < ends[idx_smallest_arr]; idx += WARP_SIZE) {
//        auto val_comp = input_array[idx_smallest_arr][idx];
//        bool has_match = true;
//        for(auto i = 0; i < rows; i++) {
//            if (i == idx_smallest_arr) continue;
//            auto has_res = dev_lower_bound<DataType,DataType,CntType>(
//                    val_comp, input_array[i], starts[i], ends[i], dummy);
//            if (!has_res)  {
//                has_match = false;
//                break;
//            }
//        }
//        if (has_match) l_cnt++;
//    }
//    __syncwarp();
//    return l_cnt;
//}

/*
 * Accelerated LFTJ (ALFTJ), with DFS scheme
 * */
template <typename DataType, typename CntType, bool WS=false>
__global__ __launch_bounds__(MAX_NUM_THREADS_PER_BLOCK, MIN_BLOCKS_PERS_SM)
void  ALFTJ_count(
        IndexedTrie<DataType, CntType> *Tries,
        uint32_t num_tables,
        uint32_t num_attrs,         //number of attributes
        AttrDataMapping<DataType, CntType, uint32_t> *ad_map,

        uint32_t smallest_rel_1st,      //loop over this rel on attr 0
        CntType smallest_rel_card_1st, //the cardinality of this rel

        uint32_t *first_se_idx_per_table, //the index of first attribute in start and end
        uint32_t *scanned_num_rels_each_attr, //the index of first array in start and end (in terms of attribute)

        CntType *g_probe_iterator,
        CntType *countGlobal,
        GCQueue<CntType, CntType> *cq,  //concurrent task queue
        LFTJTaskBook<DataType, CntType, CntType> *tb, //task book
        CBarrier *br)
{
    __shared__ CntType iterators[WARPS_PER_BLOCK][MAX_NUM_ATTRS][MAX_REL_PER_LEVEL][WARP_SIZE/2];//last attr do not need iterators
    __shared__ CntType probe_iterator[WARPS_PER_BLOCK];
    __shared__ char msIdx[WARPS_PER_BLOCK][MAX_NUM_ATTRS];
    __shared__ CntType localCounter;

    /*specifing the search range in each Trie array to be [arr_head, arr_tail), range_start will change
     * the current valid range is [range_start,arr_tail)*/
    __shared__ CntType range_start[WARPS_PER_BLOCK][MAX_NUM_ARR];
    __shared__ CntType arr_head[WARPS_PER_BLOCK][MAX_NUM_ARR];
    __shared__ CntType arr_tail[WARPS_PER_BLOCK][MAX_NUM_ARR];

    __shared__ CntType s_scanned_rels[MAX_NUM_ATTRS+1];
    __shared__ CntType s_first_se_idx_per_table[MAX_NUM_TABLES];
    __shared__ CntType cur_iterators[WARPS_PER_BLOCK][MAX_NUM_ATTRS*MAX_REL_PER_LEVEL];
    __shared__ bool sharing[WARPS_PER_BLOCK];
    __shared__ CntType auc[WARPS_PER_BLOCK];
    __shared__ bool triggerWS[WARPS_PER_BLOCK];

    auto tid = (CntType)threadIdx.x;
    auto lwarpId = tid >> WARP_BITS;
    auto lane = tid & WARP_MASK;
    char cur_attr = 0, start_attr = 0;
    bool found;
    CntType privateCounter = 0;

    /*init the data structures*/
    if (0 == tid) localCounter = 0;
    if (tid < num_attrs+1)  s_scanned_rels[tid] = scanned_num_rels_each_attr[tid];
    if (tid < num_tables) s_first_se_idx_per_table[tid] = first_se_idx_per_table[tid];
    if (tid < WARPS_PER_BLOCK) {
        sharing[tid] = false;
        triggerWS[tid] = false;
    }
    __syncthreads();
    if (0 == lane) {
        msIdx[lwarpId][0] = 0;
        range_start[lwarpId][s_scanned_rels[start_attr]] = 0;
        arr_tail[lwarpId][s_scanned_rels[start_attr]] = 0;
    }
    __syncthreads();

    while (true) {
        if (0 == msIdx[lwarpId][cur_attr]) { //get the next match
            if ((cur_attr == start_attr) &&
                (range_start[lwarpId][s_scanned_rels[start_attr]] >= arr_tail[lwarpId][s_scanned_rels[start_attr]])) { //need to get a new root item
                if (WS && triggerWS[lwarpId]) //dequeue when triggerWS is set
                    SIN_L(sharing[lwarpId]=cq->dequeue(auc[lwarpId]));
                if (!sharing[lwarpId]) { //fetch a new probe item
                    /*each warp claims WARP_SIZE items at a time*/
                    SIN_L(probe_iterator[lwarpId] = atomicAdd(g_probe_iterator, WARP_SIZE));
                    if (probe_iterator[lwarpId] >= smallest_rel_card_1st) { //flush and return
                        if (!WS) goto TAG_LOOP_END;
                        SIN_L(
                                sharing[lwarpId] = cq->dequeue(auc[lwarpId]);
                                triggerWS[lwarpId] = true);
                        while (true) {
                            if (sharing[lwarpId]) break; //go back to work
                            SIN_L(br->setActive(false));
                            while (!sharing[lwarpId]) {
                                if (0 == lane) found = cq->isEmpty();
                                found = __shfl_sync(0xffffffff, found, 0);
                                if (!found) {
                                    SIN_L(
                                            br->setActive(true);
                                            sharing[lwarpId] = cq->dequeue(auc[lwarpId]));
                                    if (!sharing[lwarpId]) SIN_L(br->setActive(false));
                                }
                                /*all warps reach this barrier, exit*/
                                if (br->isTerminated())    goto TAG_LOOP_END;
                            }
                        }
                    }
                }
                if (sharing[lwarpId]) { //branch 1: process the newly generated sharing task
                    /* start_attr should be 1 less than cur_attr */
                    cur_attr = start_attr = tb->cur_attr[auc[lwarpId]];

                    /* Update the start and end of next attr (attr idx >= cur_attr)
                     * of the tables having attrs that are recovered in above.
                     * Need to check cur_attr * MAX_REL_PER_LEVEL rels */
                    auto tb_offset = auc[lwarpId]*MAX_REL_PER_LEVEL*(tb->res_len);
                    for(auto i = lane; i < MAX_REL_PER_LEVEL*cur_attr; i+= WARP_SIZE) {
                        auto l_attr = i/MAX_REL_PER_LEVEL;
                        auto l_rel_idx = i%MAX_REL_PER_LEVEL; //the idx for iterating the relations

                        /*recover the iterators*/
                        iterators[lwarpId][l_attr][l_rel_idx][0] = tb->iterators[tb_offset+l_attr*MAX_REL_PER_LEVEL+l_rel_idx];
                        auto l_rel = ad_map[l_attr].rel[l_rel_idx]; //the relation
                        auto l_attr_in_rel = ad_map[l_attr].levels[l_rel_idx];
                        if (l_attr_in_rel < Tries[l_rel].num_attrs - 1) { //when we have next level
                            auto next_se_idx = ad_map[l_attr].next_se_idx[l_rel_idx];
                            if(next_se_idx > s_scanned_rels[cur_attr]) { //only update the unprocessed attrs
                                auto l_offset = iterators[lwarpId][l_attr][l_rel_idx][0];
                                range_start[lwarpId][next_se_idx] = Tries[l_rel].offsets[l_attr_in_rel][l_offset] - Tries[l_rel].trie_offsets[l_attr_in_rel+1];
                                arr_head[lwarpId][next_se_idx] = Tries[l_rel].offsets[l_attr_in_rel][l_offset] - Tries[l_rel].trie_offsets[l_attr_in_rel+1];
                                arr_tail[lwarpId][next_se_idx] = Tries[l_rel].offsets[l_attr_in_rel][l_offset+1] - Tries[l_rel].trie_offsets[l_attr_in_rel+1];
                            }
                        }
                    }
                    if (0 == lane) { //recover the start and end of cur_attr
                        range_start[lwarpId][s_scanned_rels[cur_attr]] = tb->rel_0_start[auc[lwarpId]];
                        arr_head[lwarpId][s_scanned_rels[cur_attr]] = tb->rel_0_start[auc[lwarpId]];
                        arr_tail[lwarpId][s_scanned_rels[cur_attr]] = tb->rel_0_end[auc[lwarpId]];

                        range_start[lwarpId][s_scanned_rels[cur_attr]+1] = tb->rel_1_start[auc[lwarpId]];
                        arr_head[lwarpId][s_scanned_rels[cur_attr]+1] = tb->rel_1_start[auc[lwarpId]];
                        arr_tail[lwarpId][s_scanned_rels[cur_attr]+1] = tb->rel_1_end[auc[lwarpId]];
                    }
                }
                else { //branch 2: init data for original tasks
                    cur_attr = start_attr = 0; //reset cur_attr and start_attr

                    /*init the start and end of the first attr of each relation*/
                    if (lane < num_tables) {
                        if (lane == smallest_rel_1st) {
                            range_start[lwarpId][s_first_se_idx_per_table[lane]] = probe_iterator[lwarpId];
                            arr_head[lwarpId][s_first_se_idx_per_table[lane]] = probe_iterator[lwarpId];
                            arr_tail[lwarpId][s_first_se_idx_per_table[lane]] =
                                    (probe_iterator[lwarpId] + WARP_SIZE > smallest_rel_card_1st) ?
                                    smallest_rel_card_1st : probe_iterator[lwarpId] + WARP_SIZE;
                        }
                        else {
                            range_start[lwarpId][s_first_se_idx_per_table[lane]] = 0;
                            arr_head[lwarpId][s_first_se_idx_per_table[lane]] = 0;
                            arr_tail[lwarpId][s_first_se_idx_per_table[lane]] = Tries[lane].data_len[0];
                        }
                    }
                }
                __syncwarp();
            }

            /*coming to the last attribute*/
            if (cur_attr == num_attrs - 1) {
                int comp_off = 0; //indicating which input is smaller
                if (s_scanned_rels[cur_attr+1] - s_scanned_rels[cur_attr] < 2) {
                    if (0 == lane) {
                        privateCounter += (arr_tail[lwarpId][s_scanned_rels[cur_attr]] - range_start[lwarpId][s_scanned_rels[cur_attr]]);
                    }
                }
                else {
                    if (arr_tail[lwarpId][s_scanned_rels[cur_attr]] - range_start[lwarpId][s_scanned_rels[cur_attr]] >
                        arr_tail[lwarpId][s_scanned_rels[cur_attr]+1] - range_start[lwarpId][s_scanned_rels[cur_attr]+1]) {
                        comp_off = 1;
                    }
#ifdef ALFTJ_DEBUG
                    if(0 == lane) {
                        printf("Last: cur_attr:%d, Array 0[%lu,%lu), Array 1[%lu,%lu)\n",
                               cur_attr,
                               range_start[lwarpId][s_scanned_rels[cur_attr]+comp_off],
                               arr_tail[lwarpId][s_scanned_rels[cur_attr]+comp_off],
                               range_start[lwarpId][s_scanned_rels[cur_attr]+1-comp_off],
                               arr_tail[lwarpId][s_scanned_rels[cur_attr]+1-comp_off]);
                        printf("Array 0: ");
                        for(auto x = range_start[lwarpId][s_scanned_rels[cur_attr]+comp_off];
                            x < arr_tail[lwarpId][s_scanned_rels[cur_attr]+comp_off];
                            x++) {
                            printf("%d, ", ad_map[cur_attr].data[comp_off][x]);
                        }
                        printf("\n");
                        printf("Array 1: ");
                        for(auto x = range_start[lwarpId][s_scanned_rels[cur_attr]+1-comp_off];
                            x < arr_tail[lwarpId][s_scanned_rels[cur_attr]+1-comp_off];
                            x++) {
                            printf("%d, ", ad_map[cur_attr].data[1-comp_off][x]);
                        }
                        printf("\n");
                    }
                    __syncwarp();
#endif

                    CntType start2 = range_start[lwarpId][s_scanned_rels[cur_attr]+1-comp_off];
                    for(auto idx = range_start[lwarpId][s_scanned_rels[cur_attr]+comp_off] + lane;
                        idx < arr_tail[lwarpId][s_scanned_rels[cur_attr]+comp_off];
                        idx += WARP_SIZE) {
                        auto val_comp = ad_map[cur_attr].data[comp_off][idx];
                        auto res = dev_lower_bound_galloping<DataType, DataType, CntType>(
                                val_comp, ad_map[cur_attr].data[1-comp_off],
                                start2,
                                arr_tail[lwarpId][s_scanned_rels[cur_attr]+1-comp_off],
                                start2);
                        if (res) {
                            privateCounter++;
                        }
                    }

//                    auto temp_res = warp_intersection_count_only_binary(
//                            ad_map[cur_attr].data[comp_off], ad_map[cur_attr].data[1-comp_off],
//                            range_start[lwarpId][s_scanned_rels[cur_attr]+comp_off],
//                            arr_tail[lwarpId][s_scanned_rels[cur_attr]+comp_off],
//                            range_start[lwarpId][s_scanned_rels[cur_attr]+1-comp_off],
//                            arr_tail[lwarpId][s_scanned_rels[cur_attr]+1-comp_off],
//                            lane);
//                    privateCounter += temp_res;
                }
                __syncwarp();
                cur_attr--;
                continue;
            }
            else {
                /*only a single array in this attr, no need to intersect*/
                if (s_scanned_rels[cur_attr+1] - s_scanned_rels[cur_attr] < 2) {
                    if (0 == lane) {
                        auc[lwarpId] = false;
                    }
                    __syncwarp();
                    if (range_start[lwarpId][s_scanned_rels[cur_attr]]+lane < arr_tail[lwarpId][s_scanned_rels[cur_attr]]) {
                        auto active_group = coalesced_threads();
                        iterators[lwarpId][cur_attr][0][lane] = range_start[lwarpId][s_scanned_rels[cur_attr]]+lane;
                        if (0 == active_group.thread_rank()) {
                            auc[lwarpId] = true;
                            msIdx[lwarpId][cur_attr] = (char)(active_group.size()-1);
                        }
                    }
                    __syncwarp();
                    found = auc[lwarpId];
                    if (0 == lane) {
                        if (found) range_start[lwarpId][s_scanned_rels[cur_attr]] += WARP_SIZE; //update range_start
                        else       range_start[lwarpId][s_scanned_rels[cur_attr]] = arr_head[lwarpId][s_scanned_rels[cur_attr]]; //reset range_start
                    }
                    __syncwarp();
                }
                else {
                    int small_idx_offset = 0; //rel0 is smallest table by default
                    auto *l_range_start = range_start[lwarpId];
                    auto *l_range_end = arr_tail[lwarpId];
                    auto l_small_range = l_range_end[s_scanned_rels[cur_attr]] - l_range_start[s_scanned_rels[cur_attr]];

                    if (l_range_end[s_scanned_rels[cur_attr]+1] - l_range_start[s_scanned_rels[cur_attr]+1] < l_small_range) {
                        small_idx_offset = 1; //rel1 now becomes the small table
                        l_small_range = l_range_end[s_scanned_rels[cur_attr]+1] - l_range_start[s_scanned_rels[cur_attr]+1];
                    }

                    /*need to split the intersection range*/
                    if (WS && (l_small_range > BUC_THRESHOLD)) {
                        SIN_L(triggerWS[lwarpId]= true); //enable work-sharing for this warp
                        if (lane < cur_attr) { //gather the current intermediate results into a 1-D array
                            for(auto i = 0; i < MAX_REL_PER_LEVEL; i++) {
                                auto cur_msx = msIdx[lwarpId][lane];
                                cur_iterators[lwarpId][lane*MAX_REL_PER_LEVEL+i] = iterators[lwarpId][lane][i][cur_msx];
                            }
                        }
                        __syncwarp();
                        for(auto p_small_range_start = l_range_start[s_scanned_rels[cur_attr]+small_idx_offset] + lane * BUC_THRESHOLD;
                            p_small_range_start < l_range_end[s_scanned_rels[cur_attr]+small_idx_offset];
                            p_small_range_start += BUC_THRESHOLD * WARP_SIZE) {
                            CntType p_small_range_end = (p_small_range_start + BUC_THRESHOLD >= l_range_end[s_scanned_rels[cur_attr]+small_idx_offset])?
                                                        l_range_end[s_scanned_rels[cur_attr]+small_idx_offset] :
                                                        p_small_range_start + BUC_THRESHOLD;
                            CntType p_large_range_start, p_large_range_end;
                            dev_lower_bound_galloping(ad_map[cur_attr].data[small_idx_offset][p_small_range_start],
                                                      ad_map[cur_attr].data[1-small_idx_offset],
                                                      l_range_start[s_scanned_rels[cur_attr]+1-small_idx_offset],
                                                      l_range_end[s_scanned_rels[cur_attr]+1-small_idx_offset],
                                                      p_large_range_start);
                            dev_upper_bound_galloping(ad_map[cur_attr].data[small_idx_offset][p_small_range_end-1],
                                                      ad_map[cur_attr].data[1-small_idx_offset],
                                                      l_range_start[s_scanned_rels[cur_attr]+1-small_idx_offset],
                                                      l_range_end[s_scanned_rels[cur_attr]+1-small_idx_offset],
                                                      p_large_range_end);

                            auto task_id = tb->push_task(
                                    p_small_range_start, p_small_range_end,
                                    p_large_range_start, p_large_range_end,
                                    cur_iterators[lwarpId], nullptr, cur_attr);
                            cq->enqueue(task_id); //push the task to the concurrent queue
                        }
                        __syncwarp();
                        found = false; //skip the task
                    }
                    else {
#ifdef ALFTJ_DEBUG
                        if(0 == lane) {
                            printf("cur_attr:%d, Array 0[%lu,%lu), Array 1[%lu,%lu)\n",
                                   cur_attr,
                                   range_start[lwarpId][s_scanned_rels[cur_attr]+small_idx_offset],
                                   arr_tail[lwarpId][s_scanned_rels[cur_attr]+small_idx_offset],
                                   range_start[lwarpId][s_scanned_rels[cur_attr]+1-small_idx_offset],
                                   arr_tail[lwarpId][s_scanned_rels[cur_attr]+1-small_idx_offset]);
                            printf("Array 0: ");
                            for(auto    x = range_start[lwarpId][s_scanned_rels[cur_attr]+small_idx_offset];
                                x < arr_tail[lwarpId][s_scanned_rels[cur_attr]+small_idx_offset];
                                x++) {
                                printf("%d, ", ad_map[cur_attr].data[small_idx_offset][x]);
                            }
                            printf("\n");
                            printf("Array 1: ");
                            for(auto    x = range_start[lwarpId][s_scanned_rels[cur_attr]+1-small_idx_offset];
                                x < arr_tail[lwarpId][s_scanned_rels[cur_attr]+1-small_idx_offset];
                                x++) {
                                printf("%d, ", ad_map[cur_attr].data[1-small_idx_offset][x]);
                            }
                            printf("\n");
                        }
                        __syncwarp();
#endif
                        found = warp_intersection_binary( //do the normal intersection
                                ad_map[cur_attr].data[small_idx_offset], ad_map[cur_attr].data[1-small_idx_offset],
                                range_start[lwarpId][s_scanned_rels[cur_attr]+small_idx_offset],
                                arr_tail[lwarpId][s_scanned_rels[cur_attr]+small_idx_offset],
                                range_start[lwarpId][s_scanned_rels[cur_attr]+1-small_idx_offset],
                                arr_tail[lwarpId][s_scanned_rels[cur_attr]+1-small_idx_offset],
                                iterators[lwarpId][cur_attr],
                                auc[lwarpId],
                                msIdx[lwarpId][cur_attr],
                                lane, small_idx_offset);
                        if (!found) { //reset range_start
                            if (0 == lane) {
                                range_start[lwarpId][s_scanned_rels[cur_attr]+small_idx_offset] = arr_head[lwarpId][s_scanned_rels[cur_attr]+small_idx_offset];
                                range_start[lwarpId][s_scanned_rels[cur_attr]+1-small_idx_offset] = arr_head[lwarpId][s_scanned_rels[cur_attr]+1-small_idx_offset];
                            }
                            __syncwarp();
                        }
#ifdef ALFTJ_DEBUG
                        if(0 == lane) {
                            printf("After intersection, %d res\n", (int)msIdx[lwarpId][cur_attr]+1);
                            printf("msIdx[%d]=%d\n", cur_attr, msIdx[lwarpId][cur_attr]);
                            printf("Array 0 res: ");
                            for (auto i = 0; i < msIdx[lwarpId][cur_attr]+1; i++) {
                                printf("%lu, ", iterators[lwarpId][cur_attr][0][i]);
                            }
                            printf("\n");

                            printf("Array 1 res: ");
                            for (auto i = 0; i < msIdx[lwarpId][cur_attr]+1; i++) {
                                printf("%lu, ", iterators[lwarpId][cur_attr][1][i]);
                            }
                            printf("\n");
                        }
#endif
                    }
                }
                /*if the intersection returns no results, backtrack to the last level*/
                if (!found) {
                    if (cur_attr > start_attr) { //backtrack to the last attribute, todo: here all threads in a warp do this
                        cur_attr--;
                    }
                    else if (0 == lane) { //reach the first attr, set values for getting a new probe item
                        range_start[lwarpId][s_scanned_rels[start_attr]] = arr_tail[lwarpId][s_scanned_rels[start_attr]];
                    }
                    __syncwarp();
                    continue;
                }
            }
            __syncwarp();
        }
        else {
            SIN_L(msIdx[lwarpId][cur_attr]--);

            /* reset all but not the smallest_rel_1st when cur_attr = start_attr_idx */
            if ((start_attr == cur_attr) && (lane < num_tables) && (lane != smallest_rel_1st)) {
                range_start[lwarpId][s_first_se_idx_per_table[lane]] = 0;
                arr_tail[lwarpId][s_first_se_idx_per_table[lane]] = Tries[lane].data_len[0];
            }
        }
        __syncwarp();

        /*update the start and end of next level of the tables appeared in this level*/
//        if (lane < MAX_REL_PER_LEVEL) {
        if (lane < ad_map[cur_attr].num) {
            auto cur_rel = ad_map[cur_attr].rel[lane];
            auto cur_level_in_rel = ad_map[cur_attr].levels[lane];
            if (cur_level_in_rel < Tries[cur_rel].num_attrs - 1) { //when we have next level
                auto cur_msIdx = msIdx[lwarpId][cur_attr];
                auto cur_offset = iterators[lwarpId][cur_attr][lane][cur_msIdx];
                auto next_se_idx = ad_map[cur_attr].next_se_idx[lane];
                range_start[lwarpId][next_se_idx] = Tries[cur_rel].offsets[cur_level_in_rel][cur_offset] - Tries[cur_rel].trie_offsets[cur_level_in_rel+1];
                arr_tail[lwarpId][next_se_idx] = Tries[cur_rel].offsets[cur_level_in_rel][cur_offset+1] - Tries[cur_rel].trie_offsets[cur_level_in_rel+1];
                arr_head[lwarpId][next_se_idx] = Tries[cur_rel].offsets[cur_level_in_rel][cur_offset] - Tries[cur_rel].trie_offsets[cur_level_in_rel+1];
            }
        }
        __syncwarp();
        cur_attr++; //advance to the next attr
        SIN_L(msIdx[lwarpId][cur_attr] = 0); //init the msIdx for the next attr
    }

    TAG_LOOP_END:
    __syncwarp();

    WARP_REDUCE(privateCounter);
    if (lane == 0) atomicAdd(&localCounter, privateCounter);

    __syncthreads();
    if (0 == tid) atomicAdd(countGlobal, localCounter);
}

/*
 * Accelerated LFTJ (ALFTJ), with DFS scheme
 * */
template <typename DataType, typename CntType, bool WS=false>
__global__ __launch_bounds__(MAX_NUM_THREADS_PER_BLOCK, MIN_BLOCKS_PERS_SM)
void ALFTJ_write(
        IndexedTrie<DataType, CntType> *Tries,
        uint32_t num_tables,
        uint32_t num_attrs,         //number of attributes
        AttrDataMapping<DataType, CntType, uint32_t> *ad_map,

        uint32_t smallest_rel_1st,      //loop over this rel on attr 0
        CntType smallest_rel_card_1st, //the cardinality of this rel

        uint32_t *first_se_idx_per_table, //the index of first attribute in start and end
        uint32_t *scanned_num_rels_each_attr, //the index of first array in start and end (in terms of attribute)

        CntType *g_probe_iterator,
        CntType *countGlobal,
        DataType **res_tuples,

        GCQueue<CntType, CntType> *cq,  //concurrent task queue
        LFTJTaskBook<DataType, CntType, CntType> *tb, //task book
        CBarrier *br)
{
    __shared__ CntType iterators[WARPS_PER_BLOCK][MAX_NUM_ATTRS][MAX_REL_PER_LEVEL][WARP_SIZE/2];//last attr do not need iterators
    __shared__ CntType probe_iterator[WARPS_PER_BLOCK];
    __shared__ char msIdx[WARPS_PER_BLOCK][MAX_NUM_ATTRS];
    __shared__ DataType iRes[WARPS_PER_BLOCK][MAX_NUM_ATTRS];

    /*specifing the search range in each Trie array to be [arr_head, arr_tail), range_start will change
     * the current valid range is [range_start,arr_tail)*/
    __shared__ CntType range_start[WARPS_PER_BLOCK][MAX_NUM_ARR];
    __shared__ CntType arr_head[WARPS_PER_BLOCK][MAX_NUM_ARR];
    __shared__ CntType arr_tail[WARPS_PER_BLOCK][MAX_NUM_ARR];

    __shared__ CntType s_scanned_rels[MAX_NUM_ATTRS+1];
    __shared__ CntType s_first_se_idx_per_table[MAX_NUM_TABLES];
    __shared__ CntType cur_iterators[WARPS_PER_BLOCK][MAX_NUM_ATTRS*MAX_REL_PER_LEVEL];
    __shared__ bool sharing[WARPS_PER_BLOCK];
    __shared__ CntType auc[WARPS_PER_BLOCK];
    __shared__ bool triggerWS[WARPS_PER_BLOCK];

    auto tid = (CntType)threadIdx.x;
    auto lwarpId = tid >> WARP_BITS;
    auto lane = tid & WARP_MASK;
    char cur_attr = 0, start_attr = 0;
    bool found;

    /*init the data structures*/
    if (tid < num_attrs+1)  s_scanned_rels[tid] = scanned_num_rels_each_attr[tid];
    if (tid < num_tables) s_first_se_idx_per_table[tid] = first_se_idx_per_table[tid];
    if (tid < WARPS_PER_BLOCK) {
        sharing[tid] = false;
        triggerWS[tid] = false;
    }
    __syncthreads();
    if (0 == lane) {
        msIdx[lwarpId][0] = 0;
        range_start[lwarpId][s_scanned_rels[start_attr]] = 0;
        arr_tail[lwarpId][s_scanned_rels[start_attr]] = 0;
    }
    __syncthreads();

    while (true) {
        if (0 == msIdx[lwarpId][cur_attr]) { //get the next match
            if ((cur_attr == start_attr) &&
                (range_start[lwarpId][s_scanned_rels[start_attr]] >= arr_tail[lwarpId][s_scanned_rels[start_attr]])) { //need to get a new root item
                if (WS && triggerWS[lwarpId]) //dequeue when triggerWS is set
                SIN_L(sharing[lwarpId]=cq->dequeue(auc[lwarpId]));
                if (!sharing[lwarpId]) { //fetch a new probe item
                    /*each warp claims WARP_SIZE items at a time*/
                    SIN_L(probe_iterator[lwarpId] = atomicAdd(g_probe_iterator,WARP_SIZE));
                    if (probe_iterator[lwarpId] >= smallest_rel_card_1st) { //flush and return
                        if (!WS) return;
                        SIN_L(
                                sharing[lwarpId] = cq->dequeue(auc[lwarpId]);
                                triggerWS[lwarpId] = true);
                        while (true) {
                            if (sharing[lwarpId]) break; //go back to work
                            SIN_L(br->setActive(false));
                            while (!sharing[lwarpId]) {
                                if (0 == lane) found = cq->isEmpty();
                                found = __shfl_sync(0xffffffff, found, 0);
                                if (!found) {
                                    SIN_L(
                                            br->setActive(true);
                                            sharing[lwarpId] = cq->dequeue(auc[lwarpId]));
                                    if (!sharing[lwarpId]) SIN_L(br->setActive(false));
                                }
                                /*all warps reach this barrier, exit*/
                                if (br->isTerminated())    return;
                            }
                        }
                    }
                }
                if (WS && sharing[lwarpId]) { //branch 1: process the newly generated sharing task
                    /* start_attr should be 1 less than cur_attr */
                    cur_attr = start_attr = tb->cur_attr[auc[lwarpId]];

                    /* Update the start and end of next attr (attr idx >= cur_attr)
                     * of the tables having attrs that are recovered in above.
                     * Need to check cur_attr * MAX_REL_PER_LEVEL rels */
                    auto tb_offset = auc[lwarpId]*MAX_REL_PER_LEVEL*(tb->res_len);
                    for(auto i = lane; i < MAX_REL_PER_LEVEL*cur_attr; i+= WARP_SIZE) {
                        auto l_attr = i/MAX_REL_PER_LEVEL;
                        auto l_rel_idx = i%MAX_REL_PER_LEVEL; //the idx for iterating the relations

                        /*recover the iterators*/
                        iterators[lwarpId][l_attr][l_rel_idx][0] = tb->iterators[tb_offset+l_attr*MAX_REL_PER_LEVEL+l_rel_idx];
                        auto l_rel = ad_map[l_attr].rel[l_rel_idx]; //the relation
                        auto l_attr_in_rel = ad_map[l_attr].levels[l_rel_idx];
                        if (l_attr_in_rel < Tries[l_rel].num_attrs - 1) { //when we have next level
                            auto next_se_idx = ad_map[l_attr].next_se_idx[l_rel_idx];
                            if(next_se_idx > s_scanned_rels[cur_attr]) { //only update the unprocessed attrs
                                auto l_offset = iterators[lwarpId][l_attr][l_rel_idx][0];
                                range_start[lwarpId][next_se_idx] = Tries[l_rel].offsets[l_attr_in_rel][l_offset] - Tries[l_rel].trie_offsets[l_attr_in_rel+1];
                                arr_head[lwarpId][next_se_idx] = Tries[l_rel].offsets[l_attr_in_rel][l_offset] - Tries[l_rel].trie_offsets[l_attr_in_rel+1];
                                arr_tail[lwarpId][next_se_idx] = Tries[l_rel].offsets[l_attr_in_rel][l_offset+1] - Tries[l_rel].trie_offsets[l_attr_in_rel+1];
                            }
                        }
                    }
                    if (0 == lane) { //recover the start and end of cur_attr
                        range_start[lwarpId][s_scanned_rels[cur_attr]] = tb->rel_0_start[auc[lwarpId]];
                        arr_head[lwarpId][s_scanned_rels[cur_attr]] = tb->rel_0_start[auc[lwarpId]];
                        arr_tail[lwarpId][s_scanned_rels[cur_attr]] = tb->rel_0_end[auc[lwarpId]];

                        range_start[lwarpId][s_scanned_rels[cur_attr]+1] = tb->rel_1_start[auc[lwarpId]];
                        arr_head[lwarpId][s_scanned_rels[cur_attr]+1] = tb->rel_1_start[auc[lwarpId]];
                        arr_tail[lwarpId][s_scanned_rels[cur_attr]+1] = tb->rel_1_end[auc[lwarpId]];
                    }
                    tb_offset = auc[lwarpId]*num_attrs;
                    if (lane < num_attrs) { //restore the iRes data
                        iRes[lwarpId][lane] = tb->iRes[tb_offset+lane];
                    }
                }
                else { //branch 2: init data for original tasks
                    cur_attr = start_attr = 0; //reset cur_attr and start_attr

                    /*init the start and end of the first attr of each relation*/
                    if (lane < num_tables) {
                        if (lane == smallest_rel_1st) {
                            range_start[lwarpId][s_first_se_idx_per_table[lane]] = probe_iterator[lwarpId];
                            arr_head[lwarpId][s_first_se_idx_per_table[lane]] = probe_iterator[lwarpId];
                            arr_tail[lwarpId][s_first_se_idx_per_table[lane]] =
                                    (probe_iterator[lwarpId] + WARP_SIZE > smallest_rel_card_1st) ?
                                    smallest_rel_card_1st : probe_iterator[lwarpId] + WARP_SIZE;
                        }
                        else {
                            range_start[lwarpId][s_first_se_idx_per_table[lane]] = 0;
                            arr_head[lwarpId][s_first_se_idx_per_table[lane]] = 0;
                            arr_tail[lwarpId][s_first_se_idx_per_table[lane]] = Tries[lane].data_len[0];
                        }
                    }
                }
                __syncwarp();
            }

            /*coming to the last attribute*/
            if (cur_attr == num_attrs - 1) {
                CntType write_loc;
                int comp_off = 0; //indicating which input is smaller
                if (s_scanned_rels[cur_attr+1] - s_scanned_rels[cur_attr] < 2) {
                    for(auto p = range_start[lwarpId][s_scanned_rels[cur_attr]] + lane;
                        p < arr_tail[lwarpId][s_scanned_rels[cur_attr]];
                        p += WARP_SIZE) {
                        auto active_group = coalesced_threads();
                        if (0 == active_group.thread_rank()) {
                            write_loc = atomicAdd(countGlobal, active_group.size());
                        }
                        write_loc = active_group.shfl(write_loc, 0) + active_group.thread_rank();

                        #pragma unroll
                        for(auto q = 0; q < num_attrs-1; q++) {
                            res_tuples[q][write_loc] = iRes[lwarpId][q];
                        }
                        res_tuples[num_attrs-1][write_loc] = ad_map[cur_attr].data[0][p];
                    }
                }
                else {
                    if (arr_tail[lwarpId][s_scanned_rels[cur_attr]] - range_start[lwarpId][s_scanned_rels[cur_attr]] >
                        arr_tail[lwarpId][s_scanned_rels[cur_attr]+1] - range_start[lwarpId][s_scanned_rels[cur_attr]+1]) {
                        comp_off = 1;
                    }
#ifdef ALFTJ_DEBUG
                    if(0 == lane) {
                        printf("Last: cur_attr:%d, Array 0[%lu,%lu), Array 1[%lu,%lu)\n",
                               cur_attr,
                               range_start[lwarpId][s_scanned_rels[cur_attr]+comp_off],
                               arr_tail[lwarpId][s_scanned_rels[cur_attr]+comp_off],
                               range_start[lwarpId][s_scanned_rels[cur_attr]+1-comp_off],
                               arr_tail[lwarpId][s_scanned_rels[cur_attr]+1-comp_off]);
                        printf("Array 0: ");
                        for(auto x = range_start[lwarpId][s_scanned_rels[cur_attr]+comp_off];
                            x < arr_tail[lwarpId][s_scanned_rels[cur_attr]+comp_off];
                            x++) {
                            printf("%d, ", ad_map[cur_attr].data[comp_off][x]);
                        }
                        printf("\n");
                        printf("Array 1: ");
                        for(auto x = range_start[lwarpId][s_scanned_rels[cur_attr]+1-comp_off];
                            x < arr_tail[lwarpId][s_scanned_rels[cur_attr]+1-comp_off];
                            x++) {
                            printf("%d, ", ad_map[cur_attr].data[1-comp_off][x]);
                        }
                        printf("\n");
                    }
                    __syncwarp();
#endif
                    CntType start2 = range_start[lwarpId][s_scanned_rels[cur_attr]+1-comp_off];
                    for(auto idx = range_start[lwarpId][s_scanned_rels[cur_attr]+comp_off] + lane;
                        idx < arr_tail[lwarpId][s_scanned_rels[cur_attr]+comp_off];
                        idx += WARP_SIZE) {
                        auto val_comp = ad_map[cur_attr].data[comp_off][idx];
                        auto res = dev_lower_bound_galloping<DataType, DataType, CntType>(
                                val_comp, ad_map[cur_attr].data[1-comp_off],
                                start2,
                                arr_tail[lwarpId][s_scanned_rels[cur_attr]+1-comp_off],
                                start2);
                        if (res) {
                            auto active_group = coalesced_threads();
                            if (0 == active_group.thread_rank()) {
                                write_loc = atomicAdd(countGlobal, active_group.size());
                            }
                            write_loc = active_group.shfl(write_loc, 0) + active_group.thread_rank();

                            #pragma unroll
                            for(auto p = 0; p < num_attrs-1; p++) {
                                res_tuples[p][write_loc] = iRes[lwarpId][p];
                            }
                            res_tuples[num_attrs-1][write_loc] = val_comp;
                        }
                    }
                }
                __syncwarp();
                cur_attr--;
                continue;
            }
            else {
                /*only a single array in this attr, no need to intersect*/
                if (s_scanned_rels[cur_attr+1] - s_scanned_rels[cur_attr] < 2) {
                    found = (arr_tail[lwarpId][s_scanned_rels[cur_attr]] > range_start[lwarpId][s_scanned_rels[cur_attr]]);
                    if (range_start[lwarpId][s_scanned_rels[cur_attr]]+lane < arr_tail[lwarpId][s_scanned_rels[cur_attr]]) {
                        auto active_group = coalesced_threads();
                        iterators[lwarpId][cur_attr][0][lane] = range_start[lwarpId][s_scanned_rels[cur_attr]]+lane;
                        if (0 == lane) {
                            msIdx[lwarpId][cur_attr] = (char)(active_group.size()-1);
                        }
                    }
                    __syncwarp();

                    if (0 == lane) {
                        if (found) { //update range_start and iRes
                            range_start[lwarpId][s_scanned_rels[cur_attr]] += WARP_SIZE;
                            auto cur_iterator = iterators[lwarpId][cur_attr][0][msIdx[lwarpId][cur_attr]];
                            iRes[lwarpId][cur_attr] = ad_map[cur_attr].data[0][cur_iterator];
                        }
                        else range_start[lwarpId][s_scanned_rels[cur_attr]] = arr_head[lwarpId][s_scanned_rels[cur_attr]]; //reset range_start
                    }
                    __syncwarp();
                }
                else {
                    int small_idx_offset = 0; //rel0 is smallest table by default
                    auto *l_range_start = range_start[lwarpId];
                    auto *l_range_end = arr_tail[lwarpId];
                    auto l_small_range = l_range_end[s_scanned_rels[cur_attr]] - l_range_start[s_scanned_rels[cur_attr]];

                    if (l_range_end[s_scanned_rels[cur_attr]+1] - l_range_start[s_scanned_rels[cur_attr]+1] < l_small_range) {
                        small_idx_offset = 1; //rel1 now becomes the small table
                        l_small_range = l_range_end[s_scanned_rels[cur_attr]+1] - l_range_start[s_scanned_rels[cur_attr]+1];
                    }

                    /*need to split the intersection range*/
                    if (WS && (l_small_range > BUC_THRESHOLD)) {
                        SIN_L(triggerWS[lwarpId]= true); //enable work-sharing for this warp
                        if (lane < cur_attr) { //gather the current intermediate results into a 1-D array
                            for(auto i = 0; i < MAX_REL_PER_LEVEL; i++) {
                                auto cur_msx = msIdx[lwarpId][lane];
                                cur_iterators[lwarpId][lane*MAX_REL_PER_LEVEL+i] = iterators[lwarpId][lane][i][cur_msx];
                            }
                        }
                        __syncwarp();
                        for(auto p_small_range_start = l_range_start[s_scanned_rels[cur_attr]+small_idx_offset] + lane * BUC_THRESHOLD;
                            p_small_range_start < l_range_end[s_scanned_rels[cur_attr]+small_idx_offset];
                            p_small_range_start += BUC_THRESHOLD * WARP_SIZE) {
                            CntType p_small_range_end = (p_small_range_start + BUC_THRESHOLD >= l_range_end[s_scanned_rels[cur_attr]+small_idx_offset])?
                                                        l_range_end[s_scanned_rels[cur_attr]+small_idx_offset] :
                                                        p_small_range_start + BUC_THRESHOLD;
                            CntType p_large_range_start, p_large_range_end;
                            dev_lower_bound_galloping(ad_map[cur_attr].data[small_idx_offset][p_small_range_start],
                                                      ad_map[cur_attr].data[1-small_idx_offset],
                                                      l_range_start[s_scanned_rels[cur_attr]+1-small_idx_offset],
                                                      l_range_end[s_scanned_rels[cur_attr]+1-small_idx_offset],
                                                      p_large_range_start);
                            dev_upper_bound_galloping(ad_map[cur_attr].data[small_idx_offset][p_small_range_end-1],
                                                      ad_map[cur_attr].data[1-small_idx_offset],
                                                      l_range_start[s_scanned_rels[cur_attr]+1-small_idx_offset],
                                                      l_range_end[s_scanned_rels[cur_attr]+1-small_idx_offset],
                                                      p_large_range_end);

                            auto task_id = tb->push_task(
                                    p_small_range_start, p_small_range_end,
                                    p_large_range_start, p_large_range_end,
                                    cur_iterators[lwarpId], iRes[lwarpId], cur_attr);
                            cq->enqueue(task_id); //push the task to the concurrent queue
                        }
                        __syncwarp();
                        found = false; //skip the task
                    }
                    else {
#ifdef ALFTJ_DEBUG
                        if(0 == lane) {
                            printf("cur_attr:%d, Array 0[%lu,%lu), Array 1[%lu,%lu)\n",
                                   cur_attr,
                                   range_start[lwarpId][s_scanned_rels[cur_attr]+small_idx_offset],
                                   arr_tail[lwarpId][s_scanned_rels[cur_attr]+small_idx_offset],
                                   range_start[lwarpId][s_scanned_rels[cur_attr]+1-small_idx_offset],
                                   arr_tail[lwarpId][s_scanned_rels[cur_attr]+1-small_idx_offset]);
                            printf("Array 0: ");
                            for(auto    x = range_start[lwarpId][s_scanned_rels[cur_attr]+small_idx_offset];
                                x < arr_tail[lwarpId][s_scanned_rels[cur_attr]+small_idx_offset];
                                x++) {
                                printf("%d, ", ad_map[cur_attr].data[small_idx_offset][x]);
                            }
                            printf("\n");
                            printf("Array 1: ");
                            for(auto    x = range_start[lwarpId][s_scanned_rels[cur_attr]+1-small_idx_offset];
                                x < arr_tail[lwarpId][s_scanned_rels[cur_attr]+1-small_idx_offset];
                                x++) {
                                printf("%d, ", ad_map[cur_attr].data[1-small_idx_offset][x]);
                            }
                            printf("\n");
                        }
                        __syncwarp();
#endif
                        found = warp_intersection_binary( //do the normal intersection
                                ad_map[cur_attr].data[small_idx_offset], ad_map[cur_attr].data[1-small_idx_offset],
                                range_start[lwarpId][s_scanned_rels[cur_attr]+small_idx_offset],
                                arr_tail[lwarpId][s_scanned_rels[cur_attr]+small_idx_offset],
                                range_start[lwarpId][s_scanned_rels[cur_attr]+1-small_idx_offset],
                                arr_tail[lwarpId][s_scanned_rels[cur_attr]+1-small_idx_offset],
                                iterators[lwarpId][cur_attr],
                                auc[lwarpId],
                                msIdx[lwarpId][cur_attr],
                                lane, small_idx_offset);
                        if (!found) { //reset range_start
                            if (0 == lane) {
                                range_start[lwarpId][s_scanned_rels[cur_attr]+small_idx_offset] = arr_head[lwarpId][s_scanned_rels[cur_attr]+small_idx_offset];
                                range_start[lwarpId][s_scanned_rels[cur_attr]+1-small_idx_offset] = arr_head[lwarpId][s_scanned_rels[cur_attr]+1-small_idx_offset];
                            }
                            __syncwarp();
                        }
                        else { //update iRes
                            if (0 == lane) {
                                auto cur_iterator = iterators[lwarpId][cur_attr][0][msIdx[lwarpId][cur_attr]];
                                iRes[lwarpId][cur_attr] = ad_map[cur_attr].data[0][cur_iterator];
                            }
                        }
                        __syncwarp();
#ifdef ALFTJ_DEBUG
                        if(0 == lane) {
                            printf("After intersection, %d res\n", (int)msIdx[lwarpId][cur_attr]+1);
                            printf("msIdx[%d]=%d\n", cur_attr, msIdx[lwarpId][cur_attr]);
                            printf("Array 0 res: ");
                            for (auto i = 0; i < msIdx[lwarpId][cur_attr]+1; i++) {
                                printf("%lu, ", iterators[lwarpId][cur_attr][0][i]);
                            }
                            printf("\n");

                            printf("Array 1 res: ");
                            for (auto i = 0; i < msIdx[lwarpId][cur_attr]+1; i++) {
                                printf("%lu, ", iterators[lwarpId][cur_attr][1][i]);
                            }
                            printf("\n");
                        }
#endif
                    }
                }
                /*if the intersection returns no results, backtrack to the last level*/
                if (!found) {
                    if (cur_attr > start_attr) { //backtrack to the last attribute
                        cur_attr--;
                    }
                    else if (0 == lane) { //reach the first attr, set values for getting a new probe item
                        range_start[lwarpId][s_scanned_rels[start_attr]] = arr_tail[lwarpId][s_scanned_rels[start_attr]];
                    }
                    __syncwarp();
                    continue;
                }
            }
            __syncwarp();
        }
        else {
            SIN_L(msIdx[lwarpId][cur_attr]--);

            if (0 == lane) { //update iRes
                auto cur_iterator = iterators[lwarpId][cur_attr][0][msIdx[lwarpId][cur_attr]];
                iRes[lwarpId][cur_attr] = ad_map[cur_attr].data[0][cur_iterator];
            }
            __syncwarp();

            /* reset all but not the smallest_rel_1st when cur_attr = start_attr_idx */
            if ((start_attr == cur_attr) && (lane < num_tables) && (lane != smallest_rel_1st)) {
                range_start[lwarpId][s_first_se_idx_per_table[lane]] = 0;
                arr_tail[lwarpId][s_first_se_idx_per_table[lane]] = Tries[lane].data_len[0];
            }
        }
        __syncwarp();

        /*update the start and end of next level of the tables appeared in this level*/
//        if (lane < MAX_REL_PER_LEVEL) {
        if (lane < ad_map[cur_attr].num) {
            auto cur_rel = ad_map[cur_attr].rel[lane];
            auto cur_level_in_rel = ad_map[cur_attr].levels[lane];
            if (cur_level_in_rel < Tries[cur_rel].num_attrs - 1) { //when we have next level
                auto cur_msIdx = msIdx[lwarpId][cur_attr];
                auto cur_offset = iterators[lwarpId][cur_attr][lane][cur_msIdx];
                auto next_se_idx = ad_map[cur_attr].next_se_idx[lane];
                range_start[lwarpId][next_se_idx] = Tries[cur_rel].offsets[cur_level_in_rel][cur_offset] - Tries[cur_rel].trie_offsets[cur_level_in_rel+1];
                arr_tail[lwarpId][next_se_idx] = Tries[cur_rel].offsets[cur_level_in_rel][cur_offset+1] - Tries[cur_rel].trie_offsets[cur_level_in_rel+1];
                arr_head[lwarpId][next_se_idx] = Tries[cur_rel].offsets[cur_level_in_rel][cur_offset] - Tries[cur_rel].trie_offsets[cur_level_in_rel+1];
            }
        }
        __syncwarp();
        cur_attr++; //advance to the next attr
        SIN_L(msIdx[lwarpId][cur_attr] = 0); //init the msIdx for the next attr
    }
}

__global__ void dummy_1 (int *a) {
    int tid = threadIdx.x;
    a[0] = tid;
}
__global__ void dummy_2 (int *a) {
    int tid = threadIdx.x;
    a[0] = tid;
}

/*
 * Definition of ALFTJ class
 * */
template<typename DataType, typename CntType>
class ALFTJ : public LFTJ_Base<DataType,CntType> {
    /*DFS-specific data structures*/
    GCQueue<CntType,CntType> *cq;
    LFTJTaskBook<DataType,CntType,CntType> *tb;
    CBarrier *br;
    AttrDataMapping<DataType,CntType,uint32_t> *ad_map;
    uint32_t *scanned_num_rels_each_attr;
    int grid_size;
public:
    ALFTJ(uint32_t num_tables, uint32_t num_attrs, uint32_t *attr_order, CUDAMemStat *memstat, CUDATimeStat *timing){
        this->num_tables = num_tables;
        this->num_attrs = num_attrs;
        this->attr_order = attr_order;
        this->memstat = memstat;
        this->timing = timing;
        CUDA_MALLOC(&this->ad_map, sizeof(AttrDataMapping<DataType,CntType,uint32_t>)*num_attrs, memstat);
        CUDA_MALLOC(&this->scanned_num_rels_each_attr, sizeof(uint32_t)*(num_attrs+1), memstat);

        /*kernel setting for persistent warps*/
        cudaDeviceProp prop;
        int accBlocksPerSM;
        cudaGetDeviceProperties(&prop, DEVICE_ID);
        auto maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
        auto numSM = prop.multiProcessorCount;
        checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &accBlocksPerSM, ALFTJ_count<DataType, CntType>,BLOCK_SIZE_DEFAULT, 0));//todo: accBlocksPerSM may be inaccurate for individual SM
        this->grid_size = numSM * accBlocksPerSM; //set the grid_size,
        log_info("grisSize = %d, blockSize = %d.", grid_size, BLOCK_SIZE_DEFAULT);
        log_info("Kernel: ALFTJ_count, occupancy: %d/%d.",accBlocksPerSM, maxThreadsPerSM/BLOCK_SIZE_DEFAULT);

        /*data structure for work-sharing*/
        CUDA_MALLOC(&this->cq, sizeof(GCQueue<CntType,CntType>), memstat);
        CUDA_MALLOC(&this->tb, sizeof(LFTJTaskBook<CntType,CntType,CntType>), memstat);
        CUDA_MALLOC(&this->br, sizeof(CBarrier), memstat);
        this->cq->init(750000, memstat);
        this->tb->init(750000, num_attrs, memstat);
        this->br->initWithWarps(grid_size*BLOCK_SIZE_DEFAULT/WARP_SIZE, memstat);
        log_debug("Init with %d warps", grid_size*BLOCK_SIZE_DEFAULT/WARP_SIZE);
    };

    ~ALFTJ() {
#ifdef FREE_DATA
    for(auto i = 0; i < this->num_attrs; i++) this->ad_map[i].clear();
    CUDA_FREE(this->ad_map, this->memstat);
    CUDA_FREE(this->scanned_num_rels_each_attr, this->memstat);
    this->cq->clear();
    this->tb->clear();
    this->br->clear();
    CUDA_FREE(this->cq, this->memstat);
    CUDA_FREE(this->tb, this->memstat);
    CUDA_FREE(this->br, this->memstat);
#endif
    };

    CntType evaluate(IndexedTrie<DataType,CntType> *Tries,
                     DataType **res_tuples, CntType *num_res,
                     bool ooc, bool work_sharing, cudaStream_t stream) override {
        log_trace("In DFS-LFTJ evaluate");
        Timer t;

        checkCudaErrors(cudaMemsetAsync(num_res, 0, sizeof(CntType),stream)); //todo: ooc revise it

//        int *dummy_num;
//        CUDA_MALLOC(&dummy_num, sizeof(int), nullptr, stream);
//        dummy_1<<<1,1,0,stream>>>(dummy_num); //todo: dummy kernel for identifying prefetching in nvvp

        /*reset the global values in this stream*/
        static bool first_in = true;
        if (first_in) { //init scanned_num_rels_each_attr with the first Trie
            /*Construct the data mapping between attributes and relations
             *the attribute-relation mapping, indicating the relations the attributes are in*/

            /*todo: only two attributes per table, need to be modified in the future*/
            uint32_t *num_rels_each_attr = new uint32_t[this->num_attrs];
            memset(num_rels_each_attr, 0, sizeof(uint32_t)*this->num_attrs);
            for(auto i = 0; i < this->num_tables; i++) {
                auto prev_attr = Tries[i].attr_list[0];
                auto suc_attr = Tries[i].attr_list[1];
                num_rels_each_attr[prev_attr]++;
                num_rels_each_attr[suc_attr]++;
            }

            /*prefix sum on num_rels_each_attr*/
            uint32_t acc = 0;
            for(auto i = 0; i < this->num_attrs; i++) {
                scanned_num_rels_each_attr[i] = acc;
                acc += num_rels_each_attr[i];
            }

            scanned_num_rels_each_attr[this->num_attrs] = acc;
            for(auto i = 0; i < this->num_attrs; i++) {
                ad_map[i].init(num_rels_each_attr[i], this->memstat);
            }
            delete[] num_rels_each_attr;
            first_in = false;
        }
        cq->reset(stream); tb->reset(stream); br->reset(stream);

        uint32_t *first_se_idx_per_table = nullptr;
        CUDA_MALLOC(&first_se_idx_per_table, sizeof(uint32_t)*this->num_tables, this->memstat, stream);

//        dummy_2<<<1,1,0,stream>>>(dummy_num); //todo: dummy kernel for identifying prefetching in nvvp

        uint32_t *scanned_num_rels_each_attr_change = new uint32_t[this->num_attrs+1];
        memcpy(scanned_num_rels_each_attr_change, scanned_num_rels_each_attr, sizeof(uint32_t)*(this->num_attrs+1));
        for(uint32_t i = 0; i < this->num_tables; i++) {
            auto key1_attr = Tries[i].attr_list[0]; /*key1*/
            auto key1_cnt = scanned_num_rels_each_attr_change[key1_attr]-scanned_num_rels_each_attr[key1_attr];
            ad_map[key1_attr].rel[key1_cnt] = i;
            ad_map[key1_attr].levels[key1_cnt] = 0;
            ad_map[key1_attr].data_len[key1_cnt] = Tries[i].data_len[0];
            ad_map[key1_attr].data[key1_cnt] = Tries[i].data[0];
            first_se_idx_per_table[i] = scanned_num_rels_each_attr_change[key1_attr]++;
#ifdef HASH_PROBE
            ad_map[key1_attr].buc_ptrs[key1_cnt] = Tries[i].buc_ptrs[0];
#endif
            auto key2_attr = Tries[i].attr_list[1]; /*key2*/
            auto key2_cnt = scanned_num_rels_each_attr_change[key2_attr]-scanned_num_rels_each_attr[key2_attr];
            ad_map[key2_attr].rel[key2_cnt] = i;
            ad_map[key2_attr].levels[key2_cnt] = 1;
            ad_map[key2_attr].data_len[key2_cnt] = Tries[i].data_len[1];
            ad_map[key2_attr].data[key2_cnt] = Tries[i].data[1];
            ad_map[key1_attr].next_se_idx[key1_cnt] = scanned_num_rels_each_attr_change[key2_attr]++;
            ad_map[key2_attr].next_se_idx[key2_cnt] = 99999;
#ifdef HASH_PROBE
            ad_map[key2_attr].buc_ptrs[key2_cnt] = Tries[i].buc_ptrs[1];
#endif
        }
        checkCudaErrors(cudaMemPrefetchAsync(first_se_idx_per_table, sizeof(uint32_t)*this->num_tables, DEVICE_ID, stream));

        auto rows = ad_map[this->attr_order[0]].num; //how many rows are intersected
        int smallest_rel_1st = -1;
        auto smallest_rel_card_1st = INT32_MAX;
        for(auto i = 0; i < rows; i++) { /*get the shortest row in attr0*/
            if (ad_map[this->attr_order[0]].data_len[i] < smallest_rel_card_1st) {
                smallest_rel_card_1st = ad_map[this->attr_order[0]].data_len[i];
                smallest_rel_1st = ad_map[this->attr_order[0]].rel[i];
            }
        }
        CntType *probeIter = nullptr;
        cudaMalloc((void**)&probeIter, sizeof(CntType));
        cudaMemset(probeIter, 0, sizeof(CntType));

        /*print the anxillary data*/
        cout<<"========= Anxillary data structures ============="<<endl;
        cout<<"Trie attrs: "<<endl;
        for(auto i = 0; i < this->num_tables; i++) {
            cout<<"Trie["<<i<<"]: ("<<Tries[i].attr_list[0]<<","<<Tries[i].attr_list[1]<<")"<<endl;
        }
        cout<<"ad_map: "<<endl;
        for(auto i = 0; i < this->num_attrs; i++) {
            cout<<"ad_map "<<i<<": ";
            for(auto j = 0; j < ad_map[i].num; j++) {
                cout<<ad_map[i].rel[j]<<' ';
            }
            cout<<endl;
        }
        cout<<"scanned_num_rels_each_attr: ";
        for(auto i = 0; i < this->num_attrs+1; i++) {
            cout<<this->scanned_num_rels_each_attr[i]<<' ';
        }
        cout<<endl;
        cout<<"first_se_idx_per_table: ";
        for(auto i = 0; i < this->num_tables; i++) {
            cout<<first_se_idx_per_table[i]<<' ';
        }
        cout<<endl;
        cout<<"smallest_rel_1st = "<<smallest_rel_1st<<endl;
        cout<<"================================================="<<endl;

        /*probe-count*/
        if (ooc) {
            if (work_sharing) {
                ALFTJ_count<DataType, CntType, true> <<<this->grid_size, BLOCK_SIZE_DEFAULT, 0, stream>>> (Tries, this->num_tables, this->num_attrs, ad_map, smallest_rel_1st, smallest_rel_card_1st, first_se_idx_per_table, scanned_num_rels_each_attr, probeIter, num_res, cq, tb, br);
            }
            else {
                ALFTJ_count<DataType, CntType, false> <<<this->grid_size, BLOCK_SIZE_DEFAULT, 0, stream>>> (Tries, this->num_tables, this->num_attrs, ad_map, smallest_rel_1st, smallest_rel_card_1st, first_se_idx_per_table, scanned_num_rels_each_attr, probeIter, num_res, cq, tb, br);
            }
        }
        else {
            auto timestamp = this->timing->get_idx();
            if (work_sharing) {
                execKernel((ALFTJ_count<DataType, CntType, true>), this->grid_size, BLOCK_SIZE_DEFAULT, this->timing, false, Tries, this->num_tables, this->num_attrs, ad_map, smallest_rel_1st, smallest_rel_card_1st, first_se_idx_per_table, scanned_num_rels_each_attr, probeIter, num_res, cq, tb, br);
            }
            else {
                execKernel((ALFTJ_count<DataType, CntType, false>), this->grid_size, BLOCK_SIZE_DEFAULT, this->timing, false, Tries, this->num_tables, this->num_attrs, ad_map, smallest_rel_1st, smallest_rel_card_1st, first_se_idx_per_table, scanned_num_rels_each_attr, probeIter, num_res, cq, tb, br);
            }
            log_info("ALFTJ_count time: %.2f ms", this->timing->diff_time(timestamp));
        }
        CntType total_cnt = num_res[0];
        log_info("Output count: %llu.", total_cnt);

        /*2.global scan & output page generation*/
        log_debug("GCQueue: qHead=%d, qRear=%d.", cq->qHead[0], cq->qRear[0]);

        /*reset the data structures for probe-write*/
        this->cq->reset(stream); this->tb->reset(stream);
        checkCudaErrors(cudaMemsetAsync(num_res, 0, sizeof(CntType),stream));
        checkCudaErrors(cudaMemsetAsync(probeIter, 0, sizeof(CntType),stream));
        br->reset_with_warps(this->grid_size*BLOCK_SIZE_DEFAULT/WARP_SIZE, stream);

        if (res_tuples == nullptr) {
            log_info("Allocate space for res_tuple");
            CUDA_MALLOC(&res_tuples, sizeof(DataType*)*this->num_attrs, this->memstat);
            for(auto i = 0; i < this->num_attrs; i++)
                CUDA_MALLOC(&res_tuples[i], sizeof(DataType)*total_cnt, this->memstat);
        }

        /*probe-write*/
        if (ooc) {
            if (work_sharing) {
                ALFTJ_write<DataType, CntType, true> <<<this->grid_size, BLOCK_SIZE_DEFAULT, 0, stream>>> (Tries, this->num_tables, this->num_attrs, ad_map, smallest_rel_1st, smallest_rel_card_1st, first_se_idx_per_table, scanned_num_rels_each_attr, probeIter, num_res, res_tuples, cq, tb, br);
                cudaStreamSynchronize(stream);
            }
            else {
                ALFTJ_write<DataType, CntType, false> <<<this->grid_size, BLOCK_SIZE_DEFAULT, 0, stream>>> (Tries, this->num_tables, this->num_attrs, ad_map, smallest_rel_1st, smallest_rel_card_1st, first_se_idx_per_table, scanned_num_rels_each_attr, probeIter, num_res, res_tuples, cq, tb, br);
                cudaStreamSynchronize(stream);
            }
        }
        else {
            auto timestamp = this->timing->get_idx();
            if (work_sharing) {
                execKernel((ALFTJ_write<DataType, CntType, true>), this->grid_size, BLOCK_SIZE_DEFAULT, this->timing, false, Tries, this->num_tables, this->num_attrs, ad_map, smallest_rel_1st, smallest_rel_card_1st, first_se_idx_per_table, scanned_num_rels_each_attr, probeIter, num_res, res_tuples, cq, tb, br);
            }
            else {
                execKernel((ALFTJ_write<DataType, CntType, false>), this->grid_size, BLOCK_SIZE_DEFAULT, this->timing, false, Tries, this->num_tables, this->num_attrs, ad_map, smallest_rel_1st, smallest_rel_card_1st, first_se_idx_per_table, scanned_num_rels_each_attr, probeIter, num_res, res_tuples, cq, tb, br);
            }
            log_info("ALFTJ_write time: %.2f ms", this->timing->diff_time(timestamp));
        }

        cout<<"res:"<<endl;
        for(int i = 0; i < (int)total_cnt; i++) {
            if (i >= 0) {
                cout<<i<<": ";
                for(auto j = 0; j < this->num_attrs; j++) {
                    cout<<res_tuples[j][i]<<' ';
                }
                cout<<endl;
            }
        }

//#ifdef FREE_DATA
//    CUDA_FREE(first_se_idx_per_table, this->memstat); //will block the CPU
//    cudaFree(probeIter);
//#endif
        return total_cnt;
    };
};
