//
// Created by Bryan on 24/2/2020.
//

#pragma once

#include "ALFTJ.cuh"
#include "LFTJ_BFS.cuh"

#define HIGH_INFINITY   (INT32_MAX)
#define LOW_INFINITY    (INT32_MAX-1)
#define SATISFIED_SCALE (0.9) //we are satisfied when current budget reaches SATISFIED_SCALE*max budget

class BudgetPlanner {
    bsize_t mem_total;      //total amount of memory available (in bytes)
    uint32_t num_attrs;     //total number of attrs
    bsize_t *acc_planned;   //accumulated amount of memory planned according to mem_total and num_attrs
    bsize_t *acc_used;      //accumulated amount of used mem
public:
    BudgetPlanner(bsize_t mem_total, uint32_t num_attr, uint32_t* num_rels_per_attr) {
        this->mem_total = mem_total;
        this->num_attrs = num_attr;
        acc_planned = new bsize_t[num_attr];
        acc_used = new bsize_t[num_attr+1];
        acc_used[0] = 0;

        uint32_t num_rels = 0;
        for(auto i = 0; i < num_attr; i++) num_rels += num_rels_per_attr[i];
        auto size_per_rel = mem_total / num_rels;
        uint32_t acc_rels = 0;
        for(auto i = 0; i < num_attr; i++) {
            acc_rels += num_rels_per_attr[i];
            acc_planned[i] = acc_rels * size_per_rel;
        }
    }

    ~BudgetPlanner() {
        delete[] acc_planned;
        delete[] acc_used;
    }

    bsize_t getAvailableBudget(uint32_t cur_attr) { //return the amount useable memory
        return acc_planned[cur_attr] - acc_used[cur_attr];
    }

    void setUsed(uint32_t cur_attr, bsize_t mem_used) { //record the used memory for this attr
        acc_used[cur_attr+1] = acc_used[cur_attr] + mem_used;
    }
};

__global__ void dummy_begin (int *a) {
    int tid = threadIdx.x;
    a[0] = tid;
}

__global__ void dummy_end (int *a) {
    int tid = threadIdx.x;
    a[0] = tid;
}

template<typename DataType, typename CntType>
class OOCWrapper {
private:
    bsize_t total_budget;

    /*Given the low and high values on level 0, produce the Trie slice by simply setting the pointer
     *The slice space will not be freed since it consists only pointers to the original Trie*/
    void provision(
            IndexedTrie<DataType,CntType> input_Trie, IndexedTrie<DataType,CntType> &output_Trie,
            DataType low, DataType high, cudaStream_t stream) {
        /*input_Trie.data[0][lower_bound,upper_bound) is the target range*/
        auto lower_bound = lower_bound_galloping(low, input_Trie.data[0], (CntType)0, input_Trie.data_len[0]);
        auto upper_bound = upper_bound_galloping(high, input_Trie.data[0], (CntType)0, input_Trie.data_len[0]);
        if(lower_bound >= upper_bound) { //there is no items in this Trie
            output_Trie.validity = false; //empty Trie
            return;
        }
        output_Trie.validity = true; //valid Trie

        /*add dummy kernel to show the situation of this stream*/
//        int *dummy_num;
//        cudaMalloc((void**)&dummy_num, sizeof(int));
//        dummy_1<<<1,1,0,stream>>>(dummy_num); //todo: dummy kernel for identifying prefetching in nvvp

        /*todo: it seems like the setting of output_Trie on the CPU hinders the following LFTJ execution, gap exists in two iterations, check it later*/
        auto moving_head = lower_bound, moving_tail = upper_bound;
        for(auto n = 0; n < output_Trie.num_attrs; n++) { /*partition the split var*/
            output_Trie.trie_offsets[n] = moving_head; //set trie_offset
            output_Trie.data_len[n] = moving_tail-moving_head; //set data_len
            output_Trie.data[n] = input_Trie.data[n]+output_Trie.trie_offsets[n]; //set val pointers
            checkCudaErrors(cudaMemPrefetchAsync(output_Trie.data[n], sizeof(DataType)*output_Trie.data_len[n], DEVICE_ID, stream));

            if (n < output_Trie.num_attrs-1) {
                output_Trie.offsets[n] = input_Trie.offsets[n]+output_Trie.trie_offsets[n]; //set offset pointers
                checkCudaErrors(cudaMemPrefetchAsync(output_Trie.offsets[n], sizeof(CntType)*(output_Trie.data_len[n]+1), DEVICE_ID, stream));
                moving_head = input_Trie.offsets[n][moving_head]; //update head and tail
                moving_tail = input_Trie.offsets[n][moving_tail];
            }
        }
        checkCudaErrors(cudaMemPrefetchAsync(output_Trie.trie_offsets, sizeof(CntType)*output_Trie.num_attrs, DEVICE_ID, stream));
        checkCudaErrors(cudaMemPrefetchAsync(output_Trie.data_len, sizeof(CntType)*output_Trie.num_attrs, DEVICE_ID, stream));

//        dummy_2<<<1,1,0,stream>>>(dummy_num); //todo: dummy kernel for identifying prefetching in nvvp
    }

    /*todo: consider spill*/
    /*Given the next(set last time) and budget on this level, upate current
     * in terms of the direction for all the Tries.
     * Return the budget that is used*/
    bsize_t probe(IndexedTrie<DataType,CntType> *Tries, uint32_t num_Tries,
            DataType next, DataType &current,
            const bsize_t budget, int direction) {
        CntType high_limit = 0; //high_limit should be minimum of the last value of 1st attr of the Tries
        for(auto i = 0; i < num_Tries; i++) {
            auto first_attr_len = Tries[i].data_len[0];
            if (high_limit < Tries[i].data[0][first_attr_len-1])
                high_limit = Tries[i].data[0][first_attr_len-1];
        }
        if (next > high_limit) { //no results since next is greater than the max val or no input tries
            current = HIGH_INFINITY;
            return 0;
        }
        else if (next < 0) {
            current = LOW_INFINITY;
            return 0;
        }
        long long int high_scale = (direction == 1)? high_limit : next;
        long long int low_scale = (direction == 1)? next : 0;
        long long int last_best_scale = (direction == 1)? high_limit : 0;
        bsize_t used_budget_last_best = 0;

        while (low_scale <= high_scale) {
            auto mid_scale = low_scale + (high_scale-low_scale)/2;
            if (mid_scale == next) {
                current = mid_scale;
                return 0;
            }
            /*compute memory consumption when considering "VALUE" range [next,mid_scale]*/
            bsize_t temp_budget = 0;
            for(auto t = 0; t < num_Tries; t++) {
                auto scale_larger = (next > mid_scale) ? next : mid_scale;
                auto scale_smaller = (next < mid_scale) ? next : mid_scale;
                auto lower_bound = lower_bound_galloping(scale_smaller, Tries[t].data[0],
                                                         (CntType)0, Tries[t].data_len[0]);
                auto upper_bound = upper_bound_galloping(scale_larger, Tries[t].data[0],
                                                         (CntType)0, Tries[t].data_len[0]);
                /*Tries[t].data[0][lower_bound,upper_bound) is the target range*/
                temp_budget += Tries[t].get_size_from_1st(lower_bound,upper_bound);
            }

            if (temp_budget > budget) {
                if (direction == 1) high_scale = mid_scale-1;
                else if (direction == -1) low_scale = mid_scale + 1;
            }
            else if ((temp_budget <= budget) && (temp_budget>= SATISFIED_SCALE*budget)) {
                current = mid_scale;
                return temp_budget;
            }
            else { //current temp_budget is valid but there may be better results, store it
                if (temp_budget > used_budget_last_best) {
                    used_budget_last_best = temp_budget;
                    last_best_scale = mid_scale;
                }
                if (direction == 1) low_scale = mid_scale+1;
                else if (direction == -1) high_scale = mid_scale-1;
            }
        }
        /*If coming here, it means we have not got available results
         *in binary search since we have missed it. Return the stored one*/
        current = last_best_scale;
        return used_budget_last_best;
    }

    CntType execute(IndexedTrie<DataType,CntType> *input_Tries, uint32_t num_Tries,
            uint32_t num_attrs, bsize_t budget,  //memory budget (in bytes) for all the data
            bool ooc, //whether to use out-of-core processing
            bool work_sharing,
            LFTJ_Base<DataType,CntType> *core,
            CUDAMemStat *memstat) {
        assert(num_Tries <= MAX_NUM_TABLES);
        assert(num_attrs <= MAX_NUM_ATTRS);
        Timer t;
        CntType final_res;
        CntType *num_results = nullptr; //todo: delete num_results_acc and write res
        CUDA_MALLOC(&num_results, sizeof(CntType), memstat);
        cudaMemset(num_results, 0, sizeof(CntType));

        DataType **res_tuples = nullptr;
//        CUDA_MALLOC(&res_tuples, sizeof(DataType*)*num_attrs, memstat);

        int *dummy_num;
        CUDA_MALLOC(&dummy_num, sizeof(int), nullptr);

        if (!ooc) { //directly execute the single-pass process
            log_info("Out-of-core processing disabled");
            return core->evaluate(input_Tries, res_tuples, num_results, false, work_sharing, 0);
        }
        log_info("Out-of-core processing enabled");

        auto output_budget = 0;
        auto input_budget = budget - output_budget;
//        auto input_budget = 50 * 1024 * 1024; //50M

        IndexedTrie<DataType,CntType> atoms[MAX_NUM_ATTRS][MAX_NUM_TABLES];
        uint32_t slice_cnt[MAX_NUM_ATTRS] = {0};
        uint32_t slice_cnt_scanned[MAX_NUM_ATTRS] = {0};
        for(auto i = 0; i < num_Tries; i++) { //divide the input Trie into different atom groups
            assert(input_Tries[i].num_attrs > 0);
            auto first_attr = input_Tries[i].attr_list[0];
            atoms[first_attr][slice_cnt[first_attr]++] = input_Tries[i];
        }
        BudgetPlanner b_planner(input_budget, num_attrs, slice_cnt);

        uint32_t slice_acc = 0;
        for(auto i = 0; i < num_attrs; i++) { //prefix sum on the slice_cnt
            slice_cnt_scanned[i] = slice_acc;
            slice_acc += slice_cnt[i];
        }

        /*double gathered groups for updating (in provision) and using (in evalute) separately*/
        IndexedTrie<DataType,CntType> *gathered_trie = nullptr, *gathered_trie_for_evaluation = nullptr;
        CUDA_MALLOC(&gathered_trie, sizeof(IndexedTrie<DataType,CntType>)*MAX_NUM_TABLES, memstat);
        CUDA_MALLOC(&gathered_trie_for_evaluation, sizeof(IndexedTrie<DataType,CntType>)*MAX_NUM_TABLES, memstat);

        /*copy some attrs of the input Tries to the corresponding gathered_tris slots.
         * the remaining attrs of the gathered_trie are updated in provision*/
        auto trie_idx = 0;
        for(auto i = 0; i < num_attrs; i++) { //loop over the attr groups
            for(auto j = 0; j < slice_cnt[i]; j++) { //loop over the input Tries in this group
                gathered_trie[trie_idx].init(atoms[i][j].num_attrs, atoms[i][j].memstat); //init
                gathered_trie_for_evaluation[trie_idx].init(atoms[i][j].num_attrs, atoms[i][j].memstat); //init
                cudaMemcpy(gathered_trie[trie_idx].attr_list,
                           atoms[i][j].attr_list,
                           sizeof(uint32_t)*gathered_trie[trie_idx].num_attrs,
                           cudaMemcpyDeviceToDevice); //copy the attr_list
                cudaMemcpy(gathered_trie_for_evaluation[trie_idx].attr_list,
                           atoms[i][j].attr_list,
                           sizeof(uint32_t)*gathered_trie_for_evaluation[trie_idx].num_attrs,
                           cudaMemcpyDeviceToDevice); //copy the attr_list
                trie_idx++;
            }
        }
        assert(trie_idx == num_Tries);

        uint32_t cur_attr = 0, lftjs = 0, next_change_attr = 0;
        bool is_record_next_change = true; //whether to record next_change_attr in probe
        DataType current[MAX_NUM_ATTRS] = {0};
        DataType next[MAX_NUM_ATTRS] = {0};
        int direction[MAX_NUM_ATTRS];
        for(auto i = 0; i < num_attrs; i++) direction[i] = 1;

        /*prefetch on streams[0] (except prefetch for the first LFTJ, it is on streams[1]),
         * kernel execution on streams[1]*/
//        cudaStream_t streams[2];
//        for(auto i = 0; i < 2; i++) cudaStreamCreate(&streams[i]);//pipelining
        cudaStream_t streams[3];
//        for(auto i = 0; i < 3; i++) cudaStreamCreate(&streams[i]);//pipelining

        for(auto i = 0; i < 3; i++) streams[i] = 0;                 //no pipelining

        cudaDeviceSynchronize();
        t.reset();
        while (true) {

            /*
             * 1. let (0 == slice_cnt[cur_attr]) in because we need to probe empty attribute once, todo: revise
             * 2. let (0 == lfjts) in because we need to provision all the Tries at first
             * 3. let is_record_next_change in because we need to find the attr that is changed in next iteration
             * */
            auto slice_start = slice_cnt_scanned[cur_attr];
            if ((0 == lftjs) || is_record_next_change || (0 == slice_cnt[cur_attr])) { //probe and provision
                auto used_mem = probe(atoms[cur_attr], slice_cnt[cur_attr],
                                      next[cur_attr], current[cur_attr],
                                      b_planner.getAvailableBudget(cur_attr), direction[cur_attr]);
                b_planner.setUsed(cur_attr, used_mem);

                if ((0 == cur_attr)&&(current[cur_attr] == HIGH_INFINITY)) break; //exit
                if ((current[cur_attr] == HIGH_INFINITY)||
                    (current[cur_attr] == LOW_INFINITY)) { //backtrack and change direction
                    direction[cur_attr] *= -1;
                    cur_attr--;
                    continue;
                }
                if (is_record_next_change) { //record the changed attr
                    next_change_attr = cur_attr;
                    is_record_next_change = false; //only once to ensure the smallest
                }

                auto stream_for_provision = (0 == lftjs) ? streams[2] : streams[1];
                if ((0 == lftjs) || (cur_attr == next_change_attr)) { //selective provisioning
                    for(auto i = 0; i < slice_cnt[cur_attr]; i++) { //provision the Trie with head attr as cur_attr
                        auto lower = (next[cur_attr] < current[cur_attr]) ? next[cur_attr] : current[cur_attr];
                        auto higher = (next[cur_attr] > current[cur_attr]) ? next[cur_attr] : current[cur_attr];
                        dummy_1<<<1,1,0,stream_for_provision>>>(dummy_num);
                        provision(atoms[cur_attr][i], gathered_trie[slice_start+i],
                                  lower, higher, stream_for_provision);
                        dummy_2<<<1,1,0,stream_for_provision>>>(dummy_num);
                    }
                }
                next[cur_attr] = current[cur_attr] + direction[cur_attr]; //for next probe
            }
            else { //update next[cur_attr] for those unchanged Trie slices
                if (direction[cur_attr] == 1) { //find the maximum val appeared in last LFTJ
                    DataType last_max = gathered_trie[slice_start].data[0][gathered_trie[slice_start].data_len[0]-1];
                    for(auto i = 1; i < slice_cnt[cur_attr]; i++) {
                        if (gathered_trie[slice_start+i].data[0][gathered_trie[slice_start+i].data_len[0]-1] > last_max)
                            last_max = gathered_trie[slice_start+i].data[0][gathered_trie[slice_start+i].data_len[0]-1];
                    }
                    next[cur_attr] = last_max+1;
                }
                else if (direction[cur_attr] == -1) {
                    DataType last_min = gathered_trie[slice_start].data[0][0];
                    for(auto i = 1; i < slice_cnt[cur_attr]; i++) {
                        if (gathered_trie[slice_start+i].data[0][0] < last_min)
                            last_min = gathered_trie[slice_start+i].data[0][0];
                    }
                    next[cur_attr] = last_min-1;
                }
            }

            if (cur_attr < num_attrs-1) { //advance to the next attr
                cur_attr++;
                if (slice_cnt[cur_attr] == 0) { //to ensure correctness for empty attrs
                    next[cur_attr] = 0;
                    direction[cur_attr] = 1;
                }
            }
            else { /*Perform LFTJ*/
                bool validity = true;
                for(auto i = 0; i < num_Tries; i++) //check the validity of each gathered Trie
                    validity &= gathered_trie[i].validity;
                if (validity) {
                    if (0 != lftjs) { //synchronize the device except the first prefetch
                        cudaDeviceSynchronize();
                        CntType cur_acc_res_1;
                        cudaMemcpyAsync(&cur_acc_res_1, num_results, sizeof(CntType), cudaMemcpyDeviceToHost,streams[0]);

//                        cudaMemcpy(&cur_acc_res, num_results_acc, sizeof(CntType), cudaMemcpyDeviceToHost);
//                        cudaMemcpyAsync(&cur_acc_res, num_results_acc, sizeof(CntType)*10000, cudaMemcpyDeviceToHost, streams[2]);
                        log_info("current accumulated res: %llu", cur_acc_res_1);
                    }

                    /*copy the gathered_trie data to gathered_trie_for_evaluation*/
                    for(auto x = 0; x < num_Tries; x++) { // copy 3 Tries
                        for(auto y = 0; y < gathered_trie[x].num_attrs; y++) { //each Trie 2 attrs
                            gathered_trie_for_evaluation[x].trie_offsets[y] = gathered_trie[x].trie_offsets[y];
                            gathered_trie_for_evaluation[x].data_len[y] = gathered_trie[x].data_len[y];
                            gathered_trie_for_evaluation[x].data[y] = gathered_trie[x].data[y];
                            if (y < gathered_trie[x].num_attrs - 1)
                                gathered_trie_for_evaluation[x].offsets[y] = gathered_trie[x].offsets[y];
                        }
                    }

                    log_info("-----Perform LFTJ %d-----", lftjs);
                    for(auto i = 0; i < num_Tries; i++) {
                        log_info("Trie %d(%d,%d): val:%d-%d,off[0]:%llu-%llu,off[1]:%llu-%llu,size: %.2f MB", i,
                                 gathered_trie[i].attr_list[0], gathered_trie[i].attr_list[1],
                                 gathered_trie[i].data[0][0],
                                 gathered_trie[i].data[0][gathered_trie[i].data_len[0]-1],
                                 gathered_trie[i].trie_offsets[0],
                                 gathered_trie[i].trie_offsets[0]+gathered_trie[i].data_len[0],
                                 gathered_trie[i].trie_offsets[1],
                                 gathered_trie[i].trie_offsets[1]+gathered_trie[i].data_len[1],
                                 1.0*gathered_trie[i].get_size_from_1st()/1024/1024);
                    }
                    core->evaluate(gathered_trie_for_evaluation, nullptr, num_results, true, work_sharing, streams[2]); //todo: revise
                    is_record_next_change = true;
                    lftjs++;
                }
            }
        }
        cudaDeviceSynchronize();
        log_info("loop time: %.2f s", t.elapsed());
        cudaMemcpy(&final_res, num_results, sizeof(CntType), cudaMemcpyDeviceToHost);
        log_info("Final res: %llu", final_res);
#ifdef FREE_DATA
        for(auto i = 0; i < num_Tries; i++) gathered_trie[i].clear(false); //do not delete data and offsets
        for(auto i = 0; i < num_Tries; i++) gathered_trie_for_evaluation[i].clear(false); //do not delete data and offsets
        CUDA_FREE(gathered_trie_for_evaluation, memstat);
        CUDA_FREE(gathered_trie, memstat);
#endif
        return final_res;
    }
public:
    OOCWrapper() {
        size_t free_byte, total_byte ;
        auto cuda_status = cudaMemGetInfo(&free_byte, &total_byte);//show memory usage of GPU
        if (cudaSuccess != cuda_status){
            log_error("cudaMemGetInfo fails, %s", cudaGetErrorString(cuda_status));
            exit(1);
        }
        auto used_byte = total_byte - free_byte;
        log_info("GPU memory usage: used = %.1f MB, free = %.1f MB, total = %.1f MB",
                 1.0*used_byte/1024.0/1024.0, 1.0*free_byte/1024.0/1024.0, 1.0*total_byte/1024.0/1024.0);
        total_budget = ((bsize_t)free_byte)/2; //double buffer
        log_info("Initialize OOCWrapper with budget: %.1f MB", 1.0*total_budget/1024.0/1024.0);
    }

    OOCWrapper(bsize_t budget) { //user-defined budget in MB
        this->total_budget = budget * 1024 * 1024;
        log_info("User defined budget for OOCWrapper: %.1f MB", 1.0*total_budget/1024.0/1024.0);
    }

    CntType execute_with_DFS(
            IndexedTrie<DataType, CntType> *Tries,
            uint32_t num_tables,
            uint32_t num_attrs,
            uint32_t *attr_order,
            bool ooc, //whether to use out-of-core processing
            bool work_sharing,
            CUDAMemStat *memstat,
            CUDATimeStat *timing) {
        ALFTJ<DataType,CntType> core(num_tables, num_attrs, attr_order, memstat, timing);
        return execute(Tries, num_tables, num_attrs, total_budget, ooc, work_sharing, &core, memstat);
    }
    CntType execute_with_BFS(
            IndexedTrie<DataType, CntType> *Tries,
            uint32_t num_tables,
            uint32_t num_attrs,
            uint32_t *attr_order,
            bool ooc, //whether to use out-of-core processing
            bool work_sharing,
            CUDAMemStat *memstat,
            CUDATimeStat *timing) {
        LFTJ_BFS<DataType,CntType> core(num_tables, num_attrs, attr_order, memstat, timing);
        return execute(Tries, num_tables, num_attrs, total_budget, ooc, work_sharing, &core, memstat);
    }
};