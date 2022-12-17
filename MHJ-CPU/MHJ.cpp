//
// Created by Bryan on 16/2/2019.
//
#include "omp.h"
#include "mem_helper.h"
#include "timer.h"
#include "helper.h"
#include "file_io.h"
#include "task_queue.h"
using namespace std;

/** holds the arguments passed to each thread */
template<typename DataType, typename CntType>
struct TBJ_arg_t {
    Relation<DataType,CntType> *m_probe_table; //the probe table
    uint32_t m_probe_table_part_len; //probe slice length of each thread
    uint32_t m_probe_offset; //probe slice offset of each thread

    uint32_t m_num_hash_tables; //number of hash tables
    Relation<DataType,CntType> *m_build_tables[MAX_NUM_BUILD_TABLES]; //the build tables
    CntType **m_build_hists[MAX_NUM_BUILD_TABLES]; //histograms for all threads (each thread has a head pointer)
    CntType m_build_offset[MAX_NUM_BUILD_TABLES]; //start index of each partition of each thread
    HashTable<DataType,CntType> *m_tmp_hash_tables[MAX_NUM_BUILD_TABLES];  //result after 1-pass partition
    HashTable<DataType,CntType> *m_final_hash_tables[MAX_NUM_BUILD_TABLES]; //output hash table memory space
    CntType m_build_part_len[MAX_NUM_BUILD_TABLES]; //build slice length of each build table of each thread

    CntType **m_buc_start_pos; //start positions of the buckets
    TaskQueue<DataType,CntType> *m_part_queue; //the task queue

    int m_first_pass_bits[MAX_NUM_BUILD_TABLES]; //number of bits referred in 1st pass partitioning
    int m_second_pass_bits[MAX_NUM_BUILD_TABLES]; //number of bits referred in 2nd pass partitioning
    int m_total_hash_bits[MAX_NUM_BUILD_TABLES]; //number of bits referred in pass partitioning

    /*join related parameters*/
    int m_num_res_attrs;
    int m_num_attr_idxes_in_iRes;
    vector<bool*> m_used_for_compare;
    vector<AttrType> m_attr_idxes_in_iRes;
    vector<CntType> m_buckets;
    ResChunk<DataType,CntType> **m_join_res; //join results data

    /* stats of the thread */
    pthread_barrier_t *m_barrier; //barrier struct
    int32_t m_tid; //tid
    int     m_nthreads; //total number of threads
} __attribute__((aligned(CACHE_LINE_SIZE)));

/** holds arguments passed for partitioning */
template<typename DataType, typename CntType>
struct part_t {
    Relation<DataType,CntType> *origin;
    HashTable<DataType,CntType> *tmp_hash_table;
    CntType offset;
    CntType **hist;
    CntType *output;
    TBJ_arg_t<DataType,CntType>* thrargs;
    uint32_t   total_tuples;
    CntType   num_tuples;
    int   first_pass_bits;
    int   second_pass_bits;
} __attribute__((aligned(CACHE_LINE_SIZE)));

/* hash probe function */
template<typename DataType, typename CntType, bool single_join_key>
bool probe_tbj(vector<DataType> iRes, HashTable<DataType,CntType> hash_table,
               CntType &buc_start, CntType buc_end, bool *used_for_compare, CntType &iterator) {
    for(auto i = buc_start; i < buc_end; i ++) {
        bool is_chosen = true;
        if (!single_join_key) {
            for(auto a = 0; a < hash_table.origin->num_attrs; a++) {
                if (used_for_compare[a]) {
                    auto origin_idx = hash_table.hash_tuples[i].idx_in_origin;
                    if (iRes[hash_table.origin->attr_list[a]] != hash_table.origin->data[a][origin_idx]) {
                        is_chosen = false;
                        break;
                    }
                }
            }
        }
        if (((!single_join_key) && (is_chosen)) ||
            ((single_join_key) && (iRes[hash_table.hash_attr] == hash_table.hash_tuples[i].hash_key))) {
            iterator = i;
            buc_start = i+1;
            return true; //a match is found
        }
    }
    return false; //no match is found
}

/* serial radix shuffling performed by each thread independently */
template<typename DataType, typename CntType>
void serial_radix_partition(task_t<DataType,CntType> *const task) {
    const auto fanout = 1 << (task->D);
    auto *hist = (CntType*)calloc(fanout+1, sizeof(CntType));
    auto MASK = fanout - 1;
    auto *input = task->input;
    auto *output = task->output;
    auto length = task->length;
    auto task_start_pos = task->task_start_pos;
    auto *task_his_start = task->task_his_start;
    CntType dst[fanout];
    CntType offset = 0;

    for(CntType i = 0; i < length; i++) { /* count tuples per cluster */
        auto idx = HASH_BIT_MODULO(input[i].hash_key, MASK, 0);
        hist[idx]++;
    }
    for (CntType i = 0; i < fanout; i++) { /* prefix scan */
        dst[i] = offset;
        offset += hist[i];
        task_his_start[i] = dst[i] + task_start_pos; /*complete the final start positions*/
    }
    for(CntType i = 0; i < length; i++){ /* shuffle */
        auto idx = HASH_BIT_MODULO(input[i].hash_key, MASK, 0);
        output[dst[idx]] = input[i];
        ++dst[idx];
    }
    free(hist);
}

/* parallel radix shuffling performed by all threads cooperatively */
template<typename DataType, typename CntType>
void parallel_radix_partition(part_t<DataType,CntType> *const part) {
    auto offset = part->offset;
    const DataType* keys = part->origin->data[0] + part->offset;
    auto **hist = part->hist;
    auto *output = part->output;
    const auto my_tid = part->thrargs->m_tid;
    const auto nthreads = part->thrargs->m_nthreads;
    const auto num_tuples = part->num_tuples;
    const auto R = part->first_pass_bits;
    const auto D = part->second_pass_bits;
    const auto fanout = (uint32_t) (1 << R);
    const auto MASK = (fanout - 1) << D;
    CntType dst[fanout+1];
    int rv;

    /* compute local histogram for the assigned region of origin */
    auto *my_hist = hist[my_tid];
    for(CntType i = 0; i < num_tuples; i++) { /* compute histogram */
        auto idx = HASH_BIT_MODULO(keys[i], MASK, D);
        my_hist[idx]++;
    }
    CntType sum = 0;
    for(CntType i = 0; i < fanout; i++) { /* compute local prefix sum on hist */
        sum += my_hist[i];
        my_hist[i] = sum;
    }
    BARRIER_ARRIVE(part->thrargs->m_barrier, rv);//wait at a barrier until each thread complete histograms

    /* determine the start and end of each cluster */
    for(auto i = 0; i < my_tid; i++) {
        for(auto j = 0; j < fanout; j++) {
            output[j] += hist[i][j];
        }
    }
    for(auto i = my_tid; i < nthreads; i++) {
        for(auto j = 1; j < fanout; j++) {
            output[j] += hist[i][j-1];
        }
    }
    for(CntType i = 0; i < fanout; i++) dst[i] = output[i];
    output[fanout] = part->total_tuples;

    /* Copy tuples to their corresponding clusters */
    auto *tmp_hash_table = part->tmp_hash_table;
    for(CntType i = 0; i < num_tuples; i++){
        auto idx = HASH_BIT_MODULO(keys[i], MASK, D);
        tmp_hash_table->hash_tuples[dst[idx]].hash_key = keys[i];
        tmp_hash_table->hash_tuples[dst[idx]].idx_in_origin = i + offset;
        ++dst[idx];
    }
}

template <typename DataType, typename CntType, bool single_join_key>
void probe_write(TBJ_arg_t<DataType,CntType> *args) {
    auto tid = args->m_tid;
    auto probe_table = *(args->m_probe_table);
    auto probe_offset = args->m_probe_offset;
    auto probe_part_length = args->m_probe_table_part_len;
    auto **hash_tables = args->m_final_hash_tables;
    auto num_hash_tables = args->m_num_hash_tables;
    auto num_res_attrs = args->m_num_res_attrs;
    auto bucket_vec = args->m_buckets;
    auto used_for_compare = args->m_used_for_compare;
    auto num_attr_idxes_in_iRes = args->m_num_attr_idxes_in_iRes;
    auto attr_idxes_in_iRes = args->m_attr_idxes_in_iRes;
    auto &res_chain = args->m_join_res[tid];
    CntType probe_iter = 0;

    vector<uint32_t> masks;
    for(auto i = 0; i < num_hash_tables; i++) {
        masks.emplace_back((1 << args->m_total_hash_bits[i]) - 1);
    }

    vector<CntType> iterators(num_hash_tables);
    vector<CntType> buc_start(num_hash_tables, 0);
    vector<CntType> buc_end(num_hash_tables, 0);
    vector<DataType> iRes((size_t)num_res_attrs);

    char cur_table = 0;
    while (true) {
        if ((0 == cur_table) && (buc_start[0] >= buc_end[0])) { //get new probe item
            auto cur_probe_pos = probe_offset + probe_iter;
            if (probe_iter >= probe_part_length) return; //return
            probe_iter++;
            for(auto i = 0; i < probe_table.num_attrs; i++) { //update iRes
                iRes[probe_table.attr_list[i]] = probe_table.data[i][cur_probe_pos];
            }
            auto hash_val = iRes[hash_tables[0]->hash_attr] & (bucket_vec[0] - 1); //todo: opt bucketVec
            buc_start[0] = hash_tables[0]->buc_ptrs[hash_val]; //update buc_start and buc_end
            buc_end[0] = hash_tables[0]->buc_ptrs[hash_val+1];
        }
        if (cur_table == num_hash_tables - 1) { //reach the last table
            for(auto j = buc_start[cur_table]; j < buc_end[cur_table]; j ++) {
                bool is_chosen = true;
                if (!single_join_key) {
                    for(auto a = 0; a < hash_tables[cur_table]->origin->num_attrs; a++) {
                        if (used_for_compare[cur_table][a]) {
                            auto origin_idx = hash_tables[cur_table]->hash_tuples[j].idx_in_origin;
                            if (iRes[hash_tables[cur_table]->origin->attr_list[a]]
                                != hash_tables[cur_table]->origin->data[a][origin_idx]) {
                                is_chosen = false;
                                break;
                            }
                        }
                    }
                }
                if (((!single_join_key) && (is_chosen)) ||
                    ((single_join_key) && (iRes[hash_tables[cur_table]->hash_attr] == hash_tables[cur_table]->hash_tuples[j].hash_key))) {
                    auto origin_idx = hash_tables[cur_table]->hash_tuples[j].idx_in_origin;

                    /*check whether current chunk is full*/
                    if (res_chain->num_res >= RES_PER_CHUNK) {
                        auto *new_chunk = new ResChunk<DataType,CntType>(num_res_attrs);
                        new_chunk->next = res_chain;
                        res_chain = new_chunk;
                    }
                    #pragma unroll
                    for(auto p = 0; p < num_attr_idxes_in_iRes; p++) //write out vals in iRes
                        res_chain->data[attr_idxes_in_iRes[p]][res_chain->num_res] = iRes[attr_idxes_in_iRes[p]];
                    for(auto p = 0; p < hash_tables[cur_table]->origin->num_attrs; p++)
                        if (!used_for_compare[cur_table][p]) //this attr only appears in the last ht
                            res_chain->data[hash_tables[cur_table]->origin->attr_list[p]][res_chain->num_res] = hash_tables[cur_table]->origin->data[p][origin_idx];
                    res_chain->num_res++;
                }
            }
            cur_table--;
            continue;
        }
        else {
            auto found = probe_tbj<DataType,CntType,single_join_key>(
                    iRes, *hash_tables[cur_table],
                    buc_start[cur_table], buc_end[cur_table],
                    used_for_compare[cur_table], iterators[cur_table]);
            if (!found) { //no match is found
                if (cur_table > 0)  cur_table--;  //backtrack to the last attribute
                else                buc_start[0] = buc_end[0]; //finish this probe item
                continue;
            }
        }

        /*write iRes*/
        auto curIter = iterators[cur_table]; //msIdx is used here
        auto rel = hash_tables[cur_table]->origin;
        auto idx_in_origin_table = hash_tables[cur_table]->hash_tuples[curIter].idx_in_origin;
        for(auto i = 0; i < hash_tables[cur_table]->origin->num_attrs; i++) {
            if (!used_for_compare[cur_table][i]) {
                iRes[rel->attr_list[i]] = rel->data[i][idx_in_origin_table];
            }
        }

        /*update the start and end of next attr*/
        auto hash_val = iRes[hash_tables[cur_table+1]->hash_attr] & (bucket_vec[cur_table+1]- 1);
        buc_start[cur_table+1] = hash_tables[cur_table+1]->buc_ptrs[hash_val];
        buc_end[cur_table+1] = hash_tables[cur_table+1]->buc_ptrs[hash_val+1];
        cur_table++; //advance to the next attr
    }
}

template<typename DataType, typename CntType>
void *prj_thread(void * param) {
    Timer t;
    auto *args   = (TBJ_arg_t<DataType,CntType>*) param;
    auto my_tid = args->m_tid;
    auto nthreads = args->m_nthreads;
    auto num_hash_tables = args->m_num_hash_tables;
    auto num_res_attrs = args->m_num_res_attrs;
    auto **res = args->m_join_res;
    int rv;

    part_t<DataType,CntType> part;
    task_t<DataType,CntType> *task;
    auto *task_queue = args->m_part_queue;

    auto **total_start_pos = args->m_buc_start_pos;
    CntType *outputs[MAX_NUM_BUILD_TABLES];

    for(auto i = 0; i < num_hash_tables; i++) {
        auto fanout_first_pass = 1 << (args->m_first_pass_bits[i]);
        outputs[i] = (CntType *) calloc((fanout_first_pass+1), sizeof(CntType));
        MALLOC_CHECK(outputs[i]);
        args->m_build_hists[i][my_tid] = (CntType*)calloc(fanout_first_pass, sizeof(CntType));
    }
    BARRIER_ARRIVE(args->m_barrier, rv); //synchronization

    if(my_tid == 0) t.reset();
    /* 1st-pass multi-pass partitioning */
    for(int i = 0; i < num_hash_tables; i++) { /*partitioning for individual build table */
        part.origin             = args->m_build_tables[i]; //input partition for this thread
        part.offset             = args->m_build_offset[i];
        part.tmp_hash_table     = args->m_tmp_hash_tables[i]; //output space for all threads
        part.hist               = args->m_build_hists[i];
        part.output             = outputs[i];    //bucket start positions
        part.num_tuples         = args->m_build_part_len[i];
        part.total_tuples       = part.origin->length;
        part.first_pass_bits    = args->m_first_pass_bits[i];
        part.second_pass_bits   = args->m_second_pass_bits[i];
        part.thrargs            = args;
        parallel_radix_partition(&part);
    }
    BARRIER_ARRIVE(args->m_barrier, rv); //synchronization

    /* create partitioning tasks for 2nd pass */
    if(my_tid == 0) {
        auto *task_queue = args->m_part_queue;
        for(auto i = 0; i < num_hash_tables; i++) {
            auto fanout_first_pass = 1 << (args->m_first_pass_bits[i]);
            auto fanout_second_pass = 1 << (args->m_second_pass_bits[i]);
            for(auto k = 0; k < fanout_first_pass; k++) {
                auto *t = task_queue->task_queue_get_slot();
                t->length = outputs[i][k+1] - outputs[i][k];
                t->input = args->m_tmp_hash_tables[i]->hash_tuples + outputs[i][k];
                t->output = args->m_final_hash_tables[i]->hash_tuples + outputs[i][k];
                t->task_start_pos = outputs[i][k];
                t->task_his_start = &total_start_pos[i][k*fanout_second_pass];
                t->D = args->m_second_pass_bits[i];
                total_start_pos[i][fanout_first_pass*fanout_second_pass] = outputs[i][fanout_first_pass];
                task_queue->task_queue_add(t);
            }
        }
        log_info("Pass-2: #tasks = %llu", task_queue->queue_list->count);
    }
    BARRIER_ARRIVE(args->m_barrier, rv); //synchronization

    /* 2nd-pass multi-pass partitioning with dynamic task scheduling*/
    while((task = task_queue->task_queue_get_atomic())) {
        serial_radix_partition<DataType,CntType>(task);
    }
    BARRIER_ARRIVE(args->m_barrier, rv); //synchronization

    for(auto i = 0; i < num_hash_tables; i++) free(outputs[i]);

    /*write up the buc_ptrs of the hash tables*/
    auto **hash_tables = args->m_final_hash_tables;
    for(auto i = my_tid; i < num_hash_tables; i += nthreads) {
        memcpy(hash_tables[i]->buc_ptrs, total_start_pos[i], sizeof(CntType)*(hash_tables[i]->buckets+1));
    }
    BARRIER_ARRIVE(args->m_barrier, rv); //synchronization

    if(my_tid == 0) { /* record partitioning time*/
        log_info("Partitioning time: %.2f ms", t.elapsed()*1000);
        t.reset();
    }

    /* Probe write */
    probe_write<DataType,CntType,true>(args);
    BARRIER_ARRIVE(args->m_barrier, rv); //synchronization

    if(my_tid == 0) { /* record partitioning time*/
        log_info("Probe-write time: %.2f ms", t.elapsed()*1000);
        t.reset();
    }
    return 0;
}

template <typename DataType, typename CntType>
CntType TBJ(vector<Relation<DataType, CntType>> tables,
         vector<bool*> used_for_compare, vector<CntType> buckets,
         vector<AttrType> attr_idxes_in_iRes, int num_attr_idxes_in_iRes,
         int num_res_attrs, int nthreads) {
    log_info("In function: %s", __FUNCTION__);
    int i, rv;
    pthread_t tid[nthreads];
    pthread_attr_t attr;
    pthread_barrier_t barrier;
    TBJ_arg_t<DataType,CntType> args[nthreads];

    auto probe_table = tables[0];
    auto *build_tables = &tables[1];
    uint32_t num_hash_tables = (uint32_t)tables.size()-1;

    CntType **vec_hists[MAX_NUM_BUILD_TABLES];
    HashTable<DataType,CntType> hash_table_1st[MAX_NUM_BUILD_TABLES];
    HashTable<DataType,CntType> hash_table_2nd[MAX_NUM_BUILD_TABLES];

    CntType num_build_items_per_thread[MAX_NUM_BUILD_TABLES];
    CntType num_probe_items_per_thread = probe_table.length/nthreads;

    TaskQueue<DataType,CntType> part_queue(16);//task queue for radix partitioning
    CntType *buc_start_pos[MAX_NUM_BUILD_TABLES];//start positions of the buckets

    /*join output*/
    ResChunk<DataType,CntType> **res = new ResChunk<DataType,CntType>*[nthreads]; //result linked list, each thread has one chain
    for(auto i = 0; i < nthreads; i++) { //init the first chunk for each thread
        res[i] = new ResChunk<DataType,CntType>(num_res_attrs);
    }
    CntType num_res = 0;

    vector<int> bucket_bits_1st_pass, bucket_bits_2nd_pass;
    for(int i = 0; i < num_hash_tables; i++) { /* allocate temporary space for partitioning */
        /*each thread will allocate its own histogram memory space*/
        vec_hists[i] = (CntType**)alloc_aligned(nthreads * sizeof(CntType*));
        MALLOC_CHECK(vec_hists[i]);

        int bucket_bits = logFunc(buckets[i]);
        assert(bucket_bits >= 0);
        bucket_bits_1st_pass.emplace_back(bucket_bits/2);
        bucket_bits_2nd_pass.emplace_back(bucket_bits - bucket_bits/2);

        hash_table_1st[i].init(build_tables[i].length, 1<<bucket_bits_1st_pass[i]);
        hash_table_2nd[i].init(build_tables[i].length, buckets[i]);
        hash_table_2nd[i].origin = &build_tables[i];
        hash_table_2nd[i].hash_attr = hash_table_2nd[i].origin->attr_list[0];

        /*final bucket start positions*/
        buc_start_pos[i] = (CntType*)calloc(buckets[i]+1, sizeof(CntType));
        num_build_items_per_thread[i] = build_tables[i].length/nthreads;
    }
    rv = pthread_barrier_init(&barrier, NULL, nthreads);
    if(rv != 0){
        printf("[ERROR] Couldn't create the barrier\n");
        exit(EXIT_FAILURE);
    }
    pthread_attr_init(&attr);

    /*assign slices of probe and build tables to each thread*/
    for(i = 0; i < nthreads; i++) {
        /*set the number of bits referred in the two passes*/
        for(auto r = 0; r < num_hash_tables; r++) {
            args[i].m_total_hash_bits[r] = bucket_bits_1st_pass[r] + bucket_bits_2nd_pass[r];
            args[i].m_first_pass_bits[r] = bucket_bits_1st_pass[r];
            args[i].m_second_pass_bits[r] = bucket_bits_2nd_pass[r];
        }

        /*pass the probe table parameters*/
        args[i].m_probe_table = &probe_table;
        args[i].m_probe_offset = i * num_probe_items_per_thread;
        args[i].m_probe_table_part_len = (i == (nthreads-1)) ?
                (probe_table.length - i * num_probe_items_per_thread) : num_probe_items_per_thread;

        /*pass the build table parameters*/
        args[i].m_num_hash_tables = num_hash_tables;
        for(int k = 0; k < num_hash_tables; k++) {
            args[i].m_build_offset[k] = i*num_build_items_per_thread[k];
            args[i].m_build_tables[k] = &build_tables[k];
            args[i].m_tmp_hash_tables[k] = &hash_table_1st[k];
            args[i].m_final_hash_tables[k] = &hash_table_2nd[k];
            args[i].m_build_hists[k] = vec_hists[k];
            args[i].m_build_part_len[k] = (i == (nthreads-1)) ?
               (build_tables[k].length - i * num_build_items_per_thread[k]) : num_build_items_per_thread[k];
        }
        /*pass the join related parameters*/
        args[i].m_num_res_attrs = num_res_attrs;
        args[i].m_num_attr_idxes_in_iRes = num_attr_idxes_in_iRes;
        args[i].m_used_for_compare = used_for_compare;
        args[i].m_attr_idxes_in_iRes = attr_idxes_in_iRes;
        args[i].m_buckets = buckets;
        args[i].m_join_res = res;

        /*pass the other parameters*/
        args[i].m_tid = i;
        args[i].m_part_queue = &part_queue;
        args[i].m_buc_start_pos = buc_start_pos;
        args[i].m_barrier = &barrier;
        args[i].m_nthreads = nthreads;

        /*create threads*/
        rv = pthread_create(&tid[i], &attr, prj_thread<DataType,CntType>, (void*)&args[i]);
        if (rv) {
            log_error("Return code from pthread_create() is %d", rv);
            exit(-1);
        }
    }

    /* wait for threads to finish */
    for(i = 0; i < nthreads; i++) {
        pthread_join(tid[i], nullptr);
    }

    /*compute the number of join outputs*/
    for(auto i = 0; i < nthreads; i++) {
        auto *head = res[i];
        while (head != nullptr) {
            num_res += head->num_res;
            head = head->next;
        }
    }
    return num_res;
}

template<typename DataType, typename CntType>
void load_rels(string dir_name, vector<Relation<DataType,CntType>> &relations, uint32_t &num_tables) {
    auto file_list = list_files(dir_name.c_str());
    if (0 == file_list.size()) {
        log_error("The directory is empty");
        exit(1);
    }
    num_tables = (uint32_t)file_list.size();

    /*read the relations from files*/
    relations.resize(num_tables);
    for(auto i = 0; i < num_tables; i++) {
        string cur_file_name = file_list[i].substr(0,file_list[i].length()-rel_prefix.size());
        vector<string> properties;
        split_string(cur_file_name, properties, "_");
        auto data_from_file = read_rel_cols_mmap_CPU<DataType,CntType>(
                (dir_name+"/"+file_list[i]).c_str(),
                (int)properties.size() - 1, stoull(properties[0]));

        int cur_idx;
        log_no_newline("Enter the table order: ");
//        cin>>cur_idx; //input table order
        cur_idx = i;
        assert(cur_idx < num_tables);

        relations[cur_idx].init((uint32_t)properties.size() - 1, stoull(properties[0]));
        for(auto a = 0; a < relations[cur_idx].num_attrs; a++) {
            relations[cur_idx].attr_list[a] = (AttrType)stoi(properties[a+1]);
            relations[cur_idx].data[a] = data_from_file[a];
        }
    }
}

template<typename DataType, typename CntType>
void test_general(string dir_name) {
    auto nthreads = omp_get_max_threads(); //use all the threads
    log_info("use %d threads", nthreads);

    vector<Relation<DataType,CntType>> relations;
    vector<HashTable<DataType,CntType>> hash_tables;
    vector<CntType> buckets;
    uint32_t num_tables;
    Timer t;

    /*load data from disk*/
    load_rels(dir_name, relations, num_tables);
    uint32_t num_hash_tables = num_tables - 1;

    /*analyze the query datasets*/
    int num_attrs_in_res = 0; //number of output attrs
    int num_attr_idxes_in_iRes = 0; //number of output attrs in iRes
    vector<AttrType> attr_idxes_in_iRes(MAX_NUM_RES_ATTRS);

    /*compute used_for_compare*/
    vector<bool*> used_for_compare(num_hash_tables);
    bool attr_referred[MAX_NUM_RES_ATTRS] = {false};
    for(auto i = 0; i < num_tables; i++) { /*compute used_for_compare array*/
        if (0 != i) {
            used_for_compare[i-1] = (bool*)malloc(sizeof(bool)*relations[i].num_attrs);
        }
        for(auto a = 0; a < relations[i].num_attrs; a++) {
            if (!attr_referred[relations[i].attr_list[a]]) {//this attr has not shown in previous relations
                attr_referred[relations[i].attr_list[a]] = true;
                num_attrs_in_res++;
                if (0 != i) used_for_compare[i-1][a] = false;
                if (i != num_tables -1) attr_idxes_in_iRes[num_attr_idxes_in_iRes++] = relations[i].attr_list[a];
            }
            else if (0 != i) used_for_compare[i-1][a] = true;
        }
    }
    auto *build_tables = &relations[1];
    for(auto i = 0; i < num_hash_tables; i++) { /*compute bucket_vec*/
        buckets.emplace_back(floorPowerOf2(build_tables[i].length / BUC_RATIO));
    }

    /*build and probe with multi-thread*/
    t.reset();
    auto num_res = MHJ<DataType,CntType>(relations, used_for_compare, buckets,
                          attr_idxes_in_iRes, num_attr_idxes_in_iRes,
                          num_attrs_in_res, nthreads);
    log_info("Number of join results: %llu", num_res);
    log_info("Total CPU execution time: %.2f ms", t.elapsed()*1000);
}

/* usage:
 * ./tbj DATA_DIR
 * */
int main(int argc, char *argv[]) {
    assert(argc == 2);
    test_general<KeyType,CarType>(string(argv[1]));
    return 0;
}

