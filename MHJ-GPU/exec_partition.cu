#include "Indexing/radix_partitioning.cuh"
#include "Indexing/linear.cuh"
#include "Indexing/cuckoo_radix.cuh"
#include "Indexing/cuckoo_direct.cuh"
#include "Indexing/cudpp.cuh"

#include "log.h"
#include "file_io.h"
using namespace std;

#define BUC_RATIO (4)

#define BUILD_EXPER_TIMES   (10)
#define PROBE_EXPER_TIMES   (100)

enum BuildHashType {
    RP_2,
    RP_MUL,
    OPEN_ADDR_05,
    RADIX_SORT_KEYS,
    RADIX_SORT_KVS,
    CUCKOO_05,

    CUCKOO_DIR_05,
    CUCKOO_DIR_095,
    CUDPP_05, CUDPP_095,
    OPEN_ADDR_095
};

#define GALLOPING_SEARCH

/*binary search with galloping search*/
template<typename KType, typename VType, typename CntType>
__global__ void binary_lookup_KV(
        KType *ht_keys, VType *ht_values, CntType ht_capacity,
        KType *lookup_keys, VType *lookup_values, CntType num) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num) {
        auto key = lookup_keys[tid];
#ifdef GALLOPING_SEARCH
        int64_t lo = 0;
        int64_t hi = 0;
        int64_t scale = 8;
        while ((hi < ht_capacity) && (ht_keys[hi] < key)) {
            lo = hi;
            hi += scale;
            scale <<= 3;
        }
        if (hi > ht_capacity-1) hi = ht_capacity-1;

        while (lo <= hi) {
            scale = lo + (hi - lo)/2;
            if (key > ht_keys[scale])
                lo = scale + 1;
            else
                hi = scale - 1;
        }
        lookup_values[tid] = ht_values[lo];
#else
        /*common binary search*/
        int64_t start = 0, end = ht_capacity -1;
        int64_t middle;
        while (start <= end) {
            middle = start + (end - start)/2;
            if (ht_keys[middle] < key)
                start = middle + 1;
            else if (ht_keys[middle] > key)
                end = middle - 1;
            else {
                lookup_values[tid] = ht_values[middle];
                return;
            }
        }
#endif
    }
}

#undef GALLOPING_SEARCH

template<typename KType, typename VType, typename CntType>
void binary_lookup_vals(KType *ht_keys, VType *ht_values, CntType ht_capacity,
                        KType *lookup_keys, VType *lookup_values, CntType num, CUDATimeStat *timing) {
    int mingridsize, block_size;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &block_size,
                                       binary_lookup_KV<KType,VType,CntType>, 0, 0);
    auto grid_size = (num + block_size - 1) / block_size;
    cudaDeviceSynchronize();
    execKernel((binary_lookup_KV<uint32_t,uint32_t,uint32_t>), grid_size, block_size, timing, false, ht_keys, ht_values, ht_capacity, lookup_keys, lookup_values, num);
    cudaDeviceSynchronize();
}

void test_build_hashtables(string file_addr, BuildHashType build_type) {
    Timer t;
    CUDATimeStat timing;
    vector<string> key_attr_strs;
    string file_name = file_cut_subfix(get_file_name_from_addr(file_addr));
    split_string(file_name, key_attr_strs, "_");
    auto cnt = stoul(key_attr_strs[0].c_str());
    int num_cols = (int)key_attr_strs.size()-1;

    Relation<uint32_t,uint32_t> *rel = nullptr;
    CUDA_MALLOC(&rel, sizeof(Relation<uint32_t,uint32_t>), nullptr);
    rel->init(num_cols, stoull(key_attr_strs[0]), nullptr);

    /*load data from disk*/
    auto data_out = read_rel_cols_mmap<uint32_t,uint32_t>(file_addr.c_str(), num_cols, cnt);
    for(auto i = 0; i < num_cols; i++) {
        rel->attr_list[i] = (AttrType)stoi(key_attr_strs[i+1]);
        rel->data[i] = data_out[i];
        checkCudaErrors(cudaMemPrefetchAsync(data_out[i], sizeof(uint32_t)*cnt, DEVICE_ID));
    }
    cudaDeviceSynchronize();

    /*init the input and output data*/
    auto hash_keys_input = data_out[0];
    uint32_t *hash_keys_output = nullptr;
    uint32_t *hash_values_input = nullptr;
    uint32_t *hash_values_output = nullptr;

    uint32_t *probe_keys = hash_keys_input;
    uint32_t *probe_values = nullptr;

    CUDA_MALLOC(&hash_values_input, sizeof(uint32_t)*cnt, nullptr);
    thrust::counting_iterator<uint32_t> iter(0);
    thrust::copy(iter, iter + cnt, hash_values_input);
    CUDA_MALLOC(&probe_values, sizeof(uint32_t)*cnt, nullptr);

    float smallest_build_gpu_time = 99999, smallest_build_cpu_time = 99999;
    float smallest_probe_gpu_time = 99999, smallest_probe_cpu_time = 99999;

    switch (build_type) {
        case RP_2: {
            log_info("Case: RP_2");
            auto buckets = cnt / BUC_RATIO;
            uint32_t *buc_ptrs = nullptr;
            CUDA_MALLOC(&buc_ptrs, sizeof(uint32_t)*(buckets+1), nullptr);
            CUDA_MALLOC(&hash_keys_output, sizeof(uint32_t)*cnt, nullptr);
            CUDA_MALLOC(&hash_values_output, sizeof(uint32_t)*cnt, nullptr);
            RadixPartitioner<uint32_t, uint32_t, uint32_t> rp(cnt, 2, buckets, nullptr, &timing); //2-pass
            cudaDeviceSynchronize();

            for(auto i = 0; i < BUILD_EXPER_TIMES; i++) {
                auto gpu_time_idx = timing.get_idx();
                t.reset();
                rp.splitKV(hash_keys_input, hash_keys_output,
                           nullptr, hash_values_output, buc_ptrs);
                auto gpu_time = timing.diff_time(gpu_time_idx);
                auto cpu_time = t.elapsed() *1000;
                if (gpu_time < smallest_build_gpu_time) smallest_build_gpu_time = gpu_time;
                if (cpu_time < smallest_build_cpu_time) smallest_build_cpu_time = cpu_time;
            }
            log_info("[Build] RP: kernel time: %.2f ms, CPU time: %.2f ms", smallest_build_gpu_time, smallest_build_cpu_time);

            /*check*/
            auto mask = buckets-1;
            for(auto i = 0; i < cnt-1; i++) {
                auto last_key = hash_keys_output[i] & mask;
                auto next_key = hash_keys_output[i+1] & mask;
                assert(last_key <= next_key);
            }

            log_info("Probing");
            for(auto i = 0; i < PROBE_EXPER_TIMES; i++) {
                auto gpu_time_idx = timing.get_idx();
                t.reset();
                rp.lookup_vals(probe_keys, probe_values, cnt,
                               hash_keys_output, hash_values_output, buckets, buc_ptrs);
                auto gpu_time = timing.diff_time(gpu_time_idx);
                auto cpu_time = t.elapsed() *1000;
                if (gpu_time < smallest_probe_gpu_time) smallest_probe_gpu_time = gpu_time;
                if (cpu_time < smallest_probe_cpu_time) smallest_probe_cpu_time = cpu_time;
            }
            log_info("[Probe] RP: kernel time: %.2f ms, CPU time: %.2f ms", smallest_probe_gpu_time, smallest_probe_cpu_time);
            for(auto i = 0; i < cnt; i++) { //check the results
                assert(probe_values[i]  == hash_values_input[i]);
            }

            CUDA_FREE(hash_keys_output, nullptr);
            CUDA_FREE(hash_values_output, nullptr);
            CUDA_FREE(buc_ptrs, nullptr);

            break;
        }
        case RP_MUL: {
            log_info("Case: RP_MUL");
            auto buckets = cnt / BUC_RATIO;
            uint32_t *buc_ptrs = nullptr;
            CUDA_MALLOC(&buc_ptrs, sizeof(uint32_t)*(buckets+1), nullptr);
            CUDA_MALLOC(&hash_keys_output, sizeof(uint32_t)*cnt, nullptr);
            CUDA_MALLOC(&hash_values_output, sizeof(uint32_t)*cnt, nullptr);
            RadixPartitioner<uint32_t, uint32_t, uint32_t> rp(cnt, buckets, nullptr, &timing); //default bucket planning
            cudaDeviceSynchronize();
            for(auto i = 0; i < BUILD_EXPER_TIMES; i++) {
                auto gpu_time_idx = timing.get_idx();
                t.reset();
                rp.splitKV(hash_keys_input, hash_keys_output,
                           nullptr, hash_values_output, buc_ptrs);
                auto gpu_time = timing.diff_time(gpu_time_idx);
                auto cpu_time = t.elapsed() *1000;
                if (gpu_time < smallest_build_gpu_time) smallest_build_gpu_time = gpu_time;
                if (cpu_time < smallest_build_cpu_time) smallest_build_cpu_time = cpu_time;
            }
            log_info("[Build] RP: kernel time: %.2f ms, CPU time: %.2f ms",
                     smallest_build_gpu_time, smallest_build_cpu_time);

            /*check*/
            auto mask = buckets-1;
            for(auto i = 0; i < cnt-1; i++) {
                auto last_key = hash_keys_output[i] & mask;
                auto next_key = hash_keys_output[i+1] & mask;
                assert(last_key <= next_key);
            }

            log_info("Probing");
            for(auto i = 0; i < PROBE_EXPER_TIMES; i++) {
                auto gpu_time_idx = timing.get_idx();
                t.reset();
                rp.lookup_vals(probe_keys, probe_values, cnt,
                               hash_keys_output, hash_values_output, buckets, buc_ptrs);
                auto gpu_time = timing.diff_time(gpu_time_idx);
                auto cpu_time = t.elapsed() *1000;
                if (gpu_time < smallest_probe_gpu_time) smallest_probe_gpu_time = gpu_time;
                if (cpu_time < smallest_probe_cpu_time) smallest_probe_cpu_time = cpu_time;
            }
            log_info("[Probe] RP: kernel time: %.2f ms, CPU time: %.2f ms", smallest_probe_gpu_time, smallest_probe_cpu_time);
            for(auto i = 0; i < cnt; i++) { //check the results
                assert(probe_values[i]  == hash_values_input[i]);
            }

            CUDA_FREE(hash_keys_output, nullptr);
            CUDA_FREE(hash_values_output, nullptr);
            CUDA_FREE(buc_ptrs, nullptr);

            break;
        }
        case OPEN_ADDR_095: {
            log_info("Case: OPEN_ADDR_095");
            float ratio = 0.95;
            uint64_t capacity = (uint64_t)(cnt / ratio);
            log_info("cnt: %llu, capacity: %llu", cnt, capacity);
            LinearHashTable<uint32_t, uint32_t, uint32_t> linear_ht(capacity, nullptr, &timing);
            cudaDeviceSynchronize();

            for(auto i = 0; i < BUILD_EXPER_TIMES; i++) {
                auto gpu_time_idx = timing.get_idx();
                t.reset();
                linear_ht.insert_vals(hash_keys_input, hash_values_input, cnt);
                auto gpu_time = timing.diff_time(gpu_time_idx);
                auto cpu_time = t.elapsed() *1000;
                if (gpu_time < smallest_build_gpu_time) smallest_build_gpu_time = gpu_time;
                if (cpu_time < smallest_build_cpu_time) smallest_build_cpu_time = cpu_time;
            }
            log_info("[Build] OA: kernel time: %.2f ms, CPU time: %.2f ms",
                     smallest_build_gpu_time, smallest_build_cpu_time);

            log_info("Probing");
            for(auto i = 0; i < PROBE_EXPER_TIMES; i++) {
                auto gpu_time_idx = timing.get_idx();
                t.reset();
                linear_ht.lookup_vals(probe_keys, probe_values, cnt);
                auto gpu_time = timing.diff_time(gpu_time_idx);
                auto cpu_time = t.elapsed() *1000;
                if (gpu_time < smallest_probe_gpu_time) smallest_probe_gpu_time = gpu_time;
                if (cpu_time < smallest_probe_cpu_time) smallest_probe_cpu_time = cpu_time;
            }
            log_info("[Probe] OA: kernel time: %.2f ms, CPU time: %.2f ms",
                     smallest_probe_gpu_time, smallest_probe_cpu_time);
            for(auto i = 0; i < cnt; i++) { //check the results
                assert(probe_values[i]  == hash_values_input[i]);
            }
            break;
        }
        case OPEN_ADDR_05: {
            log_info("Case: OPEN_ADDR_05");
            float ratio = 0.5;
            uint64_t capacity = (uint64_t)(cnt / ratio);
            log_info("cnt: %llu, capacity: %llu", cnt, capacity);
            LinearHashTable<uint32_t, uint32_t, uint32_t> linear_ht(capacity, nullptr, &timing);
            cudaDeviceSynchronize();

            for(auto i = 0; i < BUILD_EXPER_TIMES; i++) {
                /*reset the hash table every time*/
                checkCudaErrors(cudaMemset(linear_ht._keys, 0xff, sizeof(uint32_t) * linear_ht._table_card));
                auto gpu_time_idx = timing.get_idx();
                t.reset();
                linear_ht.insert_vals(hash_keys_input, hash_values_input, cnt);
                auto gpu_time = timing.diff_time(gpu_time_idx);
                auto cpu_time = t.elapsed() *1000;
                if (gpu_time < smallest_build_gpu_time) smallest_build_gpu_time = gpu_time;
                if (cpu_time < smallest_build_cpu_time) smallest_build_cpu_time = cpu_time;
            }
            log_info("[Build] OA: kernel time: %.2f ms, CPU time: %.2f ms",
                     smallest_build_gpu_time, smallest_build_cpu_time);

            log_info("Probing");
            for(auto i = 0; i < PROBE_EXPER_TIMES; i++) {
                auto gpu_time_idx = timing.get_idx();
                t.reset();
                linear_ht.lookup_vals(probe_keys, probe_values, cnt);
                auto gpu_time = timing.diff_time(gpu_time_idx);
                auto cpu_time = t.elapsed() *1000;
                if (gpu_time < smallest_probe_gpu_time) smallest_probe_gpu_time = gpu_time;
                if (cpu_time < smallest_probe_cpu_time) smallest_probe_cpu_time = cpu_time;
            }
            log_info("[Probe] OA: kernel time: %.2f ms, CPU time: %.2f ms",
                     smallest_probe_gpu_time, smallest_probe_cpu_time);
            for(auto i = 0; i < cnt; i++) { //check the results
                assert(probe_values[i]  == hash_values_input[i]);
            }
            break;
        }

        case CUDPP_05: {
            log_info("Case: CUDPP_05");
            float ratio = 0.5;

            /*
             * CUDPP hash table setting:
             * https://cudpp.github.io/cudpp/2.2/hash_overview.html
             * */
            //CUDPP ratio is defined differently from ours
            //CUDPP_MULTIVALUE_HASH_TABLE will stuck on V100
            CudppHashTable<uint32_t, uint32_t, uint32_t> cudpp_ht(CUDPP_BASIC_HASH_TABLE, cnt, 1.0/ratio);
            cudaDeviceSynchronize();

            auto gpu_time_idx = timing.get_idx();
            for(auto i = 0; i < BUILD_EXPER_TIMES; i++) {
                auto gpu_time_idx = timing.get_idx();
                t.reset();
                cudpp_ht.insert_vals(hash_keys_input, hash_values_input, cnt);
                cudaDeviceSynchronize();
                auto gpu_time = timing.diff_time(gpu_time_idx);
                auto cpu_time = t.elapsed() *1000;
                if (gpu_time < smallest_build_gpu_time) smallest_build_gpu_time = gpu_time;
                if (cpu_time < smallest_build_cpu_time) smallest_build_cpu_time = cpu_time;
            }
            log_info("[Build] CUDPP: kernel time: %.2f ms, CPU time: %.2f ms",
                     smallest_build_gpu_time, smallest_build_cpu_time);

            log_info("Probing");
            for(auto i = 0; i < PROBE_EXPER_TIMES; i++) {
                auto gpu_time_idx = timing.get_idx();
                t.reset();
                cudpp_ht.lookup_vals(probe_keys, probe_values, cnt);
                auto gpu_time = timing.diff_time(gpu_time_idx);
                auto cpu_time = t.elapsed() *1000;
                if (gpu_time < smallest_probe_gpu_time) smallest_probe_gpu_time = gpu_time;
                if (cpu_time < smallest_probe_cpu_time) smallest_probe_cpu_time = cpu_time;
            }

            log_info("[Probe] CUDPP: kernel time: %.2f ms, CPU time: %.2f ms",
                     smallest_probe_gpu_time, smallest_probe_cpu_time);

            for(auto i = 0; i < cnt; i++) { //check the results
                assert(probe_values[i]  == hash_values_input[i]);
            }
            break;
        }
        case CUDPP_095: {
            log_info("Case: CUDPP_095");
            float ratio = 0.95;

            /*
             * CUDPP hash table setting:
             * https://cudpp.github.io/cudpp/2.2/hash_overview.html
             * */
            //CUDPP ratio is defined differently from ours
            //CUDPP_MULTIVALUE_HASH_TABLE will stuck on V100
            CudppHashTable<uint32_t, uint32_t, uint32_t> cudpp_ht(CUDPP_BASIC_HASH_TABLE, cnt, 1.0/ratio);
            cudaDeviceSynchronize();

            auto gpu_time_idx = timing.get_idx();
            t.reset();
            cudpp_ht.insert_vals(hash_keys_input, hash_values_input, cnt);
            cudaDeviceSynchronize();
            log_info("[Build] CUDPP: kernel time: %.2f ms, CPU time: %.2f ms", timing.diff_time(gpu_time_idx), t.elapsed() *1000);

            log_info("Probing");
            gpu_time_idx = timing.get_idx();
            t.reset();
            cudpp_ht.lookup_vals(probe_keys, probe_values, cnt);
            cudaDeviceSynchronize();
            log_info("[Probe] CUDPP: kernel time: %.2f ms, CPU time: %.2f ms", timing.diff_time(gpu_time_idx), t.elapsed() *1000);

            for(auto i = 0; i < cnt; i++) { //check the results
                assert(probe_values[i]  == hash_values_input[i]);
            }
            break;
        }
        case CUCKOO_05: {
            log_info("Case: CUCKOO_05");
            float ratio = 0.5;
            log_info("CUCKOO_HT_RATIO = %.1f", ratio);
            int evict_bound = (int)7*log(cnt);
            CuckooHashTableRadix<uint32_t, uint32_t, uint32_t> cuckoo_ht(cnt, ratio, evict_bound,
                                                                         nullptr, 0, nullptr, &timing);
            cudaDeviceSynchronize();

            int rehash_cnt = 0;
            float cur_gpu_time, cur_cpu_time;
            do { //if rehash happens, count the time again
                auto gpu_time_idx = timing.get_idx();
                t.reset();
                rehash_cnt = cuckoo_ht.insert_kvs(hash_keys_input, hash_values_input, cnt);
                cur_gpu_time = timing.diff_time(gpu_time_idx);
                cur_cpu_time = t.elapsed() *1000;
            } while (rehash_cnt != 0);
            cudaDeviceSynchronize();

            log_info("[Build] Cuckoo: kernel time: %.2f ms, CPU time: %.2f ms", cur_gpu_time, cur_cpu_time);

            log_info("Probing");
            auto gpu_time_idx = timing.get_idx();
            t.reset();
            cuckoo_ht.lookup_vals(probe_keys, probe_values, cnt);
            cudaDeviceSynchronize();
            log_info("[Probe] Cuckoo: kernel time: %.2f ms, CPU time: %.2f ms", timing.diff_time(gpu_time_idx), t.elapsed() *1000);

            for(auto i = 0; i < cnt; i++) { //check the results
                assert(probe_values[i]  == hash_values_input[i]);
            }
            break;
        }
        case CUCKOO_DIR_05: {
            log_info("Case: CUCKOO_DIR_05");
            float ratio = 0.5;
            uint64_t capacity = (uint64_t)(cnt / ratio);
            int num_functions = 4;
            int evict_bound = (int)7*log(cnt);
            log_info("cnt=%llu, capacity=%llu, ratio=%.1f, num funcs=%d, evict_bound=%d",
                     cnt, capacity, ratio, num_functions, evict_bound);
            CuckooHashTableDirect<uint32_t, uint32_t, uint32_t> cuckoo_ht(capacity, evict_bound, num_functions,
                                                                         nullptr, 0, nullptr, &timing);
            cudaDeviceSynchronize();

            for(auto i = 0; i < BUILD_EXPER_TIMES; i++) {
                auto gpu_time_idx = timing.get_idx();
                t.reset();
                cuckoo_ht.insert_vals(hash_keys_input, hash_values_input, cnt);
                auto gpu_time = timing.diff_time(gpu_time_idx);
                auto cpu_time = t.elapsed() *1000;
                if (gpu_time < smallest_build_gpu_time) smallest_build_gpu_time = gpu_time;
                if (cpu_time < smallest_build_cpu_time) smallest_build_cpu_time = cpu_time;
            }
            log_info("[Build] Direct Cuckoo: kernel time: %.2f ms, CPU time: %.2f ms",
                     smallest_build_gpu_time, smallest_build_cpu_time);
            cudaDeviceSynchronize();

            log_info("Probing");
            for(auto i = 0; i < PROBE_EXPER_TIMES; i++) {
                auto gpu_time_idx = timing.get_idx();
                t.reset();
                cuckoo_ht.lookup_vals(probe_keys, probe_values, cnt);
                auto gpu_time = timing.diff_time(gpu_time_idx);
                auto cpu_time = t.elapsed() *1000;
                if (gpu_time < smallest_probe_gpu_time) smallest_probe_gpu_time = gpu_time;
                if (cpu_time < smallest_probe_cpu_time) smallest_probe_cpu_time = cpu_time;
            }
            log_info("[Probe] Direct Cuckoo: kernel time: %.2f ms, CPU time: %.2f ms",
                     smallest_probe_gpu_time, smallest_probe_cpu_time);

            for(auto i = 0; i < cnt; i++) { //check the results
                assert(probe_values[i]  == hash_values_input[i]);
            }
            break;
        }
        case CUCKOO_DIR_095: {
            log_info("Case: CUCKOO_DIR_095");
            float ratio = 0.95;
            uint64_t capacity = (uint64_t)(cnt / ratio);
            int num_functions = 4;
            int evict_bound = (int)7*log(cnt);
            log_info("cnt=%llu, capacity=%llu, ratio=%.1f, num funcs=%d, evict_bound=%d",
                     cnt, capacity, ratio, num_functions, evict_bound);
            CuckooHashTableDirect<uint32_t, uint32_t, uint32_t> cuckoo_ht(capacity, evict_bound, num_functions,
                                                                          nullptr, 0, nullptr, &timing);
            cudaDeviceSynchronize();

            for(auto i = 0; i < BUILD_EXPER_TIMES; i++) {
                auto gpu_time_idx = timing.get_idx();
                t.reset();
                cuckoo_ht.insert_vals(hash_keys_input, hash_values_input, cnt);
                auto gpu_time = timing.diff_time(gpu_time_idx);
                auto cpu_time = t.elapsed() *1000;
                if (gpu_time < smallest_build_gpu_time) smallest_build_gpu_time = gpu_time;
                if (cpu_time < smallest_build_cpu_time) smallest_build_cpu_time = cpu_time;
            }
            log_info("[Build] Direct Cuckoo: kernel time: %.2f ms, CPU time: %.2f ms",
                     smallest_build_gpu_time, smallest_build_cpu_time);
            cudaDeviceSynchronize();

            log_info("Probing");
            for(auto i = 0; i < PROBE_EXPER_TIMES; i++) {
                auto gpu_time_idx = timing.get_idx();
                t.reset();
                cuckoo_ht.lookup_vals(probe_keys, probe_values, cnt);
                auto gpu_time = timing.diff_time(gpu_time_idx);
                auto cpu_time = t.elapsed() *1000;
                if (gpu_time < smallest_probe_gpu_time) smallest_probe_gpu_time = gpu_time;
                if (cpu_time < smallest_probe_cpu_time) smallest_probe_cpu_time = cpu_time;
            }
            log_info("[Probe] Direct Cuckoo: kernel time: %.2f ms, CPU time: %.2f ms",
                     smallest_probe_gpu_time, smallest_probe_cpu_time);

            for(auto i = 0; i < cnt; i++) { //check the results
                assert(probe_values[i]  == hash_values_input[i]);
            }
            break;
        }
        case RADIX_SORT_KEYS: {
            log_info("Case: RADIX_SORT_KEYS");
            void *d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;

            CUDA_MALLOC(&hash_keys_output, sizeof(uint32_t)*cnt, nullptr);
            CUDA_MALLOC(&hash_values_output, sizeof(uint32_t)*cnt, nullptr);
            auto *timing_ptr = &timing;
            cudaDeviceSynchronize();

            /*Sort according to values*/
            auto gpu_time_idx = timing.get_idx();
            t.reset();

            /* Sort according to keys*/
            timingKernel(
                    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, hash_keys_input, hash_keys_output, hash_values_input, hash_values_output, cnt), timing_ptr);
            CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, nullptr);
            timingKernel(
                    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, hash_keys_input, hash_keys_output, hash_values_input, hash_values_output, cnt), timing_ptr);
            cudaDeviceSynchronize();
            log_info("[Build] Sort-K: kernel time: %.2f ms, CPU time: %.2f ms", timing.diff_time(gpu_time_idx), t.elapsed() *1000);

            log_info("Probing");
            gpu_time_idx = timing.get_idx();
            t.reset();
            binary_lookup_vals(hash_keys_output, hash_values_output, cnt, probe_keys, probe_values, cnt, &timing);
            cudaDeviceSynchronize();
            log_info("[Probe] Sort-K: kernel time: %.2f ms, CPU time: %.2f ms", timing.diff_time(gpu_time_idx), t.elapsed() *1000);

            for(auto i = 0; i < cnt; i++) { //check the results
                assert(probe_values[i]  == hash_values_input[i]);
            }
            CUDA_FREE(d_temp_storage, nullptr);
            CUDA_FREE(hash_keys_output, nullptr);
            CUDA_FREE(hash_values_output, nullptr);
            break;
        }
        case RADIX_SORT_KVS: {
            log_info("Case: RADIX_SORT_KVS");
            void *d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;

            uint32_t *keys_temp = nullptr, *values_temp = nullptr;
            CUDA_MALLOC(&keys_temp, sizeof(uint32_t)*cnt, nullptr);
            CUDA_MALLOC(&values_temp, sizeof(uint32_t)*cnt, nullptr);
            auto *timing_ptr = &timing;
            cudaDeviceSynchronize();

            /*Sort according to values*/
            auto gpu_time_idx = timing.get_idx();
            t.reset();
            timingKernel(
                    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, hash_values_input, values_temp, hash_keys_input, keys_temp, cnt), timing_ptr);
            CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, nullptr);
            timingKernel(
                    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, hash_values_input, values_temp, hash_keys_input, keys_temp, cnt), timing_ptr);
            CUDA_FREE(d_temp_storage, nullptr);
            d_temp_storage = nullptr;

            /* Sort according to keys*/
            timingKernel(
                    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys_temp, hash_keys_input, values_temp, hash_values_input, cnt), timing_ptr);
            CUDA_MALLOC(&d_temp_storage, temp_storage_bytes, nullptr);
            timingKernel(
                    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys_temp, hash_keys_input, values_temp, hash_values_input, cnt), timing_ptr);
            cudaDeviceSynchronize();
            log_info("[Build] Sort-KVS: kernel time: %.2f ms, CPU time: %.2f ms", timing.diff_time(gpu_time_idx), t.elapsed() *1000);

            CUDA_FREE(d_temp_storage, nullptr);
            CUDA_FREE(keys_temp,nullptr);
            CUDA_FREE(values_temp,nullptr);

            break;
        }
    }
}

/* Usage:
 *     ./cuda-partition DATA_FILE_ADDR ALGO_TYPE
 * */
int main(int argc, char *argv[]) {
    srand(time(nullptr));
    cudaSetDevice(DEVICE_ID);
    FILE *fp;
    fp = fopen("log.txt", "a+");
    if (fp == NULL) {
        cout<<"wrong file fp"<<endl;
        exit(1);
    }
    log_set_fp(fp);

    test_build_hashtables(string(argv[1]), (BuildHashType)(stoi(argv[2])));

//    vector<uint32_t> bucket_setting;
//    for(auto i = 2; i < argc; i++) {
//        bucket_setting.emplace_back(stoul(argv[i]));
//    }
//    test_radix_partitioning(string(argv[1]), bucket_setting);

//    test_cuckoo_hashing();
    return 0;
}