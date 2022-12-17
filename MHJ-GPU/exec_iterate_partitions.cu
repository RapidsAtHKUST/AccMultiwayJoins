//
// Created by Bryan on 29/5/2020.
//

#include "Indexing/radix_partitioning.cuh"
#include "log.h"
#include "file_io.h"
#include "Relation.cuh"

using namespace std;

template<typename KType, typename VType, typename CntType>
void test_partition(string file_addr, uint32_t num_buckets) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, DEVICE_ID);
    uint32_t max_buckets = (uint32_t)(prop.sharedMemPerBlock / sizeof(CntType));

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

    uint32_t *output_hash_keys = nullptr;
    uint32_t *buc_ptrs = nullptr;
    uint32_t *hash_values_input = nullptr;
    uint32_t *hash_values_output = nullptr;
    CUDA_MALLOC(&output_hash_keys, sizeof(uint32_t)*cnt, nullptr);
    CUDA_MALLOC(&buc_ptrs, sizeof(uint32_t)*(num_buckets+1), nullptr);
    CUDA_MALLOC(&hash_values_input, sizeof(uint32_t)*cnt, nullptr);
    CUDA_MALLOC(&hash_values_output, sizeof(uint32_t)*cnt, nullptr);
    thrust::counting_iterator<uint32_t> iter(0);
    thrust::copy(iter, iter + cnt, hash_values_input);
    cudaDeviceSynchronize();

    Timer t;
    CUDATimeStat timing;

    int buc_bits = logFunc(num_buckets);
    vector<pair<string, pair<float,float>>> results;
    string fastest_setting;
    pair<double,double> fastest_time;
    double fastest_cpu_time = 999999999;

    /*cold run*/
    cout<<"cold run"<<endl; //cold run, 1-pass, make sure no memory swap here
    RadixPartitioner<uint32_t, uint32_t, uint32_t> rp_cold(cnt, 1, 256, nullptr, nullptr);
    rp_cold.splitKV(data_out[0], output_hash_keys,
                    hash_values_input, hash_values_output, buc_ptrs);

    /*pass = 1*/
    if (num_buckets <= max_buckets) {
        vector<uint32_t> bucket_setting;
        bucket_setting.emplace_back(num_buckets);
        string setting = to_string(num_buckets);
        cout<<"setting: "<<setting<<endl;

        auto gpu_time_idx = timing.get_idx();
        t.reset();
        RadixPartitioner<uint32_t, uint32_t, uint32_t> rp(cnt, bucket_setting, nullptr, &timing);
        rp.splitKV(data_out[0], output_hash_keys,
                   hash_values_input, hash_values_output, buc_ptrs);
        auto gpu_time = timing.diff_time(gpu_time_idx);
        auto cpu_time = t.elapsed() *1000;
        log_info("Hash with partitioning: kernel time: %.2f ms, CPU time: %.2f ms", gpu_time, cpu_time);
        results.emplace_back(make_pair(setting, make_pair(gpu_time, cpu_time)));

        if (cpu_time < fastest_cpu_time) {
            fastest_setting = setting;
            fastest_cpu_time = cpu_time;
            fastest_time = make_pair(gpu_time, cpu_time);
        }
    }
    /*pass = 2*/
    if (num_buckets >= 4 * 4) {
        for(auto p1_bits = 4; p1_bits < buc_bits; p1_bits++) {
            auto p2_bits = buc_bits - p1_bits;
            uint32_t buc_p1 = (uint32_t)(pow(2, p1_bits));
            uint32_t buc_p2 = (uint32_t)(pow(2, p2_bits));
            if ((buc_p1 > max_buckets) || (buc_p1 < 2)) continue;
            if ((buc_p2 > max_buckets) || (buc_p2 < 2)) continue;

            vector<uint32_t> bucket_setting;
            bucket_setting.emplace_back(buc_p1);
            bucket_setting.emplace_back(buc_p2);

            string setting = to_string(buc_p1)+"*"+to_string(buc_p2);
            cout<<"setting: "<<setting<<endl;

            RadixPartitioner<uint32_t, uint32_t, uint32_t> rp(cnt, bucket_setting, nullptr, &timing);
            cudaDeviceSynchronize();

            auto gpu_time_idx = timing.get_idx();
            t.reset();
            rp.splitKV(data_out[0], output_hash_keys,
                       hash_values_input, hash_values_output, buc_ptrs);
            auto gpu_time = timing.diff_time(gpu_time_idx);
            auto cpu_time = t.elapsed() *1000;
            log_info("Hash with partitioning: kernel time: %.2f ms, CPU time: %.2f ms", gpu_time, cpu_time);

            results.emplace_back(make_pair(setting, make_pair(gpu_time, cpu_time)));

            if (cpu_time < fastest_cpu_time) {
                fastest_setting = setting;
                fastest_cpu_time = cpu_time;
                fastest_time = make_pair(gpu_time, cpu_time);
            }
        }
    }
    /*pass = 3*/
    if (num_buckets >= 16 * 16 * 16) {
        for(auto p1_bits = 4; p1_bits < buc_bits; p1_bits++) {
            for(auto p2_bits = 4; p2_bits < buc_bits; p2_bits++) {
                auto p3_bits = buc_bits - p1_bits - p2_bits;
                uint32_t buc_p1 = (uint32_t)(pow(2, p1_bits));
                uint32_t buc_p2 = (uint32_t)(pow(2, p2_bits));
                uint32_t buc_p3 = (uint32_t)(pow(2, p3_bits));
                if ((buc_p1 > max_buckets) || (buc_p1 < 2)) continue;
                if ((buc_p2 > max_buckets) || (buc_p2 < 2)) continue;
                if ((buc_p3 > max_buckets) || (buc_p3 < 2)) continue;

                vector<uint32_t> bucket_setting;
                bucket_setting.emplace_back(buc_p1);
                bucket_setting.emplace_back(buc_p2);
                bucket_setting.emplace_back(buc_p3);

                string setting = to_string(buc_p1)+"*"+to_string(buc_p2)+"*"+to_string(buc_p3);
                cout<<"setting: "<<setting<<endl;

                RadixPartitioner<uint32_t, uint32_t, uint32_t> rp(cnt, bucket_setting, nullptr, &timing);
                cudaDeviceSynchronize();

                auto gpu_time_idx = timing.get_idx();
                t.reset();
                rp.splitKV(data_out[0], output_hash_keys,
                           hash_values_input, hash_values_output, buc_ptrs);
                auto gpu_time = timing.diff_time(gpu_time_idx);
                auto cpu_time = t.elapsed() *1000;
                log_info("Hash with partitioning: kernel time: %.2f ms, CPU time: %.2f ms", gpu_time, cpu_time);

                results.emplace_back(make_pair(setting, make_pair(gpu_time, cpu_time)));

                if (cpu_time < fastest_cpu_time) {
                    fastest_setting = setting;
                    fastest_cpu_time = cpu_time;
                    fastest_time = make_pair(gpu_time, cpu_time);
                }
            }
        }
    }
    /*pass = 4*/
    if (num_buckets >= 16 * 16 * 16 * 16) {
        for(auto p1_bits = 4; p1_bits < buc_bits; p1_bits++) {
            for(auto p2_bits = 4; p2_bits < buc_bits; p2_bits++) {
                for(auto p3_bits = 4; p3_bits < buc_bits; p3_bits++) {
                    auto p4_bits = buc_bits - p1_bits - p2_bits - p3_bits;
                    uint32_t buc_p1 = (uint32_t)(pow(2, p1_bits));
                    uint32_t buc_p2 = (uint32_t)(pow(2, p2_bits));
                    uint32_t buc_p3 = (uint32_t)(pow(2, p3_bits));
                    uint32_t buc_p4 = (uint32_t)(pow(2, p4_bits));
                    if ((buc_p1 > max_buckets) || (buc_p1 < 2)) continue;
                    if ((buc_p2 > max_buckets) || (buc_p2 < 2)) continue;
                    if ((buc_p3 > max_buckets) || (buc_p3 < 2)) continue;
                    if ((buc_p4 > max_buckets) || (buc_p4 < 2)) continue;

                    vector<uint32_t> bucket_setting;
                    bucket_setting.emplace_back(buc_p1);
                    bucket_setting.emplace_back(buc_p2);
                    bucket_setting.emplace_back(buc_p3);
                    bucket_setting.emplace_back(buc_p4);

                    string setting = to_string(buc_p1)+"*"+to_string(buc_p2)+"*"+to_string(buc_p3)+"*"+to_string(buc_p4);
                    cout<<"setting: "<<setting<<endl;

                    RadixPartitioner<uint32_t, uint32_t, uint32_t> rp(cnt, bucket_setting, nullptr, &timing);
                    cudaDeviceSynchronize();

                    auto gpu_time_idx = timing.get_idx();
                    t.reset();
                    rp.splitKV(data_out[0], output_hash_keys,
                               hash_values_input, hash_values_output, buc_ptrs);
                    auto gpu_time = timing.diff_time(gpu_time_idx);
                    auto cpu_time = t.elapsed() *1000;
                    log_info("Hash with partitioning: kernel time: %.2f ms, CPU time: %.2f ms", gpu_time, cpu_time);

                    results.emplace_back(make_pair(setting, make_pair(gpu_time, cpu_time)));

                    if (cpu_time < fastest_cpu_time) {
                        fastest_setting = setting;
                        fastest_cpu_time = cpu_time;
                        fastest_time = make_pair(gpu_time, cpu_time);
                    }
                }
            }
        }
    }
    /*pass = 5*/
    if (num_buckets >= 16 * 16 * 16 * 16 * 16) {
        for(auto p1_bits = 4; p1_bits < buc_bits; p1_bits++) {
            for(auto p2_bits = 4; p2_bits < buc_bits; p2_bits++) {
                for(auto p3_bits = 4; p3_bits < buc_bits; p3_bits++) {
                    for(auto p4_bits = 4; p4_bits < buc_bits; p4_bits++) {
                        auto p5_bits = buc_bits - p1_bits - p2_bits - p3_bits - p4_bits;
                        uint32_t buc_p1 = (uint32_t)(pow(2, p1_bits));
                        uint32_t buc_p2 = (uint32_t)(pow(2, p2_bits));
                        uint32_t buc_p3 = (uint32_t)(pow(2, p3_bits));
                        uint32_t buc_p4 = (uint32_t)(pow(2, p4_bits));
                        uint32_t buc_p5 = (uint32_t)(pow(2, p5_bits));
                        if ((buc_p1 > max_buckets) || (buc_p1 < 2)) continue;
                        if ((buc_p2 > max_buckets) || (buc_p2 < 2)) continue;
                        if ((buc_p3 > max_buckets) || (buc_p3 < 2)) continue;
                        if ((buc_p4 > max_buckets) || (buc_p4 < 2)) continue;
                        if ((buc_p5 > max_buckets) || (buc_p5 < 2)) continue;

                        vector<uint32_t> bucket_setting;
                        bucket_setting.emplace_back(buc_p1);
                        bucket_setting.emplace_back(buc_p2);
                        bucket_setting.emplace_back(buc_p3);
                        bucket_setting.emplace_back(buc_p4);
                        bucket_setting.emplace_back(buc_p5);

                        string setting = to_string(buc_p1)+"*"+to_string(buc_p2)+"*"+to_string(buc_p3)
                                         +"*"+to_string(buc_p4)+"*"+to_string(buc_p5);
                        cout<<"setting: "<<setting<<endl;

                        RadixPartitioner<uint32_t, uint32_t, uint32_t> rp(cnt, bucket_setting, nullptr, &timing);
                        cudaDeviceSynchronize();

                        auto gpu_time_idx = timing.get_idx();
                        t.reset();
                        rp.splitKV(data_out[0], output_hash_keys,
                                   hash_values_input, hash_values_output, buc_ptrs);
                        auto gpu_time = timing.diff_time(gpu_time_idx);
                        auto cpu_time = t.elapsed() *1000;
                        log_info("Hash with partitioning: kernel time: %.2f ms, CPU time: %.2f ms", gpu_time, cpu_time);

                        results.emplace_back(make_pair(setting, make_pair(gpu_time, cpu_time)));

                        if (cpu_time < fastest_cpu_time) {
                            fastest_setting = setting;
                            fastest_cpu_time = cpu_time;
                            fastest_time = make_pair(gpu_time, cpu_time);
                        }
                    }
                }
            }
        }
    }
    /*pass = 6*/
    if (num_buckets >= 16 * 16 * 16 * 16 * 16 * 16) {
        for(auto p1_bits = 4; p1_bits < buc_bits; p1_bits++) {
            for(auto p2_bits = 4; p2_bits < buc_bits; p2_bits++) {
                for(auto p3_bits = 4; p3_bits < buc_bits; p3_bits++) {
                    for(auto p4_bits = 4; p4_bits < buc_bits; p4_bits++) {
                        for(auto p5_bits = 4; p5_bits < buc_bits; p5_bits++) {
                            auto p6_bits = buc_bits - p1_bits - p2_bits - p3_bits - p4_bits - p5_bits;
                            uint32_t buc_p1 = (uint32_t)(pow(2, p1_bits));
                            uint32_t buc_p2 = (uint32_t)(pow(2, p2_bits));
                            uint32_t buc_p3 = (uint32_t)(pow(2, p3_bits));
                            uint32_t buc_p4 = (uint32_t)(pow(2, p4_bits));
                            uint32_t buc_p5 = (uint32_t)(pow(2, p5_bits));
                            uint32_t buc_p6 = (uint32_t)(pow(2, p6_bits));
                            if ((buc_p1 > max_buckets) || (buc_p1 < 2)) continue;
                            if ((buc_p2 > max_buckets) || (buc_p2 < 2)) continue;
                            if ((buc_p3 > max_buckets) || (buc_p3 < 2)) continue;
                            if ((buc_p4 > max_buckets) || (buc_p4 < 2)) continue;
                            if ((buc_p6 > max_buckets) || (buc_p6 < 2)) continue;

                            vector<uint32_t> bucket_setting;
                            bucket_setting.emplace_back(buc_p1);
                            bucket_setting.emplace_back(buc_p2);
                            bucket_setting.emplace_back(buc_p3);
                            bucket_setting.emplace_back(buc_p4);
                            bucket_setting.emplace_back(buc_p5);
                            bucket_setting.emplace_back(buc_p6);

                            string setting = to_string(buc_p1)+"*"+to_string(buc_p2)+"*"+to_string(buc_p3)
                                             +"*"+to_string(buc_p4)+"*"+to_string(buc_p5)+"*"+to_string(buc_p6);
                            cout<<"setting: "<<setting<<endl;

                            RadixPartitioner<uint32_t, uint32_t, uint32_t> rp(cnt, bucket_setting, nullptr, &timing);
                            cudaDeviceSynchronize();

                            auto gpu_time_idx = timing.get_idx();
                            t.reset();
                            rp.splitKV(data_out[0], output_hash_keys,
                                       hash_values_input, hash_values_output, buc_ptrs);
                            auto gpu_time = timing.diff_time(gpu_time_idx);
                            auto cpu_time = t.elapsed() *1000;
                            log_info("Hash with partitioning: kernel time: %.2f ms, CPU time: %.2f ms", gpu_time, cpu_time);

                            results.emplace_back(make_pair(setting, make_pair(gpu_time, cpu_time)));

                            if (cpu_time < fastest_cpu_time) {
                                fastest_setting = setting;
                                fastest_cpu_time = cpu_time;
                                fastest_time = make_pair(gpu_time, cpu_time);
                            }
                        }
                    }
                }
            }
        }
    }

    cout<<"Buckets: "<<num_buckets<<", results: "<<endl;
    for(auto const& x : results) {
        cout<<x.first<<"\t\t"<<x.second.first<<"\t\t"<<x.second.second<<endl;
    }
    cout<<"fastest setting: "<<fastest_setting<<endl;
    cout<<"fastest time: "<<fastest_time.first<<"\t"<<fastest_time.second<<endl;
}

int main(int argc, char *argv[]) {
    cudaSetDevice(DEVICE_ID);
    test_partition<uint32_t,uint32_t,uint32_t>(string(argv[1]), stoul(argv[2]));
    return 0;
}