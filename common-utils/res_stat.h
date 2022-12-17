//
// Created by Bryan on 22/11/2020.
//

#pragma once

#include <iostream>
#include <vector>
#include <sstream>

#include "timer.h"
#include "log.h"
#include "pretty_print.h"
#include "md5.h"

#define MAX_VAL_BITS    (20)
#define MAX_VAL         (1 << MAX_VAL_BITS)
#define MAX_VAL_MASK    (MAX_VAL - 1)

using namespace std;

template<class T>
std::string FormatWithCommas(T value) {
    std::stringstream ss;
    ss.imbue(std::locale(""));
    ss << std::fixed << value;
    return ss.str();
}

/* compute the histogram values appreared in the output table */
template<typename DataType, typename CntType>
vector<CntType> res_stat(DataType **values, CntType n, int num_attrs, bool is_print = false) {
    Timer histogram_timer;
    stringstream ss;

    // res-value histogram
    DataType max_res_val = 0;
    vector<CntType> histogram;
#pragma omp parallel
    {
        for(auto c = 0; c < num_attrs; c++) {
#pragma omp for reduction(max:max_res_val)
            for (CntType u = 0; u < n; u++) {
                max_res_val = max(max_res_val, values[c][u] & MAX_VAL_MASK);
            }
        }

#pragma omp single
        {
            log_info("max value (mod %d): %d", MAX_VAL, max_res_val);
            histogram = vector<CntType>(max_res_val + 1, 0);
        }
        vector<CntType> local_histogram(histogram.size());

        for(auto c = 0; c < num_attrs; c++) {
#pragma omp for
            for (auto u = 0; u < n; u++) {
                auto val = values[c][u] & MAX_VAL_MASK;
                local_histogram[val]++;
            }
        }

        // local_histogram[i] is immutable here.
        for (auto i = 0u; i < local_histogram.size(); i++) {
#pragma omp atomic
            histogram[i] += local_histogram[i];
        }
    }
    if (is_print) {
        if (histogram.size() < 400) {
            stringstream ss;
            ss << pretty_print_array(&histogram.front(), histogram.size());
            log_info("values histogram: %s", ss.str().c_str());
        } else {
            {
                stringstream ss;
                ss << pretty_print_array(&histogram.front(), 100);
                log_info("first100 values histogram: %s", ss.str().c_str());
            }
            {

                stringstream ss;
                ss << pretty_print_array(&histogram.front() + histogram.size() - 100, 100);
                log_info("last100 values histogram: %s", ss.str().c_str());
            }
        }
    }
    log_info("Histogram Time: %.9lf s", histogram_timer.elapsed());

    auto &bins = histogram;
    auto bin_cnt = 0;
    int64_t acc = 0;
    auto thresh = n / 10;
    auto last = 0;

    for (auto i = 0u; i < histogram.size(); i++) {
        if (bins[i] > 0) {
            bin_cnt++;
            acc += bins[i];
            if (acc > thresh || i == histogram.size() - 1) {
                if (is_print) {
                    log_info("bin[%d - %d]: %s", last, i, to_string(acc).c_str());
                }
                last = i + 1;
                acc = 0;
            }
        }
    }
    if (is_print) {
        log_info("Reversed Bins...");
    }
    last = histogram.size() - 1;
    acc = 0;
    for (int32_t i = histogram.size() - 1; i > -1; i--) {
        if (bins[i] > 0) {
            bin_cnt++;
            acc += bins[i];
            if (acc > thresh || i == 0) {
                if (is_print) {
                    log_info("bin[%d - %d]: %s", i, last, to_string(acc).c_str());
                }
                last = i + 1;
                acc = 0;
            }
        }
    }
    log_info("total bin counts: %d", bin_cnt);

    ss << histogram << "\n";
    log_info("Md5sum of result histogram: %s", md5(ss.str()).c_str());

    return histogram;
}
