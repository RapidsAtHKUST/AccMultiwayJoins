#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <map>
#include <omp.h>
#include <iostream>

#include "topkapi.h"
#include "LossyCountMinSketch.h"
#include "HashFunction.h"

using namespace std;

void topkapi(int *input, int len,
             int *outputIdentity, uint32_t *outputFreq,
             int &outputLen, uint32_t K)
{
    int i, j;
    /* Default Parameter values for Sketch */
    unsigned range = 8192; /* range of buckets for the sketch */
    unsigned log2range = (unsigned) log2(range); /* log2 value of range */
    unsigned num_hash_func = 4; /* number of hash functions */
    unsigned frac_epsilon = 10*K; /* error margin */
    outputLen = 0;

    /* Data for Sketch */
    LossySketch* th_local_sketch; /* array of thread local sketches */
    LossySketch* node_final_sketch; /* final sketch in a node after merging thread local sketches */

    /* Random numbers for hash functions */
    bool rand_differ;

    /* Data structures for sorting the heavy hitters to find topK */
    std::map<int,int> topk_words;

    /* variables for OpenMP */
    int num_threads = omp_get_num_procs(); /* default to number of procs available */

    /* hash function specific different random number generartion */
    auto hash_func_rands = new uint32_t[num_hash_func];
    for (i = 0; i < num_hash_func; ++i)
    {
        rand_differ = false;
        hash_func_rands[i] = (uint32_t) (rand() % 47);
        while (!rand_differ)
        {
            rand_differ = true;
            for (j = 0; j < i; ++j)
            {
                if (hash_func_rands[i] == hash_func_rands[j])
                {
                    rand_differ = false;
                    break;
                }
            }
            if (!rand_differ)
            {
                hash_func_rands[i] = (uint32_t) (rand() % 47);
            }
        }
    }

    th_local_sketch = new LossySketch[num_hash_func*num_threads];
    /* represents the first num_hash_func number of sketches from th_local_sketch array */
    node_final_sketch = th_local_sketch;

    omp_set_num_threads( num_threads );
    omp_set_dynamic( 0 );
#pragma omp parallel firstprivate(th_local_sketch, range, num_hash_func)
    {
        int tid = omp_get_thread_num();
        int th_i;

        /* Allocate and initialize sketch variables */
        for (th_i = 0; th_i < num_hash_func; ++th_i)
        {
            allocate_sketch( &th_local_sketch[tid*num_hash_func+th_i], range);
        }
    }

    omp_set_num_threads(num_threads);
    omp_set_dynamic(0);

#pragma omp parallel for firstprivate(range, log2range,num_hash_func, th_local_sketch, hash_func_rands) private (i)
    for(int k = 0; k < len; k++)
    {
        int tid = omp_get_thread_num();
        uint32_t val = input[k];

        /* read words from the buffer one by one */
        for (int l = 0; l < num_hash_func; ++l)
            update_sketch( &th_local_sketch[tid * num_hash_func + l],
                           val, hash_func_rands[l], log2range);
    }

    /* Now merge thread local sketches to produce final sketch for a node */
    for (i = 0; i < num_hash_func; ++i)
        local_merge_sketch(th_local_sketch, num_threads, num_hash_func, i);

    /* Print TopK words */
    int count;
    uint32_t val;
    int id;
    auto is_heavy_hitter = new bool[range];
    auto threshold = (int) ((range/K)-(range/frac_epsilon));

#pragma omp parallel for schedule(static) firstprivate(threshold,\
        num_hash_func, log2range, is_heavy_hitter, hash_func_rands) private(j, count, val, id)
    for (i = 0; i < range; ++i)
    {
        is_heavy_hitter[i] = false;
        for (j = 0; j < num_hash_func; ++j)
        {
            if (j == 0)
            {
                val = node_final_sketch[0].identity[i];
                count = node_final_sketch[0].lossyCount[i];

                if (count >= threshold)
                {
                    is_heavy_hitter[i] = true;
                }
            }
            else
            {
                id = gethash( &val, hash_func_rands[j], log2range );

                if (node_final_sketch[j].identity[id] != val)
                {
                    continue;
                }
                else if (node_final_sketch[j].lossyCount[id] >= threshold)
                {
                    is_heavy_hitter[i] = true;
                }
                if (node_final_sketch[j].lossyCount[id] > count)
                    count = node_final_sketch[j].lossyCount[id];
            }
        }
        node_final_sketch[0].lossyCount[i] = count;
    }

    for (i = 0; i < range; ++i)
    {
        if (is_heavy_hitter[i])
        {
            topk_words.insert( std::pair<int,int>(node_final_sketch[0].lossyCount[i], i) );
        }
    }

    auto rit = topk_words.rbegin();
    for (i = 0; (i < K) && (rit != topk_words.rend());
         ++i, ++rit)
    {
        j = rit->second;
        outputIdentity[outputLen] = node_final_sketch[0].identity[j];
        if (outputFreq) outputFreq[outputLen] = rit->first; /*freq is no need sometimes*/
        outputLen++;
    }

    /* free memories */
    for (i = 0; i < num_threads; ++i) {
        for (j = 0; j < num_hash_func; ++j) {
            deallocate_sketch( &th_local_sketch[i*num_hash_func + j]);
        }
    }
    delete[] th_local_sketch;
    delete[] is_heavy_hitter;
    delete[] hash_func_rands;
}
