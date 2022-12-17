//
// Created by Bryan on 8/3/2020.
//
#pragma once

#include <iostream>
#include "cuda/CUDAStat.cuh"
#include "IndexedTrie.cuh"

/*base class for LFTJ-DFS and LFTJ-BFS*/
template<typename DataType, typename CntType>
class LFTJ_Base {
protected:
    uint32_t num_tables; //number of input tables in this LFTJ
    uint32_t num_attrs;  //number of attributes in this LFTJ
    uint32_t *attr_order; //attr order
    CUDAMemStat *memstat; //memstat associlated
    CUDATimeStat *timing; //timing associlated
public:
    virtual CntType evaluate(IndexedTrie<DataType,CntType> *Tries,
                             DataType **res_tuples, CntType *num_res,
                             bool ooc, bool work_sharing, cudaStream_t stream) = 0;
};