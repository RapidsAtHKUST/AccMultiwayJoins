//
// Created by Bryan on 27/6/2020.
//
#include "../../dependencies/cudpp/include/cudpp.h"
#include "../../dependencies/cudpp/include/cudpp_hash.h"
#include "../types.h"
#include "../Relation.cuh"
#include "helper.h"
#include "../common_kernels.cuh"
#include "timer.h"

template<typename KType, typename VType, typename CntType>
class CudppHashTable {
private:
    const CntType _table_card_ori;  //cardinality of the original input table
    CUDPPHashTableType _type;
    float _space_usage;
    CUDPPHandle _cudpp;
    CUDPPHandle _hash_table_handle;

public:
    CudppHashTable(CUDPPHashTableType type, CntType table_card, float space_usage):
            _type(type), _table_card_ori(table_card), _space_usage(space_usage) {
        cudppCreate(&_cudpp);

        CUDPPHashTableConfig config;
        config.type = _type;
        config.kInputSize = _table_card_ori;
        config.space_usage = _space_usage;

        cudppHashTable(_cudpp, &_hash_table_handle, &config);
    }
    ~CudppHashTable() {
        cudppDestroyHashTable(_cudpp, _hash_table_handle);
        cudppDestroy(_cudpp);
    }
    void insert_vals(KType *keys, VType *values, CntType n);
    void lookup_vals(KType *keys, VType *values, CntType n);
};

template<typename KType, typename VType, typename CntType>
void
CudppHashTable<KType,VType,CntType>::insert_vals(KType *keys, VType *values, CntType n) {
    assert(n == _table_card_ori);
    cudppHashInsert(_hash_table_handle, keys, values, n);
};

template<typename KType, typename VType, typename CntType>
void
CudppHashTable<KType,VType,CntType>::lookup_vals(KType *keys, VType *values, CntType n) {
    assert(n == _table_card_ori);
    cudppHashRetrieve(_hash_table_handle, keys, values, n);
};

