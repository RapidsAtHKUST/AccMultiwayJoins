//
// Created by Bryan on 31/12/2018.
//

#ifndef GPU_OPERATORS_TOPKAPI_H
#define GPU_OPERATORS_TOPKAPI_H

#include <iostream>

void topkapi(int *input, int len,
             int *outputIdentity, uint32_t *outputFreq,
             int &outputLen, uint32_t K);

#endif //GPU_OPERATORS_TOPKAPI_H
