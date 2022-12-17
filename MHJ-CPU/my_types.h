//
// Created by Bryan on 16/6/2019.
//

#ifndef GPU_OPERATORS_MY_TYPES_H
#define GPU_OPERATORS_MY_TYPES_H

#include <iostream>
#include <vector>

#define OUTPUT_PAGE_SIZE        (1024)      //items per column per page

typedef std::vector<uint32_t> Vector;
typedef std::pair<Vector*,Vector*> Match;

/*CSR format of a relation R(x,y)*/
struct CSR
{
    Vector keys;
    Vector offsets;
    Vector vals;

    CSR(Vector &keys, Vector &offsets, Vector &vals)
    {
        this->keys = keys;
        this->offsets = offsets;
        this->vals = vals;
    }
};

struct OutputPage
{
    int num_col;                //number of columns
    uint32_t num_data;          //number of data items in the page
    std::vector<std::vector<uint32_t>> data;

    OutputPage() {}
    OutputPage(int num_col)
    {
        this->num_col = num_col;
        num_data = 0;

        data.resize(num_col);
        for(int i = 0; i < num_col; ++i)
        {
            data[i].reserve(OUTPUT_PAGE_SIZE);
        }
    }
};

/*arguments for each thread*/
/*should pass the pointers of the vectors instead of the vector, to avoid vector copying*/
struct LFTJ_arg_t
{
    Vector *shared_mat_set_0;    //first set of the shared matching set pair
    Vector *shared_mat_set_1;    //second set of the shared matching set pair
    uint32_t num_rels;          //number of relations
    std::vector<CSR> *my_csrs;        //vectors of CSRs

    uint32_t start_pos;         //start position in the shared matching set processed
    uint32_t end_pos;           //end position in the shared matching set processed, exclusive

    uint32_t *num_res;          //number of results returned
    std::vector<OutputPage> *res;
};

#endif //GPU_OPERATORS_MY_TYPES_H
