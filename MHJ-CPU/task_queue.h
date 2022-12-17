#pragma once

#include <pthread.h>
#include <cstdlib>

#include "types.h"

/* record the input, output, length of the partitioning task */
template<typename DataType, typename CntType>
struct task_t {
    hash_tuple_t<DataType,CntType> *input;
    hash_tuple_t<DataType,CntType> *output;
    CntType length;

    CntType task_start_pos;
    CntType *task_his_start;
    int D;

    task_t<DataType,CntType> *next;
};

template<typename DataType, typename CntType>
struct task_list_t {
    task_t<DataType,CntType> *tasks;
    task_list_t<DataType,CntType> *next;
    int curr;
};

template<typename DataType, typename CntType>
struct task_queue_t {
    pthread_mutex_t lock;
    pthread_mutex_t alloc_lock;
    task_t<DataType,CntType> *head;
    task_list_t<DataType,CntType> *free_list;
    int32_t         count;
    int32_t         alloc_size;
};

template<typename DataType, typename CntType>
struct TaskQueue {
    task_queue_t<DataType,CntType> *queue_list;
    TaskQueue(int alloc_size) {
        queue_list = (task_queue_t<DataType,CntType>*) malloc(sizeof(task_queue_t<DataType,CntType>));
        queue_list->free_list = (task_list_t<DataType,CntType>*) malloc(sizeof(task_list_t<DataType,CntType>));
        queue_list->free_list->tasks = (task_t<DataType,CntType>*) malloc(alloc_size * sizeof(task_t<DataType,CntType>));
        queue_list->free_list->curr = 0;
        queue_list->free_list->next = nullptr;
        queue_list->count      = 0;
        queue_list->alloc_size = alloc_size;
        queue_list->head       = nullptr;
        pthread_mutex_init(&queue_list->lock, nullptr);
        pthread_mutex_init(&queue_list->alloc_lock, nullptr);
    }
    ~TaskQueue() {
        auto *tmp = queue_list->free_list;
        while(tmp) {
            free(tmp->tasks);
            auto *tmp2 = tmp->next;
            free(tmp);
            tmp = tmp2;
        }
        free(queue_list);
    }
    task_t<DataType,CntType> *task_queue_get_atomic() {
        pthread_mutex_lock(&queue_list->lock);
        task_t<DataType,CntType> *ret = nullptr;
        if(queue_list->count > 0){
            ret      = queue_list->head;
            queue_list->head = ret->next;
            queue_list->count --;
        }
        pthread_mutex_unlock(&queue_list->lock);
        return ret;
    }
    void task_queue_add(task_t<DataType,CntType> *t) {
        t->next  = queue_list->head;
        queue_list->head = t;
        queue_list->count ++;
    }
    task_t<DataType,CntType> *task_queue_get_slot() {
        auto *l = queue_list->free_list;
        task_t<DataType,CntType> * ret;
        if(l->curr < queue_list->alloc_size) {
            ret = &(l->tasks[l->curr]);
            l->curr++;
        }
        else {
            auto *nl = (task_list_t<DataType,CntType>*) malloc(sizeof(task_list_t<DataType,CntType>));
            nl->tasks = (task_t<DataType,CntType>*) malloc(queue_list->alloc_size * sizeof(task_t<DataType,CntType>));
            nl->curr = 1;
            nl->next = queue_list->free_list;
            queue_list->free_list = nl;
            ret = &(nl->tasks[0]);
        }
        return ret;
    }
};
