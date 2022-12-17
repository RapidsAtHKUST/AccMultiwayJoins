//
// Created by Bryan on 24/4/2020.
//

#pragma once

#define PTHREAD_BARRIER_SERIAL_THREAD (0)

struct pthread_barrier_t {};
struct pthread_barrierattr_t {};

int pthread_barrier_wait(pthread_barrier_t *barrier);
int pthread_barrier_destroy(pthread_barrier_t *barrier);
int pthread_barrier_init(pthread_barrier_t *barrier, const pthread_barrierattr_t *attr, unsigned count);