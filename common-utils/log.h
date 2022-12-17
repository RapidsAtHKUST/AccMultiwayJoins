/**
 * Copyright (c) 2017 rxi
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the MIT license. See `log.c` for details.
 */

#ifndef LOG_H
#define LOG_H

#ifdef USE_LOG

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <ctime>
#include <cstring>
#include <mutex>
#include <cmath>

using namespace std;
using namespace std::chrono;

std::mutex global_log_mutex;
time_point<high_resolution_clock> clk_beg = high_resolution_clock::now();

#define LOG_VERSION "0.1.0"

typedef void (*log_LockFn)(void *udata, int lock);

enum {
    LOG_TRACE, LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_FATAL
};

#define log_trace(...) log_log(LOG_TRACE, __FILE__, __LINE__, __VA_ARGS__)
#define log_debug(...) log_log(LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define log_info(...)  log_log(LOG_INFO,  __FILE__, __LINE__, __VA_ARGS__)
#define log_warn(...)  log_log(LOG_WARN,  __FILE__, __LINE__, __VA_ARGS__)
#define log_error(...) log_log(LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define log_fatal(...) log_log(LOG_FATAL, __FILE__, __LINE__, __VA_ARGS__)
#define log_no_newline(...) log_log_no_newline(LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)

static struct {
    void *udata;
    log_LockFn lock;
    FILE *fp;
    int level;
    int quiet;
} L;

static const char *level_names[] = {
        "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"
};
#ifndef LOG_USE_COLOR
#define LOG_USE_COLOR
#endif
#ifdef LOG_USE_COLOR
static const char *level_colors[] = {
        "\x1b[94m", "\x1b[36m", "\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[35m"
};
#endif


static void lock(void) {
    if (L.lock) {
        L.lock(L.udata, 1);
    }
}


static void unlock(void) {
    if (L.lock) {
        L.lock(L.udata, 0);
    }
}


inline void log_set_udata(void *udata) {
    L.udata = udata;
}


inline void log_set_lock(log_LockFn fn) {
    L.lock = fn;
}


inline void log_set_fp(FILE *fp) {
    L.fp = fp;
}


inline void log_set_level(int level) {
    L.level = level;
}


inline void log_set_quiet(int enable) {
    L.quiet = enable ? 1 : 0;
}


inline void log_log(int level, const char *fileAbs, int line, const char *fmt, ...) {
    if (level < L.level) {
        return;
    }

    using namespace std::chrono;
    time_point<high_resolution_clock> clock_now = high_resolution_clock::now();
    auto elapsed_time = duration_cast<nanoseconds>(clock_now - clk_beg).count();
    {
        unique_lock<mutex> lock_global(global_log_mutex);
        /* Acquire lock */
        lock();

        /* Get current time */
        time_t t = time(nullptr);
        struct tm *lt = localtime(&t);

        /*convert the file path into short term*/
        char file[256] = {'\0'};
        int finalSlashIdx = 0;
        for(int i = strlen(fileAbs)-1; i >= 0; i--)
        if (fileAbs[i] == '/')
        {
            finalSlashIdx = i;
            break;
        }

        int fileIdx = 0;
        for(int i = finalSlashIdx + 1; i < strlen(fileAbs); i++)
        file[fileIdx++] = fileAbs[i];

        /* Log to stderr */
        if (!L.quiet) {
            va_list args;
            char buf[64];
            buf[strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", lt)] = '\0';
#ifdef LOG_USE_COLOR
            fprintf(
                    stderr, "%s %s%-5s (ts: %.6lf s, et: %.6lf s) \x1b[0m \x1b[90m%s:%d:\x1b[0m ",
                    buf, level_colors[level], level_names[level],
                    duration_cast<nanoseconds>(clock_now.time_since_epoch()).count() / pow(10, 9),
                    elapsed_time / pow(10, 9),
                    file, line);
#else
            fprintf(stderr, "%s %-5s (ts: %.6lf s, et: %.6lf s) %s:%d: ", buf, level_names[level],
                    duration_cast<nanoseconds>(clock_now.time_since_epoch()).count() / pow(10, 9),
                    elapsed_time / pow(10, 9), file, line);
#endif
            va_start(args, fmt);
            vfprintf(stderr, fmt, args);
            va_end(args);
            fprintf(stderr, "\n");
        }

        /* Log to file */
        if (L.fp) {
            va_list args;
            char buf[32];
            buf[strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", lt)] = '\0';
            fprintf(L.fp, "%s %-5s (ts: %.6lf s,  et: %.6lf s) %s:%d: ", buf, level_names[level],
                    duration_cast<nanoseconds>(clock_now.time_since_epoch()).count() / pow(10, 9),
                    elapsed_time / pow(10, 9), file, line);
            va_start(args, fmt);
            vfprintf(L.fp, fmt, args);
            va_end(args);
            fprintf(L.fp, "\n");
        }

        /* Release lock */
        unlock();
    }
}

inline void log_log_no_newline(int level, const char *fileAbs, int line, const char *fmt, ...) {
    if (level < L.level) {
        return;
    }

    using namespace std::chrono;
    time_point<high_resolution_clock> clock_now = high_resolution_clock::now();
    auto elapsed_time = duration_cast<nanoseconds>(clock_now - clk_beg).count();
    {
        unique_lock<mutex> lock_global(global_log_mutex);
        /* Acquire lock */
        lock();

        /* Get current time */
        time_t t = time(nullptr);
        struct tm *lt = localtime(&t);

        /*convert the file path into short term*/
        char file[256] = {'\0'};
        int finalSlashIdx = 0;
        for(int i = strlen(fileAbs)-1; i >= 0; i--)
        if (fileAbs[i] == '/')
        {
            finalSlashIdx = i;
            break;
        }

        int fileIdx = 0;
        for(int i = finalSlashIdx + 1; i < strlen(fileAbs); i++)
        file[fileIdx++] = fileAbs[i];

        /* Log to stderr */
        if (!L.quiet) {
            va_list args;
            char buf[64];
            buf[strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", lt)] = '\0';
#ifdef LOG_USE_COLOR
            fprintf(
                    stderr, "%s %s%-5s (ts: %.6lf s, et: %.6lf s) \x1b[0m \x1b[90m%s:%d:\x1b[0m ",
                    buf, level_colors[level], level_names[level],
                    duration_cast<nanoseconds>(clock_now.time_since_epoch()).count() / pow(10, 9),
                    elapsed_time / pow(10, 9),
                    file, line);
#else
            fprintf(stderr, "%s %-5s (ts: %.6lf s, et: %.6lf s) %s:%d: ", buf, level_names[level],
                    duration_cast<nanoseconds>(clock_now.time_since_epoch()).count() / pow(10, 9),
                    elapsed_time / pow(10, 9), file, line);
#endif
            va_start(args, fmt);
            vfprintf(stderr, fmt, args);
            va_end(args);
            //fprintf(stderr, "\n"); //no new line
        }

        /* Log to file */
        if (L.fp) {
            va_list args;
            char buf[32];
            buf[strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", lt)] = '\0';
            fprintf(L.fp, "%s %-5s (ts: %.6lf s,  et: %.6lf s) %s:%d: ", buf, level_names[level],
                    duration_cast<nanoseconds>(clock_now.time_since_epoch()).count() / pow(10, 9),
                    elapsed_time / pow(10, 9), file, line);
            va_start(args, fmt);
            vfprintf(L.fp, fmt, args);
            va_end(args);
            //fprintf(L.fp, "\n"); //no new line
        }

        /* Release lock */
        unlock();
    }
}

#else //use log
#define log_trace(...)
#define log_debug(...)
#define log_info(...)
#define log_warn(...)
#define log_error(...)
#define log_fatal(...)
#define log_no_newline(...)
#endif //use log
#endif
