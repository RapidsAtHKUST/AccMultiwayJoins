//
// Created by Bryan on 31/7/2019.
//
#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <sys/stat.h>
#include <regex>
#include <dirent.h>
#include "log.h"
using namespace std;

vector<uint64_t> power2table = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,
131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432,67108864,134217728,268435456,
536870912,1073741824,2147483648};

template<typename DataType>
inline bool isPowerOf2(DataType input) {
    return input && (! (input & (input-1)));
}

template<typename DataType>
DataType floorPowerOf2(DataType val) {
    int i;
    for(i = (int)power2table.size() - 1; i >= 0; i--)
    {
        if (power2table[i] <= (uint64_t)val)
            return (DataType)power2table[i];
    }
    assert(i >= 0);
    return 0;
}

template<typename DataType>
DataType ceilPowerOf2(DataType val) {
    int i;
    for(i = 0; i < (int)power2table.size(); i++) {
        if (power2table[i] >= (uint64_t)val)
            return (DataType)power2table[i];
    }
    assert(i < power2table.size());
    return 0;
}

int logFunc(uint32_t num) {
    for(auto i = 0; i < power2table.size(); i++) {
        if (power2table[i] == (uint64_t)num) return i;
    }
    return -1;
}

/*check whether a precedes b in arr*/
template<typename DataType, typename CntType>
bool precede(DataType *arr, CntType num, const DataType a, const DataType b) {
    bool meet_a = false;
    for(auto i = 0; i < num; i++) {
        if (arr[i] == a)
            meet_a = true;
        else if (arr[i] == b) {
            if (meet_a) return true;    //have met a when meeting b
            else        return false;   //have not met a when meeting b
        }
    }
    return false;
}

inline bool file_exists(const std::string name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

/*
 * Split a string with a specific separator
 * */
void split_string(const string &origin, vector<string> &partitions, const string &separator) {
    string::size_type pos1, pos2;
    pos2 = origin.find(separator);
    pos1 = 0;
    while(string::npos != pos2) {
        partitions.push_back(origin.substr(pos1, pos2-pos1));

        pos1 = pos2 + separator.size();
        pos2 = origin.find(separator, pos1);
    }
    if(pos1 != origin.length())
        partitions.push_back(origin.substr(pos1));
}

/*
 * Transform a double value to string, removing all the 0's at the tail
 * */
static std::string double_to_string(double val) {
    auto res = std::to_string(val);
    const std::string format("$1");
    try {
        std::regex r("(\\d*)\\.0{6}|");
        std::regex r2("(\\d*\\.{1}0*[^0]+)0*");
        res = std::regex_replace(res, r2, format);
        res = std::regex_replace(res, r, format);
    }
    catch (const std::exception & e) {
        return res;
    }
    return res;
}

/*
 * List the files of a directory
 * */
vector<string> list_files(string dir_name) {
    vector<string> res_list;
    DIR *dp;
    struct dirent *dirp;
    if((dp = opendir(dir_name.c_str())) == nullptr) {
        log_error("Cannot open the directory: %s", dir_name.c_str());
        exit(1);
    }
    while((dirp = readdir(dp)) != nullptr) {
        if (!strcmp(dirp->d_name, ".") || !strcmp(dirp->d_name, "..")) //skip "." and ".."
            continue;
        res_list.emplace_back(string(dirp->d_name));
    }
    closedir(dp);
    return res_list;
}

string get_file_name_from_addr(string file_addr) {
    string::size_type iPos = file_addr.find_last_of('/')+1;
    string file_name = file_addr.substr(iPos, file_addr.length() - iPos);
    return file_name;
}

string file_cut_subfix(string file_name) {
    string file_name_cut = file_name.substr(0,file_name.rfind("."));
    return file_name_cut;
}

/*find lower_bound of needle in range haystacks[hay_start, hay_end)*/
template<typename NeedleType, typename HaystackType, typename CntType>
CntType lower_bound_galloping( //faster than binary search
        NeedleType needle, HaystackType *haystacks,
        CntType hay_start, CntType hay_end) {
    long long int lo = hay_start;
    long long int hi = hay_start;
    long long int scale = 8;
    while ((hi < hay_end) && (haystacks[hi] < needle)) {
        lo = hi;
        hi += scale;
        scale <<= 3;
    }
    if (hi > hay_end-1) hi = hay_end-1;

    while (lo <= hi) { //lo and hi must be signed values
        scale = lo + (hi - lo)/2;
        if (needle > haystacks[scale]) lo = scale + 1;
        else hi = scale - 1;
    }
    return lo;
}

/*find upper_bound of needle in range haystacks[hay_start, hay_end)*/
template<typename NeedleType, typename HaystackType, typename CntType>
CntType upper_bound_galloping( //faster than binary search
        NeedleType needle, HaystackType *haystacks,
        CntType hay_start, CntType hay_end) {
    long long lo = hay_start;
    long long hi = hay_start;
    long long scale = 8;

    //haystacks[hi] can be equal to needle in upper_bound
    while ((hi < hay_end) && (haystacks[hi] <= needle)) {
        lo = hi;
        hi += scale;
        scale <<= 3;
    }
    if (hi > hay_end-1) hi = hay_end-1;

    while (lo <= hi) { //lo and hi must be signed values
        scale = lo + (hi - lo)/2;
        assert(scale <= hay_end);
        if (needle >= haystacks[scale]) lo = scale + 1;
        else hi = scale - 1;
    }
    return lo;
}