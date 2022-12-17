/*
Copyright (c) 2009-2011, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// written by Roman Dementiev
//            Otto Bruggeman
//            Thomas Willhalm
//            Pat Fay
//

// Changelog:
//
// - Added #ifdef __LP64__ check at line 568 to compile on 32-bit (cagri)
//

//#define PCM_TEST_FALLBACK_TO_ATOM

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#ifdef INTELPCM_EXPORTS
// Intelpcm.h includes cpucounters.h
#include "Intelpcm.dll\Intelpcm.h"
#else
#include "cpucounters.h"
#endif
#include "msr.h"
#include "pci.h"
#include "types.h"

#ifdef _MSC_VER
#include <intrin.h>
#include <windows.h>
#include <tchar.h>
#include "winring0/OlsApiInit.h"
#else
#include <pthread.h>
#include <errno.h>
#endif

#include <string.h>

#include <map>

#ifdef _MSC_VER

HMODULE hOpenLibSys = NULL;

bool PCM::initWinRing0Lib()
{
	const BOOL result = InitOpenLibSys(&hOpenLibSys);
	
	if(result == FALSE) hOpenLibSys = NULL;

	return result==TRUE;
}

class SystemWideLock
{
    HANDLE globalMutex;

public:
    SystemWideLock()
    {
        globalMutex = CreateMutex(NULL, FALSE,
                                  L"Global\\Intel(r) Performance Counter Monitor instance create/destroy lock");
        // lock
        WaitForSingleObject(globalMutex, INFINITE);
    }
    ~SystemWideLock()
    {
        // unlock
        ReleaseMutex(globalMutex);
    }
};
#else

class SystemWideLock
{
    const char * globalSemaphoreName;
    sem_t * globalSemaphore;

public:
    SystemWideLock() : globalSemaphoreName("/Intel(r) Performance Counter Monitor instance create-destroy lock")
    {
        while (1)
        {
            globalSemaphore = sem_open(globalSemaphoreName, O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO, 1);
            if (SEM_FAILED == globalSemaphore)
            {
                if (EACCES == errno)
                {
                    std::cout << "PCM Error, do not have permissions to open semaphores in /dev/shm/. Waiting one second and retrying..." << std::endl;
                    sleep(1);
                }
            }
            else
            {
                break;         // success
            }
        }
        sem_wait(globalSemaphore);
    }
    ~SystemWideLock()
    {
        sem_post(globalSemaphore);
    }
};
#endif

PCM * PCM::instance = NULL;

int bitCount(uint64 n)
{
    int count = 0;
    while (n)
    {
        count += (int)(n & 0x00000001);
        n >>= 1;
    }
    return count;
}

PCM * PCM::getInstance()
{
    // no lock here
    if (instance) return instance;

    SystemWideLock lock;
    if (instance) return instance;

    return instance = new PCM();
}

uint32 build_bit_ui(int beg, int end)
{
    uint32 myll = 0;
    if (end == 31)
    {
        myll = (uint32)(-1);
    }
    else
    {
        myll = (1 << (end + 1)) - 1;
    }
    myll = myll >> beg;
    return myll;
}

uint32 extract_bits_ui(uint32 myin, uint32 beg, uint32 end)
{
    uint32 myll = 0;
    uint32 beg1, end1;

    // Let the user reverse the order of beg & end.
    if (beg <= end)
    {
        beg1 = beg;
        end1 = end;
    }
    else
    {
        beg1 = end;
        end1 = beg;
    }
    myll = myin >> beg1;
    myll = myll & build_bit_ui(beg1, end1);
    return myll;
}

uint64 build_bit(uint32 beg, uint32 end)
{
    uint64 myll = 0;
    if (end == 63)
    {
        myll = (uint64)(-1);
    }
    else
    {
        myll = (1LL << (end + 1)) - 1;
    }
    myll = myll >> beg;
    return myll;
}

uint64 extract_bits(uint64 myin, uint32 beg, uint32 end)
{
    uint64 myll = 0;
    uint32 beg1, end1;

    // Let the user reverse the order of beg & end.
    if (beg <= end)
    {
        beg1 = beg;
        end1 = end;
    }
    else
    {
        beg1 = end;
        end1 = beg;
    }
    myll = myin >> beg1;
    myll = myll & build_bit(beg1, end1);
    return myll;
}

PCM::PCM() :
    MSR(NULL),
    mode(INVALID_MODE),
    cpu_family(-1),
    cpu_model(-1),
    threads_per_core(0),
    num_cores(0),
    num_sockets(0),
    core_gen_counter_num_max(0),
    core_gen_counter_num_used(0), // 0 means no core gen counters used
    core_gen_counter_width(0),
    core_fixed_counter_num_max(0),
    core_fixed_counter_num_used(0),
    core_fixed_counter_width(0),
    uncore_gen_counter_num_max(8),
    uncore_gen_counter_num_used(0),
    uncore_gen_counter_width(48),
    uncore_fixed_counter_num_max(1),
    uncore_fixed_counter_num_used(0),
    uncore_fixed_counter_width(48),
    perfmon_version(0),
    perfmon_config_anythread(1),
    qpi_speed(0)
{
    int32 i = 0;
    char buffer[1024];
    const char * UnsupportedMessage = "Error: unsupported processor. Only Intel(R) processors are supported (Atom(R) and microarchitecture codename Nehalem, Westmere and Sandy Bridge).";

        #ifdef _MSC_VER
    // version for Windows
    int cpuinfo[4];
    int max_cpuid;
    __cpuid(cpuinfo, 0);
    memset(buffer, 0, 1024);
    ((int *)buffer)[0] = cpuinfo[1];
    ((int *)buffer)[1] = cpuinfo[3];
    ((int *)buffer)[2] = cpuinfo[2];
    if (strncmp(buffer, "GenuineIntel", 4 * 3) != 0)
    {
        std::cout << UnsupportedMessage << std::endl;
        return;
    }
    max_cpuid = cpuinfo[0];

    __cpuid(cpuinfo, 1);
    cpu_family = (((cpuinfo[0]) >> 8) & 0xf) | ((cpuinfo[0] & 0xf00000) >> 16);
    cpu_model = (((cpuinfo[0]) & 0xf0) >> 4) | ((cpuinfo[0] & 0xf0000) >> 12);


    if (max_cpuid >= 0xa)
    {
        // get counter related info
        __cpuid(cpuinfo, 0xa);
        perfmon_version = extract_bits_ui(cpuinfo[0], 0, 7);
        core_gen_counter_num_max = extract_bits_ui(cpuinfo[0], 8, 15);
        core_gen_counter_width = extract_bits_ui(cpuinfo[0], 16, 23);
        if (perfmon_version > 1)
        {
            core_fixed_counter_num_max = extract_bits_ui(cpuinfo[3], 0, 4);
            core_fixed_counter_width = extract_bits_ui(cpuinfo[3], 5, 12);
        }
    }

    if (cpu_family != 6)
    {
        std::cout << UnsupportedMessage << " CPU Family: " << cpu_family << std::endl;
        return;
    }
    if(cpu_model == NEHALEM_EP_2) cpu_model = NEHALEM_EP;

    if (cpu_model != NEHALEM_EP
        && cpu_model != NEHALEM_EX
        && cpu_model != WESTMERE_EP
        && cpu_model != WESTMERE_EX
        && cpu_model != ATOM
        && cpu_model != CLARKDALE
        && cpu_model != SANDY_BRIDGE
        )
    {
        std::cout << UnsupportedMessage << " CPU Model: " << cpu_model << std::endl;
/* FOR TESTING PURPOSES ONLY */
#ifdef PCM_TEST_FALLBACK_TO_ATOM
        std::cout << "Fall back to ATOM functionality." << std::endl;
        if (1)
            cpu_model = ATOM;
        else
#endif
        return;
    }

#ifdef COMPILE_FOR_WINDOWS_7
    DWORD GroupStart[5];     // at most 4 groups on Windows 7
    GroupStart[0] = 0;
    GroupStart[1] = GetMaximumProcessorCount(0);
    GroupStart[2] = GroupStart[1] + GetMaximumProcessorCount(1);
    GroupStart[3] = GroupStart[2] + GetMaximumProcessorCount(2);
    GroupStart[4] = GetMaximumProcessorCount(ALL_PROCESSOR_GROUPS);
    if (GroupStart[3] + GetMaximumProcessorCount(3) != GetMaximumProcessorCount(ALL_PROCESSOR_GROUPS))
    {
        std::cout << "Error in processor group size counting (1)" << std::endl;
        return;
    }
    char * slpi = new char[sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)];
    DWORD len = sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX);
    DWORD res = GetLogicalProcessorInformationEx(RelationAll, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)slpi, &len);

    while (res == FALSE)
    {
        delete[] slpi;

        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER)
        {
            slpi = new char[len];
            res = GetLogicalProcessorInformationEx(RelationAll, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)slpi, &len);
        }
        else
        {
            std::cout << "Error in Windows function 'GetLogicalProcessorInformationEx': " <<
            GetLastError() << std::endl;
            return;
        }
    }

    char * base_slpi = slpi;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX pi = NULL;

    for ( ; slpi < base_slpi + len; slpi += pi->Size)
    {
        pi = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)slpi;
        if (pi->Relationship == RelationProcessorCore)
        {
            threads_per_core = (pi->Processor.Flags == LTP_PC_SMT) ? 2 : 1;
            // std::cout << "thr per core: "<< threads_per_core << std::endl;
            num_cores += threads_per_core;
        }
    }


    if (num_cores != GetMaximumProcessorCount(ALL_PROCESSOR_GROUPS))
    {
        std::cout << "Error in processor group size counting: " << num_cores << "!=" << GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) << std::endl;
        return;
    }

    topology.resize(num_cores);

    slpi = base_slpi;
    pi = NULL;

    for ( ; slpi < base_slpi + len; slpi += pi->Size)
    {
        pi = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)slpi;
        if (pi->Relationship == RelationNumaNode)
        {
            ++num_sockets;
            for (unsigned int c = 0; c < (unsigned int)num_cores; ++c)
            {
                // std::cout << "c:"<<c<<" GroupStart[slpi->NumaNode.GroupMask.Group]: "<<GroupStart[slpi->NumaNode.GroupMask.Group]<<std::endl;
                if (c < GroupStart[pi->NumaNode.GroupMask.Group] || c >= GroupStart[(pi->NumaNode.GroupMask.Group) + 1])
                {
                    //std::cout <<"core "<<c<<" is not in group "<< slpi->NumaNode.GroupMask.Group << std::endl;
                    continue;
                }
                if ((1LL << (c - GroupStart[pi->NumaNode.GroupMask.Group])) & pi->NumaNode.GroupMask.Mask)
                {
                    topology[c].core_id = c;
                    topology[c].os_id = c;
                    topology[c].socket = pi->NumaNode.NodeNumber;
                    // std::cout << "Core "<< c <<" is in NUMA node "<< topology[c].socket << " and belongs to processor group " << slpi->NumaNode.GroupMask.Group <<std::endl;
                }
            }
        }
    }

    delete[] base_slpi;

#else
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION * slpi = new SYSTEM_LOGICAL_PROCESSOR_INFORMATION[1];
    DWORD len = sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    DWORD res = GetLogicalProcessorInformation(slpi, &len);

    while (res == FALSE)
    {
        delete[] slpi;

        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER)
        {
            slpi = new SYSTEM_LOGICAL_PROCESSOR_INFORMATION[len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)];
            res = GetLogicalProcessorInformation(slpi, &len);
        }
        else
        {
            std::cout << "Error in Windows function 'GetLogicalProcessorInformation': " <<
            GetLastError() << std::endl;
            return;
        }
    }

    for (i = 0; i < (int32)(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)); ++i)
    {
        if (slpi[i].Relationship == RelationProcessorCore)
        {
            //std::cout << "Physical core found, mask: "<<slpi[i].ProcessorMask<< std::endl;
            threads_per_core = bitCount(slpi[i].ProcessorMask);
            num_cores += threads_per_core;
        }
    }
    topology.resize(num_cores);

    for (i = 0; i < (int32)(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)); ++i)
    {
        if (slpi[i].Relationship == RelationNumaNode)
        {
            //std::cout << "NUMA node "<<slpi[i].NumaNode.NodeNumber<<" cores: "<<slpi[i].ProcessorMask<< std::endl;
            ++num_sockets;
            for (int c = 0; c < num_cores; ++c)
            {
                if ((1LL << c) & slpi[i].ProcessorMask)
                {
                    topology[c].core_id = c;
                    topology[c].os_id = c;
                    topology[c].socket = slpi[i].NumaNode.NodeNumber;
                    //std::cout << "Core "<< c <<" is in NUMA node "<< topology[c].socket << std::endl;
                }
            }
        }
    }

    delete[] slpi;

#endif

        #else
    // for Linux

    // open /proc/cpuinfo
    FILE * f_cpuinfo = fopen("/proc/cpuinfo", "r");
    if (!f_cpuinfo)
    {
        std::cout << "Can not open /proc/cpuinfo file." << std::endl;
        return;
    }

    TopologyEntry entry;
    typedef std::map<uint32, uint32> socketIdMap_type;
    socketIdMap_type socketIdMap;

    while (0 != fgets(buffer, 1024, f_cpuinfo))
    {
        if (strncmp(buffer, "processor", sizeof("processor") - 1) == 0)
        {
            if (entry.os_id >= 0)
            {
                topology.push_back(entry);
                if (entry.socket == 0 && entry.core_id == 0) ++threads_per_core;
            }
            sscanf(buffer, "processor\t: %d", &entry.os_id);
            //std::cout << "os_core_id: "<<entry.os_id<< std::endl;
            continue;
        }
        if (strncmp(buffer, "physical id", sizeof("physical id") - 1) == 0)
        {
            sscanf(buffer, "physical id\t: %d", &entry.socket);
            //std::cout << "physical id: "<<entry.socket<< std::endl;
            socketIdMap[entry.socket] = 0;
            continue;
        }
        if (strncmp(buffer, "core id", sizeof("core id") - 1) == 0)
        {
            sscanf(buffer, "core id\t: %d", &entry.core_id);
            //std::cout << "core id: "<<entry.core_id<< std::endl;
            continue;
        }

        if (entry.os_id == 0)        // one time support checks
        {
            if (strncmp(buffer, "vendor_id", sizeof("vendor_id") - 1) == 0)
            {
                char vendor_id[1024];
                sscanf(buffer, "vendor_id\t: %s", vendor_id);
                if (0 != strcmp(vendor_id, "GenuineIntel"))
                {
                    std::cout << UnsupportedMessage << std::endl;
                    return;
                }
                continue;
            }
            if (strncmp(buffer, "cpu family", sizeof("cpu family") - 1) == 0)
            {
                sscanf(buffer, "cpu family\t: %d", &cpu_family);
                if (cpu_family != 6)
                {
                    std::cout << UnsupportedMessage << std::endl;
                    return;
                }
                continue;
            }
            if (strncmp(buffer, "model", sizeof("model") - 1) == 0)
            {
                sscanf(buffer, "model\t: %d", &cpu_model);
                if(cpu_model == NEHALEM_EP_2) cpu_model = NEHALEM_EP;
                if (cpu_model != NEHALEM_EP
                    && cpu_model != NEHALEM_EX
                    && cpu_model != WESTMERE_EP
                    && cpu_model != WESTMERE_EX
                    && cpu_model != ATOM
                    && cpu_model != CLARKDALE
                    && cpu_model != SANDY_BRIDGE
                    )
                {
                    std::cout << UnsupportedMessage << " CPU model: " << cpu_model << std::endl;

/* FOR TESTING PURPOSES ONLY */
#ifdef PCM_TEST_FALLBACK_TO_ATOM
                    std::cout << "Fall back to ATOM functionality." << std::endl;
                    cpu_model = ATOM;
                    continue;
#endif
                    return;
                }
                continue;
            }
        }
    }

    if (entry.os_id >= 0)
    {
        topology.push_back(entry);
        if (entry.socket == 0 && entry.core_id == 0) ++threads_per_core;
    }

    num_cores = topology.size();
    num_sockets = (std::max)(socketIdMap.size(), (size_t)1);

    fclose(f_cpuinfo);

    socketIdMap_type::iterator s = socketIdMap.begin();
    for (uint sid = 0; s != socketIdMap.end(); ++s)
    {
        s->second = sid++;
    }

    for (int i = 0; i < num_cores; ++i)
    {
        topology[i].socket = socketIdMap[topology[i].socket];
    }

#if 0
    std::cout << "Number of socket ids: " << socketIdMap.size() << "\n";
    std::cout << "Topology:\nsocket os_id core_id\n";
    for (int i = 0; i < num_cores; ++i)
    {
        std::cout << topology[i].socket << " " << topology[i].os_id << " " << topology[i].core_id << std::endl;
    }
#endif

#ifdef __LP64__
#define cpuid(func, ax, bx, cx, dx) \
    __asm__ __volatile__ ("cpuid" : \
                          "=a" (ax), "=b" (bx), "=c" (cx), "=d" (dx) : "a" (func));
#else
#define cpuid(func, ax, bx, cx, dx) \
    __asm__ __volatile__ ("cpuid" : \
                          "=a" (ax), "=S" (bx), "=c" (cx), "=d" (dx) : "a" (func));
#endif

    uint32 eax, ebx, ecx, edx, max_cpuid;
    cpuid(0, eax, ebx, ecx, edx);
    max_cpuid = eax;
    if (max_cpuid >= 0xa)
    {
        // get counter related info
        cpuid(0xa, eax, ebx, ecx, edx);
        perfmon_version = extract_bits_ui(eax, 0, 7);
        core_gen_counter_num_max = extract_bits_ui(eax, 8, 15);
        core_gen_counter_width = extract_bits_ui(eax, 16, 23);
        if (perfmon_version > 1)
        {
            core_fixed_counter_num_max = extract_bits_ui(edx, 0, 4);
            core_fixed_counter_width = extract_bits_ui(edx, 5, 12);
        }
    }


        #endif

    std::cout << "Num cores: " << num_cores << std::endl;
    std::cout << "Num sockets: " << num_sockets << std::endl;
    std::cout << "Threads per core: " << threads_per_core << std::endl;
    std::cout << "Core PMU (perfmon) version: " << perfmon_version << std::endl;
    std::cout << "Number of core PMU generic (programmable) counters: " << core_gen_counter_num_max << std::endl;
    std::cout << "Width of generic (programmable) counters: " << core_gen_counter_width << " bits" << std::endl;
    if (perfmon_version > 1)
    {
        std::cout << "Number of core PMU fixed counters: " << core_fixed_counter_num_max << std::endl;
        std::cout << "Width of fixed counters: " << core_fixed_counter_width << " bits" << std::endl;
    }


    MSR = new MsrHandle *[num_cores];

    try
    {
        for (i = 0; i < num_cores; ++i)
            MSR[i] = new MsrHandle(i);
    }
    catch (...)
    {
        // failed
        for (int j = 0; j < i; j++)
            delete MSR[j];
        delete[] MSR;
        MSR = NULL;

        std::cerr << "Can not access CPUs Model Specific Registers (MSRs)." << std::endl;
                #ifdef _MSC_VER
        std::cerr << "You must have signed msr.sys driver in your current directory and have administrator rights to run this program." << std::endl;
                #else
        std::cerr << "Try to execute 'modprobe msr' as root user and then" << std::endl;
        std::cerr << "you also must have read and write permissions for /dev/cpu/*/msr devices (the 'chown' command can help)." << std::endl;
                #endif
    }

    if (MSR)
    {
        uint64 freq;
        MSR[0]->read(PLATFORM_INFO_ADDR, &freq);

        const uint64 bus_freq = (cpu_model == SANDY_BRIDGE) ? (100000000ULL) : (133333333ULL);

        nominal_frequency = ((freq >> 8) & 255) * bus_freq;
        std::cout << "Nominal core frequency: " << nominal_frequency << " Hz" << std::endl;
    }

}

PCM::~PCM()
{
    SystemWideLock lock;
    if (instance)
    {
        if (MSR)
        {
            for (int i = 0; i < num_cores; ++i)
                if (MSR[i]) delete MSR[i];
            delete[] MSR;
        }
        instance = NULL;
    }
}

bool PCM::good()
{
    return MSR != NULL;
}

class TemporalThreadAffinity  // speedup trick for Linux
{
#ifndef _MSC_VER
    cpu_set_t old_affinity;
    TemporalThreadAffinity(); // forbiden

public:
    TemporalThreadAffinity(uint32 core_id)
    {
        pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &old_affinity);

        cpu_set_t new_affinity;
        CPU_ZERO(&new_affinity);
        CPU_SET(core_id, &new_affinity);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &new_affinity);
    }
    ~TemporalThreadAffinity()
    {
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &old_affinity);
    }
#else // not implemented
    TemporalThreadAffinity(); // forbiden

public:
    TemporalThreadAffinity(uint32) { }
#endif
};

PCM::ErrorCode PCM::program(PCM::ProgramMode mode_, void * parameter_)
{
    SystemWideLock lock;

    if (!MSR) return PCM::MSRAccessDenied;

    ///std::cout << "Checking for other instances of PCM..." << std::endl;
        #ifdef _MSC_VER
#if 1
    numInstancesSemaphore = CreateSemaphore(NULL, 0, 1 << 20, L"Global\\Number of running Intel Processor Counter Monitor instances");
    if (!numInstancesSemaphore)
    {
        std::cout << "Error in Windows function 'CreateSemaphore': " << GetLastError() << std::endl;
        return PCM::UnknownError;
    }
    LONG prevValue = 0;
    if (!ReleaseSemaphore(numInstancesSemaphore, 1, &prevValue))
    {
        std::cout << "Error in Windows function 'ReleaseSemaphore': " << GetLastError() << std::endl;
        return PCM::UnknownError;
    }
    if (prevValue > 0)  // already programmed since another instance exists
    {
        std::cout << "Number of monitor instances: " << (prevValue + 1) << std::endl;
        return PCM::Success;
    }
#endif
        #else
    numInstancesSemaphore = sem_open("/Number of running Intel(r) Performance Counter Monitor instances", O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO, 0);
    if (SEM_FAILED == numInstancesSemaphore)
    {
        if (EACCES == errno)
            std::cout << "PCM Error, do not have permissions to open semaphores in /dev/shm/. Clean up them." << std::endl;
        return PCM::UnknownError;
    }
    sem_post(numInstancesSemaphore);
    int curValue = 0;
    sem_getvalue(numInstancesSemaphore, &curValue);
    if (curValue > 1)  // already programmed since another instance exists
    {
        std::cout << "Number of PCM instances: " << curValue << std::endl;
        return PCM::Success;
    }
        #endif

    if (PMUinUse())
    {
        decrementInstanceSemaphore();
        return PCM::PMUBusy;
    }

    mode = mode_;
    
    // copy custom event descriptions
    if (mode == CUSTOM_CORE_EVENTS)
    {
        CustomCoreEventDescription * pDesc = (CustomCoreEventDescription *)parameter_;
        coreEventDesc[0] = pDesc[0];
        coreEventDesc[1] = pDesc[1];
        if (cpu_model != ATOM)
        {
            coreEventDesc[2] = pDesc[2];
            coreEventDesc[3] = pDesc[3];
            core_gen_counter_num_used = 4;
        }
        else
            core_gen_counter_num_used = 2;
    }
    else
    {
        if (cpu_model == ATOM)
        {
            coreEventDesc[0].event_number = ARCH_LLC_REFERENCE_EVTNR;
            coreEventDesc[0].umask_value = ARCH_LLC_REFERENCE_UMASK;
            coreEventDesc[1].event_number = ARCH_LLC_MISS_EVTNR;
            coreEventDesc[1].umask_value = ARCH_LLC_MISS_UMASK;
            core_gen_counter_num_used = 2;
        }
        else if (SANDY_BRIDGE == cpu_model)
        {
            coreEventDesc[0].event_number = ARCH_LLC_MISS_EVTNR;
            coreEventDesc[0].umask_value = ARCH_LLC_MISS_UMASK;
            coreEventDesc[1].event_number = MEM_LOAD_UOPS_LLC_HIT_RETIRED_XSNP_NONE_EVTNR;
            coreEventDesc[1].umask_value = MEM_LOAD_UOPS_LLC_HIT_RETIRED_XSNP_NONE_UMASK;
            coreEventDesc[2].event_number = MEM_LOAD_UOPS_LLC_HIT_RETIRED_XSNP_HITM_EVTNR;
            coreEventDesc[2].umask_value = MEM_LOAD_UOPS_LLC_HIT_RETIRED_XSNP_HITM_UMASK;
            coreEventDesc[3].event_number = MEM_LOAD_UOPS_RETIRED_L2_HIT_EVTNR;
            coreEventDesc[3].umask_value = MEM_LOAD_UOPS_RETIRED_L2_HIT_UMASK;
            core_gen_counter_num_used = 4;
        }
        else
        {   // Nehalem or Westmere
	    if(NEHALEM_EP == cpu_model || WESTMERE_EP == cpu_model || CLARKDALE == cpu_model)
	    {
		coreEventDesc[0].event_number = MEM_LOAD_RETIRED_L3_MISS_EVTNR;
		coreEventDesc[0].umask_value = MEM_LOAD_RETIRED_L3_MISS_UMASK;
	    }
	    else
	    {
		coreEventDesc[0].event_number = ARCH_LLC_MISS_EVTNR;
		coreEventDesc[0].umask_value = ARCH_LLC_MISS_UMASK;
	    }
            coreEventDesc[1].event_number = MEM_LOAD_RETIRED_L3_UNSHAREDHIT_EVTNR;
            coreEventDesc[1].umask_value = MEM_LOAD_RETIRED_L3_UNSHAREDHIT_UMASK;
            coreEventDesc[2].event_number = MEM_LOAD_RETIRED_L2_HITM_EVTNR;
            coreEventDesc[2].umask_value = MEM_LOAD_RETIRED_L2_HITM_UMASK;
            coreEventDesc[3].event_number = MEM_LOAD_RETIRED_L2_HIT_EVTNR;
            coreEventDesc[3].umask_value = MEM_LOAD_RETIRED_L2_HIT_UMASK;
            core_gen_counter_num_used = 4;
        }
    }

    core_fixed_counter_num_used = 3;
    
    ExtendedCustomCoreEventDescription * pExtDesc = (ExtendedCustomCoreEventDescription *)parameter_;
    
    if(EXT_CUSTOM_CORE_EVENTS == mode_ && pExtDesc && pExtDesc->gpCounterCfg)
    {
        core_gen_counter_num_used = (std::min)(core_gen_counter_num_used,pExtDesc->nGPCounters);
    }

    for (int i = 0; i < num_cores; ++i)
    {
        // program core counters

        TemporalThreadAffinity tempThreadAffinity(i); // speedup trick for Linux

        // disable counters while programming
        MSR[i]->write(IA32_CR_PERF_GLOBAL_CTRL, 0);

        FixedEventControlRegister ctrl_reg;
        MSR[i]->read(IA32_CR_FIXED_CTR_CTRL, &ctrl_reg.value);

	
	if(EXT_CUSTOM_CORE_EVENTS == mode_ && pExtDesc && pExtDesc->fixedCfg)
	{
	  ctrl_reg = *(pExtDesc->fixedCfg);
	}
	else
	{
	  ctrl_reg.fields.os0 = 1;
	  ctrl_reg.fields.usr0 = 1;
	  ctrl_reg.fields.any_thread0 = 0;
	  ctrl_reg.fields.enable_pmi0 = 0;

	  ctrl_reg.fields.os1 = 1;
	  ctrl_reg.fields.usr1 = 1;
	  ctrl_reg.fields.any_thread1 = (perfmon_version >= 3) ? 1 : 0;         // sum the nuber of cycles from both logical cores on one physical core
	  ctrl_reg.fields.enable_pmi1 = 0;

	  ctrl_reg.fields.os2 = 1;
	  ctrl_reg.fields.usr2 = 1;
	  ctrl_reg.fields.any_thread2 = (perfmon_version >= 3) ? 1 : 0;         // sum the nuber of cycles from both logical cores on one physical core
	  ctrl_reg.fields.enable_pmi2 = 0;	
	}

        MSR[i]->write(IA32_CR_FIXED_CTR_CTRL, ctrl_reg.value);

        EventSelectRegister event_select_reg;

        for (uint32 j = 0; j < core_gen_counter_num_used; ++j)
        {
	    if(EXT_CUSTOM_CORE_EVENTS == mode_ && pExtDesc && pExtDesc->gpCounterCfg)
	    {
	      event_select_reg = pExtDesc->gpCounterCfg[j];
	    }
	    else
	    {
	  
	      MSR[i]->read(IA32_PERFEVTSEL0_ADDR + j, &event_select_reg.value);
	      
	      event_select_reg.fields.event_select = coreEventDesc[j].event_number;
	      event_select_reg.fields.umask = coreEventDesc[j].umask_value;
	      event_select_reg.fields.usr = 1;
	      event_select_reg.fields.os = 1;
	      event_select_reg.fields.edge = 0;
	      event_select_reg.fields.pin_control = 0;
	      event_select_reg.fields.apic_int = 0;
	      event_select_reg.fields.any_thread = 0;
	      event_select_reg.fields.enable = 1;
	      event_select_reg.fields.invert = 0;
	      event_select_reg.fields.cmask = 0;
	    
	    }
            MSR[i]->write(IA32_PMC0 + j, 0);
            MSR[i]->write(IA32_PERFEVTSEL0_ADDR + j, event_select_reg.value);
        }

        // start counting, enable all (4 programmable + 3 fixed) counters
        uint64 value = (1ULL << 0) + (1ULL << 1) + (1ULL << 2) + (1ULL << 3) + (1ULL << 32) + (1ULL << 33) + (1ULL << 34);

        if (cpu_model == ATOM)       // Atom has only 2 programmable counters
            value = (1ULL << 0) + (1ULL << 1) + (1ULL << 32) + (1ULL << 33) + (1ULL << 34);

        MSR[i]->write(IA32_CR_PERF_GLOBAL_CTRL, value);


        // program uncore counters
#if 1

#define CPUCNT_INIT_THE_REST_OF_EVTCNT \
    unc_event_select_reg.fields.occ_ctr_rst = 1; \
    unc_event_select_reg.fields.edge = 0; \
    unc_event_select_reg.fields.enable_pmi = 0; \
    unc_event_select_reg.fields.enable = 1; \
    unc_event_select_reg.fields.invert = 0; \
    unc_event_select_reg.fields.cmask = 0;

        if (cpu_model == NEHALEM_EP || cpu_model == WESTMERE_EP || cpu_model == CLARKDALE)
        {
            uncore_gen_counter_num_used = 8;

            UncoreEventSelectRegister unc_event_select_reg;

            MSR[i]->read(MSR_UNCORE_PERFEVTSEL0_ADDR, &unc_event_select_reg.value);

            unc_event_select_reg.fields.event_select = UNC_QMC_WRITES_FULL_ANY_EVTNR;
            unc_event_select_reg.fields.umask = UNC_QMC_WRITES_FULL_ANY_UMASK;

            CPUCNT_INIT_THE_REST_OF_EVTCNT

                MSR[i]->write(MSR_UNCORE_PERFEVTSEL0_ADDR, unc_event_select_reg.value);


            MSR[i]->read(MSR_UNCORE_PERFEVTSEL1_ADDR, &unc_event_select_reg.value);

            unc_event_select_reg.fields.event_select = UNC_QMC_NORMAL_READS_ANY_EVTNR;
            unc_event_select_reg.fields.umask = UNC_QMC_NORMAL_READS_ANY_UMASK;

            CPUCNT_INIT_THE_REST_OF_EVTCNT

                MSR[i]->write(MSR_UNCORE_PERFEVTSEL1_ADDR, unc_event_select_reg.value);


            MSR[i]->read(MSR_UNCORE_PERFEVTSEL2_ADDR, &unc_event_select_reg.value);
            unc_event_select_reg.fields.event_select = UNC_QHL_REQUESTS_EVTNR;
            unc_event_select_reg.fields.umask = UNC_QHL_REQUESTS_IOH_READS_UMASK;
            CPUCNT_INIT_THE_REST_OF_EVTCNT
                MSR[i]->write(MSR_UNCORE_PERFEVTSEL2_ADDR, unc_event_select_reg.value);

            MSR[i]->read(MSR_UNCORE_PERFEVTSEL3_ADDR, &unc_event_select_reg.value);
            unc_event_select_reg.fields.event_select = UNC_QHL_REQUESTS_EVTNR;
            unc_event_select_reg.fields.umask = UNC_QHL_REQUESTS_IOH_WRITES_UMASK;
            CPUCNT_INIT_THE_REST_OF_EVTCNT
                MSR[i]->write(MSR_UNCORE_PERFEVTSEL3_ADDR, unc_event_select_reg.value);

            MSR[i]->read(MSR_UNCORE_PERFEVTSEL4_ADDR, &unc_event_select_reg.value);
            unc_event_select_reg.fields.event_select = UNC_QHL_REQUESTS_EVTNR;
            unc_event_select_reg.fields.umask = UNC_QHL_REQUESTS_REMOTE_READS_UMASK;
            CPUCNT_INIT_THE_REST_OF_EVTCNT
                MSR[i]->write(MSR_UNCORE_PERFEVTSEL4_ADDR, unc_event_select_reg.value);

            MSR[i]->read(MSR_UNCORE_PERFEVTSEL5_ADDR, &unc_event_select_reg.value);
            unc_event_select_reg.fields.event_select = UNC_QHL_REQUESTS_EVTNR;
            unc_event_select_reg.fields.umask = UNC_QHL_REQUESTS_REMOTE_WRITES_UMASK;
            CPUCNT_INIT_THE_REST_OF_EVTCNT
                MSR[i]->write(MSR_UNCORE_PERFEVTSEL5_ADDR, unc_event_select_reg.value);

            MSR[i]->read(MSR_UNCORE_PERFEVTSEL6_ADDR, &unc_event_select_reg.value);
            unc_event_select_reg.fields.event_select = UNC_QHL_REQUESTS_EVTNR;
            unc_event_select_reg.fields.umask = UNC_QHL_REQUESTS_LOCAL_READS_UMASK;
            CPUCNT_INIT_THE_REST_OF_EVTCNT
                MSR[i]->write(MSR_UNCORE_PERFEVTSEL6_ADDR, unc_event_select_reg.value);

            MSR[i]->read(MSR_UNCORE_PERFEVTSEL7_ADDR, &unc_event_select_reg.value);
            unc_event_select_reg.fields.event_select = UNC_QHL_REQUESTS_EVTNR;
            unc_event_select_reg.fields.umask = UNC_QHL_REQUESTS_LOCAL_WRITES_UMASK;
            CPUCNT_INIT_THE_REST_OF_EVTCNT
                MSR[i]->write(MSR_UNCORE_PERFEVTSEL7_ADDR, unc_event_select_reg.value);


#undef CPUCNT_INIT_THE_REST_OF_EVTCNT

            // start uncore counting
            value = 255 + (1ULL << 32);           // enable all counters
            MSR[i]->write(MSR_UNCORE_PERF_GLOBAL_CTRL_ADDR, value);

            // synchronise counters
            MSR[i]->write(MSR_UNCORE_PMC0, 0);
            MSR[i]->write(MSR_UNCORE_PMC1, 0);
            MSR[i]->write(MSR_UNCORE_PMC2, 0);
            MSR[i]->write(MSR_UNCORE_PMC3, 0);
            MSR[i]->write(MSR_UNCORE_PMC4, 0);
            MSR[i]->write(MSR_UNCORE_PMC5, 0);
            MSR[i]->write(MSR_UNCORE_PMC6, 0);
            MSR[i]->write(MSR_UNCORE_PMC7, 0);
        }
        else if (cpu_model == NEHALEM_EX || cpu_model == WESTMERE_EX)
        {
            // program Beckton uncore

            if (i == 0) computeQPIspeed(i);

            value = 1 << 29ULL;           // reset all counters
            MSR[i]->write(U_MSR_PMON_GLOBAL_CTL, value);

            BecktonUncorePMUZDPCTLFVCRegister FVCreg;
            FVCreg.value = 0;
            if (cpu_model == NEHALEM_EX)
            {
                FVCreg.fields.bcmd = 0;             // rd_bcmd
                FVCreg.fields.resp = 0;             // ack_resp
                FVCreg.fields.evnt0 = 5;            // bcmd_match
                FVCreg.fields.evnt1 = 6;            // resp_match
                FVCreg.fields.pbox_init_err = 0;
            }
            else
            {
                FVCreg.fields_wsm.bcmd = 0;             // rd_bcmd
                FVCreg.fields_wsm.resp = 0;             // ack_resp
                FVCreg.fields_wsm.evnt0 = 5;            // bcmd_match
                FVCreg.fields_wsm.evnt1 = 6;            // resp_match
                FVCreg.fields_wsm.pbox_init_err = 0;
            }
            MSR[i]->write(MB0_MSR_PMU_ZDP_CTL_FVC, FVCreg.value);
            MSR[i]->write(MB1_MSR_PMU_ZDP_CTL_FVC, FVCreg.value);

            BecktonUncorePMUCNTCTLRegister CNTCTLreg;
            CNTCTLreg.value = 0;
            CNTCTLreg.fields.en = 1;
            CNTCTLreg.fields.pmi_en = 0;
            CNTCTLreg.fields.count_mode = 0;
            CNTCTLreg.fields.storage_mode = 0;
            CNTCTLreg.fields.wrap_mode = 1;
            CNTCTLreg.fields.flag_mode = 0;
            CNTCTLreg.fields.inc_sel = 0x0d;           // FVC_EV0
            MSR[i]->write(MB0_MSR_PMU_CNT_CTL_0, CNTCTLreg.value);
            MSR[i]->write(MB1_MSR_PMU_CNT_CTL_0, CNTCTLreg.value);
            CNTCTLreg.fields.inc_sel = 0x0e;           // FVC_EV1
            MSR[i]->write(MB0_MSR_PMU_CNT_CTL_1, CNTCTLreg.value);
            MSR[i]->write(MB1_MSR_PMU_CNT_CTL_1, CNTCTLreg.value);

            value = 1 + ((0x0C) << 1ULL);              // enable bit + (event select IMT_INSERTS_WR)
            MSR[i]->write(BB0_MSR_PERF_CNT_CTL_1, value);
            MSR[i]->write(BB1_MSR_PERF_CNT_CTL_1, value);

            MSR[i]->write(MB0_MSR_PERF_GLOBAL_CTL, 3); // enable two counters
            MSR[i]->write(MB1_MSR_PERF_GLOBAL_CTL, 3); // enable two counters

            MSR[i]->write(BB0_MSR_PERF_GLOBAL_CTL, 2); // enable second counter
            MSR[i]->write(BB1_MSR_PERF_GLOBAL_CTL, 2); // enable second counter

            // program R-Box to monitor QPI traffic

            // enable counting on all counters on the left side (port 0-3)
            MSR[i]->write(R_MSR_PMON_GLOBAL_CTL_7_0, 255);
            // ... on the right side (port 4-7)
            MSR[i]->write(R_MSR_PMON_GLOBAL_CTL_15_8, 255);

            // pick the event
            value = (1 << 7ULL) + (1 << 6ULL) + (1 << 2ULL); // count any (incoming) data responses
            MSR[i]->write(R_MSR_PORT0_IPERF_CFG0, value);
            MSR[i]->write(R_MSR_PORT1_IPERF_CFG0, value);
            MSR[i]->write(R_MSR_PORT4_IPERF_CFG0, value);
            MSR[i]->write(R_MSR_PORT5_IPERF_CFG0, value);

            // pick the event
            value = (1ULL << 30ULL); // count null idle flits sent
            MSR[i]->write(R_MSR_PORT0_IPERF_CFG1, value);
            MSR[i]->write(R_MSR_PORT1_IPERF_CFG1, value);
            MSR[i]->write(R_MSR_PORT4_IPERF_CFG1, value);
            MSR[i]->write(R_MSR_PORT5_IPERF_CFG1, value);

            // choose counter 0 to monitor R_MSR_PORT0_IPERF_CFG0
            MSR[i]->write(R_MSR_PMON_CTL0, 1 + 2 * (0));
            // choose counter 1 to monitor R_MSR_PORT1_IPERF_CFG0
            MSR[i]->write(R_MSR_PMON_CTL1, 1 + 2 * (6));
            // choose counter 8 to monitor R_MSR_PORT4_IPERF_CFG0
            MSR[i]->write(R_MSR_PMON_CTL8, 1 + 2 * (0));
            // choose counter 9 to monitor R_MSR_PORT5_IPERF_CFG0
            MSR[i]->write(R_MSR_PMON_CTL9, 1 + 2 * (6));

            // choose counter 2 to monitor R_MSR_PORT0_IPERF_CFG1
            MSR[i]->write(R_MSR_PMON_CTL2, 1 + 2 * (1));
            // choose counter 3 to monitor R_MSR_PORT1_IPERF_CFG1
            MSR[i]->write(R_MSR_PMON_CTL3, 1 + 2 * (7));
            // choose counter 10 to monitor R_MSR_PORT4_IPERF_CFG1
            MSR[i]->write(R_MSR_PMON_CTL10, 1 + 2 * (1));
            // choose counter 11 to monitor R_MSR_PORT5_IPERF_CFG1
            MSR[i]->write(R_MSR_PMON_CTL11, 1 + 2 * (7));

            // enable uncore TSC counter (fixed one)
            MSR[i]->write(W_MSR_PMON_GLOBAL_CTL, 1ULL << 31ULL);
            MSR[i]->write(W_MSR_PMON_FIXED_CTR_CTL, 1ULL);

            value = (1 << 28ULL) + 1;                  // enable all counters
            MSR[i]->write(U_MSR_PMON_GLOBAL_CTL, value);
        }

#endif
    }

    return PCM::Success;
}


void PCM::computeQPIspeed(int core_nr)
{
    // reset all counters
    MSR[core_nr]->write(U_MSR_PMON_GLOBAL_CTL, 1 << 29ULL);

    // enable counting on all counters on the left side (port 0-3)
    MSR[core_nr]->write(R_MSR_PMON_GLOBAL_CTL_7_0, 255);
    // disable on the right side (port 4-7)
    MSR[core_nr]->write(R_MSR_PMON_GLOBAL_CTL_15_8, 0);

    // count flits sent
    MSR[core_nr]->write(R_MSR_PORT0_IPERF_CFG0, 1ULL << 31ULL);

    // choose counter 0 to monitor R_MSR_PORT0_IPERF_CFG0
    MSR[core_nr]->write(R_MSR_PMON_CTL0, 1 + 2 * (0));

    // enable all counters
    MSR[core_nr]->write(U_MSR_PMON_GLOBAL_CTL, (1 << 28ULL) + 1);

    uint64 startFlits;
    MSR[core_nr]->read(R_MSR_PMON_CTR0, &startFlits);

    const uint64 timerGranularity = 1000000ULL; // mks
    uint64 startTSC = getTickCount(timerGranularity, core_nr);
    uint64 endTSC;
    do
    {
        endTSC = getTickCount(timerGranularity, core_nr);
    } while (endTSC - startTSC < 200000ULL); // spin for 200 ms

    uint64 endFlits;
    MSR[core_nr]->read(R_MSR_PMON_CTR0, &endFlits);

    qpi_speed = (endFlits - startFlits) * 8ULL * timerGranularity / (endTSC - startTSC);

    std::cout.precision(1);
    std::cout << std::fixed;
    std::cout << "Max QPI link speed: " << qpi_speed / (1e9) << " GBytes/second (" << qpi_speed / (2e9) << " GT/second)" << std::endl;
}

bool PCM::PMUinUse()
{
    // follow the "Performance Monitoring Unit Sharing Guide" by P. Irelan and Sh. Kuo
    for (int i = 0; i < num_cores; ++i)
    {
        //std::cout << "Core "<<i<<" exemine registers"<< std::endl;
        uint64 value;
        MSR[i]->read(IA32_CR_PERF_GLOBAL_CTRL, &value);
        // std::cout << "Core "<<i<<" IA32_CR_PERF_GLOBAL_CTRL is "<< std::hex << value << std::dec << std::endl;

        EventSelectRegister event_select_reg;

        for (uint32 j = 0; j < core_gen_counter_num_max; ++j)
        {
            MSR[i]->read(IA32_PERFEVTSEL0_ADDR + j, &event_select_reg.value);

            if (event_select_reg.fields.event_select != 0 || event_select_reg.fields.apic_int != 0)
            {
                // std::cout << "Core "<<i<<" IA32_PERFEVTSEL0_ADDR are not zeroed "<< event_select_reg.value << std::endl;
                return true;
            }
        }

        FixedEventControlRegister ctrl_reg;

        MSR[i]->read(IA32_CR_FIXED_CTR_CTRL, &ctrl_reg.value);

        if (1 == ctrl_reg.fields.os0 &&
            1 == ctrl_reg.fields.usr0 &&
            0 == ctrl_reg.fields.enable_pmi0 &&
            1 == ctrl_reg.fields.os1 &&
            1 == ctrl_reg.fields.usr1 &&
            0 == ctrl_reg.fields.enable_pmi1 &&
            1 == ctrl_reg.fields.os2 &&
            1 == ctrl_reg.fields.usr2 &&
            0 == ctrl_reg.fields.enable_pmi2
            )
            continue;             // exception: this is the modus we want in PCM


        if ((ctrl_reg.fields.os0 ||
             ctrl_reg.fields.usr0 ||
             ctrl_reg.fields.enable_pmi0 ||
             ctrl_reg.fields.os1 ||
             ctrl_reg.fields.usr1 ||
             ctrl_reg.fields.enable_pmi1 ||
             ctrl_reg.fields.os2 ||
             ctrl_reg.fields.usr2 ||
             ctrl_reg.fields.enable_pmi2)
            != 0)
        {
            // std::cout << "Core "<<i<<" fixed ctrl:"<< ctrl_reg.value << std::endl;
            return true;
        }
    }

    return false;
}

void PCM::cleanupPMU()
{
    // follow the "Performance Monitoring Unit Sharing Guide" by P. Irelan and Sh. Kuo

    for (int i = 0; i < num_cores; ++i)
    {
        // disable generic counters and continue free running counting for fixed counters
        MSR[i]->write(IA32_CR_PERF_GLOBAL_CTRL, (1ULL << 32) + (1ULL << 33) + (1ULL << 34));

        for (uint32 j = 0; j < core_gen_counter_num_max; ++j)
        {
            MSR[i]->write(IA32_PERFEVTSEL0_ADDR + j, 0);
        }
    }
}

void PCM::resetPMU()
{
    for (int i = 0; i < num_cores; ++i)
    {
        // disable all counters
        MSR[i]->write(IA32_CR_PERF_GLOBAL_CTRL, 0);

        for (uint32 j = 0; j < core_gen_counter_num_max; ++j)
        {
            MSR[i]->write(IA32_PERFEVTSEL0_ADDR + j, 0);
        }


        FixedEventControlRegister ctrl_reg;
        MSR[i]->read(IA32_CR_FIXED_CTR_CTRL, &ctrl_reg.value);
        if ((ctrl_reg.fields.os0 ||
             ctrl_reg.fields.usr0 ||
             ctrl_reg.fields.enable_pmi0 ||
             ctrl_reg.fields.os1 ||
             ctrl_reg.fields.usr1 ||
             ctrl_reg.fields.enable_pmi1 ||
             ctrl_reg.fields.os2 ||
             ctrl_reg.fields.usr2 ||
             ctrl_reg.fields.enable_pmi2)
            != 0)
            MSR[i]->write(IA32_CR_FIXED_CTR_CTRL, 0);
    }
}

void PCM::cleanup()
{
    SystemWideLock lock;

    std::cout << "Cleaning up" << std::endl;
    if (!MSR) return;

    if (decrementInstanceSemaphore())
        cleanupPMU();
}

bool PCM::decrementInstanceSemaphore()
{
    bool isLastInstance = false;
                #ifdef _MSC_VER
    WaitForSingleObject(numInstancesSemaphore, 0);

    DWORD res = WaitForSingleObject(numInstancesSemaphore, 0);
    if (res == WAIT_TIMEOUT)
    {
        // I have the last instance of monitor

        isLastInstance = true;

        CloseHandle(numInstancesSemaphore);
    }
    else if (res == WAIT_OBJECT_0)
    {
        ReleaseSemaphore(numInstancesSemaphore, 1, NULL);

        // std::cout << "Someone else is running monitor instance, no cleanup needed"<< std::endl;
    }
    else
    {
        // unknown error
        std::cout << "ERROR: Bad semaphore. Performed cleanup twice?" << std::endl;
    }
        #else
    sem_wait(numInstancesSemaphore);
    int curValue = -1;
    sem_getvalue(numInstancesSemaphore, &curValue);
    if (curValue == 0)
    {
        // I have the last instance of monitor

        isLastInstance = true;

        // std::cout << "I am the last one"<< std::endl;
    }
        #endif

    return isLastInstance;
}

uint64 PCM::getTickCount(uint64 multiplier, uint32 core)
{
    return (multiplier * getInvariantTSC(CoreCounterState(), getCoreCounterState(core))) / getNominalFrequency();
}

uint64 PCM::getTickCountRDTSCP(uint64 multiplier)
{
	uint64 result = 0;

#ifdef _MSC_VER
	// Windows
	#if _MSC_VER>= 1600
	unsigned int Aux;
	result = __rdtscp(&Aux);
	#endif
#else
	// Linux
	uint32 high = 0, low = 0;
	asm volatile (
	   "rdtscp\n\t"
	   "mov %%edx, %0\n\t"
	   "mov %%eax, %1\n\t":
	   "=r" (high), "=r" (low) :: "%rax", "%rcx", "%rdx");
	result = low + (uint64(high)<<32ULL);
#endif

	result = (multiplier*result)/getNominalFrequency();

	return result;
}

SystemCounterState getSystemCounterState()
{
    PCM * inst = PCM::getInstance();
    SystemCounterState result;
    if (inst) result = inst->getSystemCounterState();
    return result;
}

SocketCounterState getSocketCounterState(uint32 socket)
{
    PCM * inst = PCM::getInstance();
    SocketCounterState result;
    if (inst) result = inst->getSocketCounterState(socket);
    return result;
}

CoreCounterState getCoreCounterState(uint32 core)
{
    PCM * inst = PCM::getInstance();
    CoreCounterState result;
    if (inst) result = inst->getCoreCounterState(core);
    return result;
}

void BasicCounterState::readAndAggregate(MsrHandle * msr)
{
    uint64 cInstRetiredAny = 0, cCpuClkUnhaltedThread = 0, cCpuClkUnhaltedRef = 0;
    uint64 cL3Miss = 0;
    uint64 cL3UnsharedHit = 0;
    uint64 cL2HitM = 0;
    uint64 cL2Hit = 0;
    uint64 cInvariantTSC = 0;

    TemporalThreadAffinity tempThreadAffinity(msr->getCoreId()); // speedup trick for Linux

    msr->read(INST_RETIRED_ANY_ADDR, &cInstRetiredAny);
    msr->read(CPU_CLK_UNHALTED_THREAD_ADDR, &cCpuClkUnhaltedThread);
    msr->read(CPU_CLK_UNHALTED_REF_ADDR, &cCpuClkUnhaltedRef);

    uint32 cpu_model = PCM::getInstance()->getCPUModel();
    switch (cpu_model)
    {
    case PCM::WESTMERE_EP:
    case PCM::NEHALEM_EP:
    case PCM::NEHALEM_EX:
    case PCM::WESTMERE_EX:
    case PCM::CLARKDALE:
    case PCM::SANDY_BRIDGE:
        msr->read(IA32_PMC0, &cL3Miss);
        msr->read(IA32_PMC1, &cL3UnsharedHit);
        msr->read(IA32_PMC2, &cL2HitM);
        msr->read(IA32_PMC3, &cL2Hit);
        break;
    case PCM::ATOM:
        msr->read(IA32_PMC0, &cL3Miss);         // for Atom mapped to ArchLLCRef field
        msr->read(IA32_PMC1, &cL3UnsharedHit);  // for Atom mapped to ArchLLCMiss field
        break;
    }

    msr->read(IA32_TIME_STAMP_COUNTER, &cInvariantTSC);

    InstRetiredAny += cInstRetiredAny;
    CpuClkUnhaltedThread += cCpuClkUnhaltedThread;
    CpuClkUnhaltedRef += cCpuClkUnhaltedRef;
    L3Miss += cL3Miss;
    L3UnsharedHit += cL3UnsharedHit;
    L2HitM += cL2HitM;
    L2Hit += cL2Hit;
    InvariantTSC += cInvariantTSC;
}

void UncoreCounterState::readAndAggregate(MsrHandle * msr)
{
    TemporalThreadAffinity tempThreadAffinity(msr->getCoreId()); // speedup trick for Linux

    uint32 cpu_model = PCM::getInstance()->getCPUModel();
    switch (cpu_model)
    {
    case PCM::WESTMERE_EP:
    case PCM::NEHALEM_EP:
    {
        uint64 cUncMCFullWrites = 0;
        uint64 cUncMCNormalReads = 0;
        msr->read(MSR_UNCORE_PMC0, &cUncMCFullWrites);
        msr->read(MSR_UNCORE_PMC1, &cUncMCNormalReads);
        UncMCFullWrites += cUncMCFullWrites;
        UncMCNormalReads += cUncMCNormalReads;
    }
    break;
    case PCM::NEHALEM_EX:
    case PCM::WESTMERE_EX:
    {
        uint64 cUncMCNormalReads = 0;
        msr->read(MB0_MSR_PMU_CNT_0, &cUncMCNormalReads);
        UncMCNormalReads += cUncMCNormalReads;
        msr->read(MB1_MSR_PMU_CNT_0, &cUncMCNormalReads);
        UncMCNormalReads += cUncMCNormalReads;

        uint64 cUncMCFullWrites = 0;                         // really good approximation of
        msr->read(BB0_MSR_PERF_CNT_1, &cUncMCFullWrites);
        UncMCFullWrites += cUncMCFullWrites;
        msr->read(BB1_MSR_PERF_CNT_1, &cUncMCFullWrites);
        UncMCFullWrites += cUncMCFullWrites;
    }
    break;
    default:;
    }
}

SystemCounterState PCM::getSystemCounterState()
{
    SystemCounterState result;
    if (MSR)
    {
        for (int32 core = 0; core < num_cores; ++core)
            result.readAndAggregate(MSR[core]);

        {
            const uint32 cores_per_socket = num_cores / num_sockets;
            result.UncMCFullWrites /= cores_per_socket;
            result.UncMCNormalReads /= cores_per_socket;
        }

        std::vector<bool> SocketProcessed(num_sockets, false);
        if (cpu_model == PCM::NEHALEM_EX || cpu_model == PCM::WESTMERE_EX)
        {
            for (int32 core = 0; core < num_cores; ++core)
            {
                uint32 s = topology[core].socket;

                if (!SocketProcessed[s])
                {
                    TemporalThreadAffinity tempThreadAffinity(core); // speedup trick for Linux

                    // incoming data responses from QPI link 0
                    MSR[core]->read(R_MSR_PMON_CTR1, &(result.incomingQPIPackets[s][0]));
                    // incoming data responses from QPI link 1 (yes, from CTR0)
                    MSR[core]->read(R_MSR_PMON_CTR0, &(result.incomingQPIPackets[s][1]));
                    // incoming data responses from QPI link 2
                    MSR[core]->read(R_MSR_PMON_CTR8, &(result.incomingQPIPackets[s][2]));
                    // incoming data responses from QPI link 3
                    MSR[core]->read(R_MSR_PMON_CTR9, &(result.incomingQPIPackets[s][3]));

                    // outgoing idle flits from QPI link 0
                    MSR[core]->read(R_MSR_PMON_CTR3, &(result.outgoingQPIIdleFlits[s][0]));
                    // outgoing idle flits from QPI link 1 (yes, from CTR0)
                    MSR[core]->read(R_MSR_PMON_CTR2, &(result.outgoingQPIIdleFlits[s][1]));
                    // outgoing idle flits from QPI link 2
                    MSR[core]->read(R_MSR_PMON_CTR10, &(result.outgoingQPIIdleFlits[s][2]));
                    // outgoing idle flits from QPI link 3
                    MSR[core]->read(R_MSR_PMON_CTR11, &(result.outgoingQPIIdleFlits[s][3]));

                    if (core == 0) MSR[core]->read(W_MSR_PMON_FIXED_CTR, &(result.uncoreTSC));

                    SocketProcessed[s] = true;
                }
            }
        }
        else if ((cpu_model == PCM::NEHALEM_EP || cpu_model == PCM::WESTMERE_EP))
        {
            if (num_sockets == 2)
            {
                uint32 SCore[2] = { 0, 0 };
                uint64 Total_Reads[2] = { 0, 0 };
                uint64 Total_Writes[2] = { 0, 0 };
                uint64 IOH_Reads[2] = { 0, 0 };
                uint64 IOH_Writes[2] = { 0, 0 };
                uint64 Remote_Reads[2] = { 0, 0 };
                uint64 Remote_Writes[2] = { 0, 0 };
                uint64 Local_Reads[2] = { 0, 0 };
                uint64 Local_Writes[2] = { 0, 0 };

                while (topology[SCore[0]].socket != 0) ++(SCore[0]);
                while (topology[SCore[1]].socket != 1) ++(SCore[1]);

                for (int s = 0; s < 2; ++s)
                {
                    TemporalThreadAffinity tempThreadAffinity(SCore[s]); // speedup trick for Linux

                    MSR[SCore[s]]->read(MSR_UNCORE_PMC0, &Total_Writes[s]);
                    MSR[SCore[s]]->read(MSR_UNCORE_PMC1, &Total_Reads[s]);
                    MSR[SCore[s]]->read(MSR_UNCORE_PMC2, &IOH_Reads[s]);
                    MSR[SCore[s]]->read(MSR_UNCORE_PMC3, &IOH_Writes[s]);
                    MSR[SCore[s]]->read(MSR_UNCORE_PMC4, &Remote_Reads[s]);
                    MSR[SCore[s]]->read(MSR_UNCORE_PMC5, &Remote_Writes[s]);
                    MSR[SCore[s]]->read(MSR_UNCORE_PMC6, &Local_Reads[s]);
                    MSR[SCore[s]]->read(MSR_UNCORE_PMC7, &Local_Writes[s]);
                }

#if 1
                // compute Remote_Reads differently
                for (int s = 0; s < 2; ++s)
                {
                    uint64 total = Total_Writes[s] + Total_Reads[s];
                    uint64 rem = IOH_Reads[s]
                                 + IOH_Writes[s]
                                 + Local_Reads[s]
                                 + Local_Writes[s]
                                 + Remote_Writes[s];
                    Remote_Reads[s] = total - rem;
                }
#endif


                // only an estimation (lower bound) - does not count NT stores correctly
                result.incomingQPIPackets[0][0] = Remote_Reads[1] + Remote_Writes[0];
                result.incomingQPIPackets[0][1] = IOH_Reads[0];
                result.incomingQPIPackets[1][0] = Remote_Reads[0] + Remote_Writes[1];
                result.incomingQPIPackets[1][1] = IOH_Reads[1];
            }
            else
            {
                // for a single socket systems no information is available
                result.incomingQPIPackets[0][0] = 0;
            }
        }
    }
    return result;
}

SocketCounterState PCM::getSocketCounterState(uint32 socket)
{
    SocketCounterState result;
    if (MSR)
    {
        for (int32 core = 0; core < num_cores; ++core)
            if (topology[core].socket == socket)
                result.readAndAggregate(MSR[core]);

        {
            const uint32 cores_per_socket = num_cores / num_sockets;
            result.UncMCFullWrites /= cores_per_socket;
            result.UncMCNormalReads /= cores_per_socket;
        }
    }
    return result;
}


CoreCounterState PCM::getCoreCounterState(uint32 core)
{
    CoreCounterState result;
    if (MSR) result.readAndAggregate(MSR[core]);

    return result;
}

uint32 PCM::getNumCores()
{
    return num_cores;
}

uint32 PCM::getNumSockets()
{
    return num_sockets;
}

uint32 PCM::getThreadsPerCore()
{
    return threads_per_core;
}

bool PCM::getSMT()
{
    return threads_per_core > 1;
}

uint64 PCM::getNominalFrequency()
{
    return nominal_frequency;
}

