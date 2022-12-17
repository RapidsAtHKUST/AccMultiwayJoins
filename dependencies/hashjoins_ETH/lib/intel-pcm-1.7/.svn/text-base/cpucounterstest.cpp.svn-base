/*
Copyright (c) 2009-2011, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// written by Roman Dementiev,
//            Thomas Willhalm
//            Patrick Ungerer


/*!     \file cpucounterstest.cpp
        \brief Example of using CPU counters: implements a simple performance counter monitoring utility
*/

#include <iostream>
#ifdef _MSC_VER
#pragma warning(disable : 4996) // for sprintf
#include <windows.h>
#include "windriver.h"
#else
#include <unistd.h>
#include <signal.h>
#endif
#include <math.h>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <assert.h>
#include "cpucounters.h"


#ifdef _MSC_VER
#define CHAR_TYPE TCHAR
#else
#define CHAR_TYPE char
#endif


#define SIZE (10000000)
#define DELAY 1 // in seconds

using namespace std;

template <class IntType>
std::string unit_format(IntType n)
{
    char buffer[1024];
    if (n <= 9999ULL)
    {
        sprintf(buffer, "%4d  ", int32(n));
        return buffer;
    }
    if (n <= 9999999ULL)
    {
        sprintf(buffer, "%4d K", int32(n / 1000ULL));
        return buffer;
    }
    if (n <= 9999999999ULL)
    {
        sprintf(buffer, "%4d M", int32(n / 1000000ULL));
        return buffer;
    }
    if (n <= 9999999999999ULL)
    {
        sprintf(buffer, "%4d G", int32(n / 1000000000ULL));
        return buffer;
    }

    sprintf(buffer, "%4d T", int32(n / (1000000000ULL * 1000ULL)));
    return buffer;
}


void print_help(CHAR_TYPE * prog_name)
{
        #ifdef _MSC_VER
    cout << " Usage: pcm <delay>|\"external_program parameters\"|--help|--uninstallDriver|--installDriver <other options>" << endl;
        #else
    cout << " Usage: pcm <delay>|\"external_program parameters\"|--help <other options>" << endl;
        #endif
    cout << endl;
    cout << " \n Other options:" << endl;
    cout << " -nc or --nocores or /nc => hides core related output" << endl;
    cout << " -ns or --nosockets or /ns => hides socket related output" << endl;
    cout << " -nsys or --nosystem or /nsys => hides system related output" << endl;
    cout << " Example:  pcm.x 1 -nc -ns " << endl;
    cout << endl;
}


#ifdef _MSC_VER
BOOL cleanup(DWORD)
{
    PCM::getInstance()->cleanup();
    return FALSE;
}
#else
void cleanup(int s)
{
    signal(s, SIG_IGN);
    PCM::getInstance()->cleanup();
    exit(0);
}
#endif

void MySleep(int delay)
{
#ifdef _MSC_VER
    if(delay) Sleep(delay*1000);
#else
    ::sleep(delay);
#endif
}

void MySleepMs(int delay_ms)
{
#ifdef _MSC_VER
    if(delay_ms) Sleep(delay_ms);
#else
    ::sleep(delay_ms/1000);
#endif
}


void MySystem(CHAR_TYPE * sysCmd)
{
    std::cout << "\n Executing \"";
#ifdef _MSC_VER
    std::wcout << sysCmd;
    std::cout << "\" command:\n" << std::endl;
    _wsystem(sysCmd);
#else
    std::cout << sysCmd;
    std::cout << "\" command:\n" << std::endl;
    system(sysCmd);
#endif
}

struct null_stream : public std::streambuf
{
    void overflow(char) { }
};

#ifdef _MSC_VER
int _tmain(int argc, CHAR_TYPE * argv[])
#else
int main(int argc, CHAR_TYPE * argv[])
#endif
{
    #ifdef PCM_FORCE_SILENT
    null_stream nullStream1, nullStream2;
    std::cout.rdbuf(&nullStream1);
    std::cerr.rdbuf(&nullStream2);
    #endif

    cout << endl;
    cout << " Intel(r) Performance Counter Monitor" << endl;
    cout << endl;
    cout << " Copyright (c) 2009-2011 Intel Corporation" << endl;
    // cout << " for internal usage only; not for production use; do not distribute" << endl;
    cout << endl;
        #ifdef _MSC_VER
    // Increase the priority a bit to improve context switching delays on Windows
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);

    TCHAR driverPath[1024];
    GetCurrentDirectory(1024, driverPath);
    wcscat(driverPath, L"\\msr.sys");

    SetConsoleCtrlHandler((PHANDLER_ROUTINE)cleanup, TRUE);
        #else
    signal(SIGINT, cleanup);
    signal(SIGKILL, cleanup);
    signal(SIGTERM, cleanup);
        #endif

    int delay = 1;

    CHAR_TYPE * sysCmd = NULL;
    bool show_core_output = true;
    bool show_socket_output = true;
    bool show_system_output = true;


    if (argc >= 2)

    {
        #ifdef _MSC_VER
        if (wcscmp(argv[1], L"--help") == 0 ||
            wcscmp(argv[1], L"-h") == 0 ||
            wcscmp(argv[1], L"/h") == 0)
        #else
        if (strcmp(argv[1], "--help") == 0 ||
            strcmp(argv[1], "-h") == 0 ||
            strcmp(argv[1], "/h") == 0)
        #endif
        {
            print_help(argv[0]);
            return -1;
        }

        for (int l = 1; l < argc; l++)
        {
            if (argc >= 2)
                #ifdef _MSC_VER
                if (wcscmp(argv[l], L"--help") == 0 ||
                    wcscmp(argv[l], L"-h") == 0 ||
                    wcscmp(argv[1], L"/h") == 0)
                #else
                if (strcmp(argv[l], "--help") == 0 ||
                    strcmp(argv[l], "-h") == 0 ||
                    strcmp(argv[l], "/h") == 0)
                #endif
                {
                    print_help(argv[l]);
                    return -1;
                }

                else

        #ifdef _MSC_VER
                if (wcscmp(argv[l], L"--nocores") == 0 ||
                    wcscmp(argv[l], L"-nc") == 0 ||
                    wcscmp(argv[l], L"/nc") == 0)
                #else
                if (strcmp(argv[l], "--nocores") == 0 ||
                    strcmp(argv[l], "-nc") == 0 ||
                    strcmp(argv[l], "/nc") == 0)
                #endif
                {
                    show_core_output = false;
                }
                else

                #ifdef _MSC_VER
                if (wcscmp(argv[l], L"--nosockets") == 0 ||
                    wcscmp(argv[l], L"-ns") == 0 ||
                    wcscmp(argv[l], L"/ns") == 0)
                #else
                if (strcmp(argv[l], "--nosockets") == 0 ||
                    strcmp(argv[l], "-ns") == 0 ||
                    strcmp(argv[l], "/ns") == 0)
                #endif
                {
                    show_socket_output = false;
                }

                else

    #ifdef _MSC_VER
                if (wcscmp(argv[l], L"--nosystem") == 0 ||
                    wcscmp(argv[l], L"-nsys") == 0 ||
                    wcscmp(argv[l], L"/nsys") == 0)
            #else
                if (strcmp(argv[l], "--nosystem") == 0 ||
                    strcmp(argv[l], "-nsys") == 0 ||
                    strcmp(argv[l], "/nsys") == 0)
           #endif
                {
                    show_system_output = false;
                }
        }


                #ifdef _MSC_VER
        if (wcscmp(argv[1], L"--uninstallDriver") == 0)
        {
            Driver tmpDrvObject;
            tmpDrvObject.uninstall();
            cout << "msr.sys driver has been uninstalled. You might need to reboot the system to make this effective." << endl;
            return 0;
        }
        if (wcscmp(argv[1], L"--installDriver") == 0)
        {
            Driver tmpDrvObject;
            if (!tmpDrvObject.start(driverPath))
            {
                cout << "Can not access CPU counters" << endl;
                cout << "You must have signed msr.sys driver in your current directory and have administrator rights to run this program" << endl;
                return -1;
            }
            return 0;
        }
                #endif

                #ifdef _MSC_VER
        delay = _wtoi(argv[1]);
                #else
        delay = atoi(argv[1]);
                #endif
        if (delay <= 0)
        {
            sysCmd = argv[1];
        }
    }
    if (argc == 1)
    {
        print_help(argv[0]);
        return -1;
    }

        #ifdef _MSC_VER
    // WARNING: This driver code (msr.sys) is only for testing purposes, not for production use
    Driver drv;
    // drv.stop();     // restart driver (usually not needed)
    if (!drv.start(driverPath))
    {
		cout << "Cannot access CPU counters" << endl;
		cout << "You must have signed msr.sys driver in your current directory and have administrator rights to run this program" << endl;
    }
        #endif

    PCM * m = PCM::getInstance();
    PCM::ErrorCode status = m->program();
    switch (status)
    {
    case PCM::Success:
        break;
    case PCM::MSRAccessDenied:
        cout << "Access to Intel(r) Performance Counter Monitor has denied (no MSR or PCI CFG space access)." << endl;
        return -1;
    case PCM::PMUBusy:
        cout << "Access to Intel(r) Performance Counter Monitor has denied (Performance Monitoring Unit is occupied by other application). Try to stop the application that uses PMU." << endl;
        cout << "Alternatively you can try to reset PMU configuration at your own risk. Try to reset? (y/n)" << endl;
        char yn;
        std::cin >> yn;
        if ('y' == yn)
        {
            m->resetPMU();
            cout << "PMU configuration has been reset. Try to rerun the program again." << endl;
        }
        return -1;
    default:
        cout << "Access to Intel(r) Performance Counter Monitor has denied (Unknown error)." << endl;
        return -1;
    }

    unsigned int i;

    CoreCounterState * cstates1 = new  CoreCounterState[PCM::getInstance()->getNumCores()];
    CoreCounterState * cstates2 = new  CoreCounterState[PCM::getInstance()->getNumCores()];
    SocketCounterState * sktstate1 = new SocketCounterState[m->getNumSockets()];
    SocketCounterState * sktstate2 = new SocketCounterState[m->getNumSockets()];
    SystemCounterState sstate1, sstate2;
    const int cpu_model = m->getCPUModel();
	uint64 TimeAfterSleep = 0;

    sstate1 = getSystemCounterState();

    if (show_socket_output)
        for (i = 0; i < m->getNumSockets(); ++i)
            sktstate1[i] = getSocketCounterState(i);

    if (show_core_output)
        for (i = 0; i < m->getNumCores(); ++i)
            cstates1[i] = getCoreCounterState(i);


    while (1)
    {
        cout << std::flush;

  	  #ifdef _MSC_VER
		int delay_ms = delay * 1000;
         // compensate slow Windows console output
        if(TimeAfterSleep) delay_ms -= (uint32)(m->getTickCount() - TimeAfterSleep);
        if(delay_ms < 0) delay_ms = 0;
      #else
        int delay_ms = delay * 1000;
      #endif

        if (sysCmd)
        {
            MySystem(sysCmd);
        }
        else
        {
            MySleepMs(delay_ms);
        }

		TimeAfterSleep = m->getTickCount();

        sstate2 = getSystemCounterState();
        for (i = 0; i < m->getNumSockets(); ++i)
            sktstate2[i] = getSocketCounterState(i);
        for (i = 0; i < m->getNumCores(); ++i)
            cstates2[i] = getCoreCounterState(i);


        cout << "\n";
        cout << " EXEC  : instructions per nominal CPU cycle" << "\n";
        cout << " IPC   : instructions per CPU cycle" << "\n";
        cout << " FREQ  : relation to nominal CPU frequency='unhalted clock ticks'/'invariant timer ticks' (includes Intel Turbo Boost)" << "\n";
        if (cpu_model != PCM::ATOM) cout << " AFREQ : relation to nominal CPU frequency while in active state (not in power-saving C state)='unhalted clock ticks'/'invariant timer ticks while in C0-state'  (includes Intel Turbo Boost)" << "\n";
        if (cpu_model != PCM::ATOM) cout << " L3MISS: L3 cache misses " << "\n";
        if (cpu_model == PCM::ATOM)
            cout << " L2MISS: L2 cache misses " << "\n";
        else
            cout << " L2MISS: L2 cache misses (including other core's L2 cache *hits*) " << "\n";
        if (cpu_model != PCM::ATOM) cout << " L3HIT : L3 cache hit ratio (0.00-1.00)" << "\n";
        cout << " L2HIT : L2 cache hit ratio (0.00-1.00)" << "\n";
        if (cpu_model != PCM::ATOM) cout << " L3CLK : ratio of CPU cycles lost due to L3 cache misses (0.00-1.00), in some cases could be >1.0 due to a higher memory latency" << "\n";
        if (cpu_model != PCM::ATOM) cout << " L2CLK : ratio of CPU cycles lost due to missing L2 cache but still hitting L3 cache (0.00-1.00)" << "\n";
        if (cpu_model != PCM::ATOM) cout << " READ  : bytes read from memory controller (in GBytes)" << "\n";
        if (cpu_model != PCM::ATOM) cout << " WRITE : bytes written to memory controller (in GBytes)" << "\n";
        cout << "\n";
        cout << "\n";
        cout.precision(2);
        cout << std::fixed;
        if (cpu_model == PCM::ATOM)
            cout << " Core (SKT) | EXEC | IPC  | FREQ | L2MISS | L2HIT " << "\n" << "\n";
        else
            cout << " Core (SKT) | EXEC | IPC  | FREQ  | AFREQ | L3MISS | L2MISS | L3HIT | L2HIT | L3CLK | L2CLK  | READ  | WRITE " << "\n" << "\n";


        if (show_core_output)
        {
            for (i = 0; i < m->getNumCores(); ++i)
            {
                if (cpu_model != PCM::ATOM)
                    cout << " " << setw(3) << i << "   " << setw(2) << m->getSocketId(i) <<
                    "     " << getExecUsage(cstates1[i], cstates2[i]) <<
                    "   " << getIPC(cstates1[i], cstates2[i]) <<
                    "   " << getRelativeFrequency(cstates1[i], cstates2[i]) <<
                    "    " << getActiveRelativeFrequency(cstates1[i], cstates2[i]) <<
                    "    " << unit_format(getL3CacheMisses(cstates1[i], cstates2[i])) <<
                    "   " << unit_format(getL2CacheMisses(cstates1[i], cstates2[i])) <<
                    "    " << getL3CacheHitRatio(cstates1[i], cstates2[i]) <<
                    "    " << getL2CacheHitRatio(cstates1[i], cstates2[i]) <<
                    "    " << getCyclesLostDueL3CacheMisses(cstates1[i], cstates2[i]) <<
                    "    " << getCyclesLostDueL2CacheMisses(cstates1[i], cstates2[i]) <<
                    "     N/A     N/A" <<
                    "\n";
                else
                    cout << " " << setw(3) << i << "   " << setw(2) << m->getSocketId(i) <<
                    "     " << getExecUsage(cstates1[i], cstates2[i]) <<
                    "   " << getIPC(cstates1[i], cstates2[i]) <<
                    "   " << getRelativeFrequency(cstates1[i], cstates2[i]) <<
                    "   " << unit_format(getL2CacheMisses(cstates1[i], cstates2[i])) <<
                    "    " << getL2CacheHitRatio(cstates1[i], cstates2[i]) <<
                    "\n";
            }
        }
        if (show_socket_output)
        {
            if (m->getNumSockets() > 1)
            {
                cout << "------------------------------------------------------------------------------------------------------------" << "\n";
                for (i = 0; i < m->getNumSockets(); ++i)
                    cout << " SKT   " << setw(2) << i <<
                    "     " << getExecUsage(sktstate1[i], sktstate2[i]) <<
                    "   " << getIPC(sktstate1[i], sktstate2[i]) <<
                    "   " << getRelativeFrequency(sktstate1[i], sktstate2[i]) <<
                    "    " << getActiveRelativeFrequency(sktstate1[i], sktstate2[i]) <<
                    "    " << unit_format(getL3CacheMisses(sktstate1[i], sktstate2[i])) <<
                    "   " << unit_format(getL2CacheMisses(sktstate1[i], sktstate2[i])) <<
                    "    " << getL3CacheHitRatio(sktstate1[i], sktstate2[i]) <<
                    "    " << getL2CacheHitRatio(sktstate1[i], sktstate2[i]) <<
                    "    " << getCyclesLostDueL3CacheMisses(sktstate1[i], sktstate2[i]) <<
                    "    " << getCyclesLostDueL2CacheMisses(sktstate1[i], sktstate2[i]) <<
                    "    " << getBytesReadFromMC(sktstate1[i], sktstate2[i]) / double(1024ULL * 1024ULL * 1024ULL) <<
                    "    " << getBytesWrittenToMC(sktstate1[i], sktstate2[i]) / double(1024ULL * 1024ULL * 1024ULL) <<
                    "\n";
            }
        }
        cout << "------------------------------------------------------------------------------------------------------------" << "\n";

        if (show_system_output)
        {
            if (cpu_model != PCM::ATOM)
            {
                cout << " TOTAL  *     " << getExecUsage(sstate1, sstate2) <<
                "   " << getIPC(sstate1, sstate2) <<
                "   " << getRelativeFrequency(sstate1, sstate2) <<
                "    " << getActiveRelativeFrequency(sstate1, sstate2) <<
                "    " << unit_format(getL3CacheMisses(sstate1, sstate2)) <<
                "   " << unit_format(getL2CacheMisses(sstate1, sstate2)) <<
                "    " << getL3CacheHitRatio(sstate1, sstate2) <<
                "    " << getL2CacheHitRatio(sstate1, sstate2) <<
                "    " << getCyclesLostDueL3CacheMisses(sstate1, sstate2) <<
                "    " << getCyclesLostDueL2CacheMisses(sstate1, sstate2);
                if (cpu_model == PCM::SANDY_BRIDGE)
                    cout << "     N/A     N/A";
                else
                    cout << "    " << getBytesReadFromMC(sstate1, sstate2) / double(1024ULL * 1024ULL * 1024ULL) <<
                    "    " << getBytesWrittenToMC(sstate1, sstate2) / double(1024ULL * 1024ULL * 1024ULL);
                cout << "\n";
            }
            else
                cout << " TOTAL  *     " << getExecUsage(sstate1, sstate2) <<
                "   " << getIPC(sstate1, sstate2) <<
                "   " << getRelativeFrequency(sstate1, sstate2) <<
                "   " << unit_format(getL2CacheMisses(sstate1, sstate2)) <<
                "    " << getL2CacheHitRatio(sstate1, sstate2) <<
                "\n";
        }

        if (show_system_output)
        {
            cout << "\n" << " Instructions retired: " << unit_format(getInstructionsRetired(sstate1, sstate2)) << " ; Active cycles: " << unit_format(getCycles(sstate1, sstate2)) << " ; Time (TSC): " << unit_format(getInvariantTSC(cstates1[0], cstates2[0])) << "ticks ; C0 (active,non-halted) core residency: "<< (getC0Residency(sstate1, sstate2)*100.)<<" %\n";
            cout << "\n" << " PHYSICAL CORE IPC                 : " << getCoreIPC(sstate1, sstate2) << " => corresponds to " << 100. * (getCoreIPC(sstate1, sstate2) / double(m->getMaxIPC())) << " % utilization for cores in active state";
            cout << "\n" << " Instructions per nominal CPU cycle: " << getTotalExecUsage(sstate1, sstate2) << " => corresponds to " << 100. * (getTotalExecUsage(sstate1, sstate2) / double(m->getMaxIPC())) << " % core utilization over time interval" << "\n";
        }

        if (show_socket_output)
        {
            if (m->getNumSockets() > 1) // QPI info only for multi socket systems
            {
                cout << "\n" << "Intel(r) QPI data traffic estimation in bytes (data traffic coming to CPU/socket through QPI links):" << "\n" << "\n";


                const uint32 qpiLinks = (uint32)m->getQPILinksPerSocket();

                cout << "              ";
                for (i = 0; i < qpiLinks; ++i)
                    cout << " QPI" << i << "    ";

                if (cpu_model == PCM::NEHALEM_EX || cpu_model == PCM::WESTMERE_EX)
                {
                    cout << "| ";
                    for (i = 0; i < qpiLinks; ++i)
                        cout << " QPI" << i << "  ";
                }

                cout << "\n" << "----------------------------------------------------------------------------------------------" << "\n";


                for (i = 0; i < m->getNumSockets(); ++i)
                {
                    cout << " SKT   " << setw(2) << i << "     ";
                    for (uint32 l = 0; l < qpiLinks; ++l)
                        cout << unit_format(getIncomingQPILinkBytes(i, l, sstate1, sstate2)) << "   ";

                    if (cpu_model == PCM::NEHALEM_EX || cpu_model == PCM::WESTMERE_EX)
                    {
                        cout << "|  ";
                        for (uint32 l = 0; l < qpiLinks; ++l)
                            cout << setw(3) << int(100. * getIncomingQPILinkUtilization(i, l, sstate1, sstate2)) << "%   ";
                    }

                    cout << "\n";
                }
            }
        }

        if (show_system_output)
        {
            cout << "----------------------------------------------------------------------------------------------" << "\n";

            if (m->getNumSockets() > 1) // QPI info only for multi socket systems
                cout << "Total QPI incoming data traffic: " << unit_format(getAllIncomingQPILinkBytes(sstate1, sstate2)) << "     QPI data traffic/Memory controller traffic: " << getQPItoMCTrafficRatio(sstate1, sstate2) << "\n";
        }

        if (show_socket_output)
        {
            if (m->getNumSockets() > 1 && (cpu_model == PCM::NEHALEM_EX || cpu_model == PCM::WESTMERE_EX)) // QPI info only for multi socket systems
            {
                cout << "\n" << "Intel(r) QPI traffic estimation in bytes (data and non-data traffic outgoing from CPU/socket through QPI links):" << "\n" << "\n";


                const uint32 qpiLinks = (uint32)m->getQPILinksPerSocket();

                cout << "              ";
                for (i = 0; i < qpiLinks; ++i)
                    cout << " QPI" << i << "    ";


                cout << "| ";
                for (i = 0; i < qpiLinks; ++i)
                    cout << " QPI" << i << "  ";


                cout << "\n" << "----------------------------------------------------------------------------------------------" << "\n";


                for (i = 0; i < m->getNumSockets(); ++i)
                {
                    cout << " SKT   " << setw(2) << i << "     ";
                    for (uint32 l = 0; l < qpiLinks; ++l)
                        cout << unit_format(getOutgoingQPILinkBytes(i, l, sstate1, sstate2)) << "   ";

                    cout << "|  ";
                    for (uint32 l = 0; l < qpiLinks; ++l)
                        cout << setw(3) << int(100. * getOutgoingQPILinkUtilization(i, l, sstate1, sstate2)) << "%   ";

                    cout << "\n";
                }

                cout << "----------------------------------------------------------------------------------------------" << "\n";
            }
        }

        if (show_system_output && (cpu_model == PCM::NEHALEM_EX || cpu_model == PCM::WESTMERE_EX))
        {
            cout << "Total QPI outgoing data and non-data traffic: " << unit_format(getAllOutgoingQPILinkBytes(sstate1, sstate2)) << "\n";
        }


        // sanity checks
        if (cpu_model == PCM::ATOM)
        {
            assert(getNumberOfCustomEvents(0, sstate1, sstate2) == getL2CacheMisses(sstate1, sstate2) + getL2CacheHits(sstate1, sstate2));
            assert(getNumberOfCustomEvents(1, sstate1, sstate2) == getL2CacheMisses(sstate1, sstate2));
        }
        else
        {
            assert(getNumberOfCustomEvents(0, sstate1, sstate2) == getL3CacheMisses(sstate1, sstate2));
            assert(getNumberOfCustomEvents(1, sstate1, sstate2) == getL3CacheHitsNoSnoop(sstate1, sstate2));
            assert(getNumberOfCustomEvents(2, sstate1, sstate2) == getL3CacheHitsSnoop(sstate1, sstate2));
            assert(getNumberOfCustomEvents(3, sstate1, sstate2) == getL2CacheHits(sstate1, sstate2));
        }

        std::swap(sstate1, sstate2);
        std::swap(sktstate1, sktstate2);
        std::swap(cstates1, cstates2);

        if (sysCmd)
        {
            // system() call removes PCM cleanup handler. need to do clean up explicitely
            PCM::getInstance()->cleanup();
            break;
        }
    }


    delete[] cstates1;
    delete[] cstates2;
    delete[] sktstate1;
    delete[] sktstate2;

    return 0;
}
