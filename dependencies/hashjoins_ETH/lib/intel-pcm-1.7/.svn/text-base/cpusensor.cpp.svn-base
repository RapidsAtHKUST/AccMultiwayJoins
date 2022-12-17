//
// monitor CPU conters for ksysguard
/*
Copyright (c) 2009-2011, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// contact: Thomas Willhalm, Patrick Ungerer
//
// This program is not a tutorial on how to write nice interpreters
// but a proof of concept on using ksysguard with performance counters
//

/*!     \file cpusensor.cpp
        \brief Example of using CPU counters: implements a graphical plugin for KDE ksysguard
*/

#include <iostream>
#include <string>
#include <sstream>
#include "cpuasynchcounter.h"

using namespace std;

int main()
{
    AsynchronCounterState counters;

    cout << "CPU counter sensor" << endl;
    cout << "(C) 2010 Intel Corp." << endl;
    // cout << "Intel internal only - do not distribute" << endl;
    cout << "ksysguardd 1.2.0" << endl;
    cout << "ksysguardd> ";

    while (1)
    {
        string s;
        cin >> s;

        // list counters
        if (s == "monitors") {
            for (int i = 0; i < counters.getNumCores(); ++i) {
                for (int a = 0; a < counters.getNumSockets(); ++a)
                    if (a == counters.getSocketId(i)) {
                        cout << "Socket" << a << "/CPU" << i << "/Frequency\tfloat" << endl;
                        cout << "Socket" << a << "/CPU" << i << "/IPC\tfloat" << endl;
                        cout << "Socket" << a << "/CPU" << i << "/L2CacheHitRatio\tfloat" << endl;
                        cout << "Socket" << a << "/CPU" << i << "/L3CacheHitRatio\tfloat" << endl;
                        cout << "Socket" << a << "/CPU" << i << "/L2CacheMisses\tinteger" << endl;
                        cout << "Socket" << a << "/CPU" << i << "/L3CacheMisses\tinteger" << endl;
                    }
            }
            for (int a = 0; a < counters.getNumSockets(); ++a) {
                cout << "Socket" << a << "/BytesReadFromMC\tfloat" << endl;
                cout << "Socket" << a << "/BytesWrittenToMC\tfloat" << endl;
                cout << "Socket" << a << "/Frequency\tfloat" << endl;
                cout << "Socket" << a << "/IPC\tfloat" << endl;
                cout << "Socket" << a << "/L2CacheHitRatio\tfloat" << endl;
                cout << "Socket" << a << "/L3CacheHitRatio\tfloat" << endl;
                cout << "Socket" << a << "/L2CacheMisses\tinteger" << endl;
                cout << "Socket" << a << "/L3CacheMisses\tinteger" << endl;
            }
            for (int a = 0; a < counters.getNumSockets(); ++a) {
                for (int l = 0; l < counters.getQPILinksPerSocket(); ++l)
                    cout << "Socket" << a << "/BytesIncomingToQPI" << l << "\tfloat" << endl;
            }

            cout << "QPI_Traffic\tfloat" << endl;
            cout << "Frequency\tfloat" << endl;
            cout << "IPC\tfloat" << endl;       //double check output
            cout << "L2CacheHitRatio\tfloat" << endl;
            cout << "L3CacheHitRatio\tfloat" << endl;
            cout << "L2CacheMisses\tinteger" << endl;
            cout << "L3CacheMisses\tinteger" << endl;
        }

        // provide metadata

        for (int i = 0; i < counters.getNumCores(); ++i) {
            for (int a = 0; a < counters.getNumSockets(); ++a)
                if (a == counters.getSocketId(i)) {
                    stringstream c;
                    c << "Socket" << a << "/CPU" << i << "/Frequency?";
                    if (s == c.str()) {
                        cout << "FREQ. CPU" << i << "\t\t\tMHz" << endl;
                    }
                }
        }
        for (int i = 0; i < counters.getNumCores(); ++i) {
            for (int a = 0; a < counters.getNumSockets(); ++a)
                if (a == counters.getSocketId(i)) {
                    stringstream c;
                    c << "Socket" << a << "/CPU" << i << "/IPC?";
                    if (s == c.str()) {
                        cout << "IPC CPU" << i << "\t0\t\t" << endl;
                        //cout << "CPU" << i << "\tInstructions per Cycle\t0\t1\t " << endl;
                    }
                }
        }
        for (int i = 0; i < counters.getNumCores(); ++i) {
            for (int a = 0; a < counters.getNumSockets(); ++a)
                if (a == counters.getSocketId(i)) {
                    stringstream c;
                    c << "Socket" << a << "/CPU" << i << "/L2CacheHitRatio?";
                    if (s == c.str()) {
                        cout << "L2 Cache Hit Ratio CPU" << i << "\t0\t\t" << endl;
                        //   cout << "CPU" << i << "\tL2 Cache Hit Ratio\t0\t1\t " << endl;
                    }
                }
        }
        for (int i = 0; i < counters.getNumCores(); ++i) {
            for (int a = 0; a < counters.getNumSockets(); ++a)
                if (a == counters.getSocketId(i)) {
                    stringstream c;
                    c << "Socket" << a << "/CPU" << i << "/L3CacheHitRatio?";
                    if (s == c.str()) {
                        cout << "L3 Cache Hit Ratio CPU" << i << "\t0\t\t " << endl;
                    }
                }
        }
        for (int i = 0; i < counters.getNumCores(); ++i) {
            for (int a = 0; a < counters.getNumSockets(); ++a)
                if (a == counters.getSocketId(i)) {
                    stringstream c;
                    c << "Socket" << a << "/CPU" << i << "/L2CacheMisses?";
                    if (s == c.str()) {
                        cout << "L2 Cache Misses CPU" << i << "\t0\t\t " << endl;
                        //cout << "CPU" << i << "\tL2 Cache Misses\t0\t1\t " << endl;
                    }
                }
        }
        for (int i = 0; i < counters.getNumCores(); ++i) {
            for (int a = 0; a < counters.getNumSockets(); ++a)
                if (a == counters.getSocketId(i)) {
                    stringstream c;
                    c << "Socket" << a << "/CPU" << i << "/L3CacheMisses?";
                    if (s == c.str()) {
                        cout << "L3 Cache Misses CPU" << i << "\t0\t\t " << endl;
                        //cout << "CPU" << i << "\tL3 Cache Misses\t0\t1\t " << endl;
                    }
                }
        }
        for (int i = 0; i < counters.getNumSockets(); ++i) {
            stringstream c;
            c << "Socket" << i << "/BytesReadFromMC?";
            if (s == c.str()) {
                cout << "read from MC Socket" << i << "\t0\t\tGB" << endl;
                //cout << "CPU" << i << "\tBytes read from memory channel\t0\t1\t GB" << endl;
            }
        }
        for (int i = 0; i < counters.getNumSockets(); ++i) {
            stringstream c;
            c << "Socket" << i << "/BytesWrittenToMC?";
            if (s == c.str()) {
                cout << "written to MC Socket" << i << "\t0\t\tGB" << endl;
                //cout << "CPU" << i << "\tBytes written to memory channel\t0\t1\t GB" << endl;
            }
        }

        for (int l = 0; l < counters.getQPILinksPerSocket(); ++l) {
            for (int i = 0; i < counters.getNumSockets(); ++i) {
                stringstream c;
                c << "Socket" << i << "/BytesIncomingToQPI" << l << "?";
                if (s == c.str()) {
                    //cout << "Socket" << i << "\tBytes incoming to QPI link\t" << l<< "\t\t GB" << endl;
                    cout << "incoming to Socket" << i << " QPI Link" << l << "\t0\t\tGB" << endl;
                }
            }
        }

        {
            stringstream c;
            c << "QPI_Traffic?";
            if (s == c.str()) {
                cout << "Traffic on all QPIs\t0\t\tGB" << endl;
            }
        }

        for (int i = 0; i < counters.getNumSockets(); ++i) {
            stringstream c;
            c << "Socket" << i << "/Frequency?";
            if (s == c.str()) {
                cout << "Socket" << i << " Frequency\t0\t\tMHz" << endl;
            }
        }

        for (int i = 0; i < counters.getNumSockets(); ++i) {
            stringstream c;
            c << "Socket" << i << "/IPC?";
            if (s == c.str()) {
                cout << "Socket" << i << " IPC\t0\t\t" << endl;
            }
        }

        for (int i = 0; i < counters.getNumSockets(); ++i) {
            stringstream c;
            c << "Socket" << i << "/L2CacheHitRatio?";
            if (s == c.str()) {
                cout << "Socket" << i << " L2 Cache Hit Ratio\t0\t\t" << endl;
            }
        }

        for (int i = 0; i < counters.getNumSockets(); ++i) {
            stringstream c;
            c << "Socket" << i << "/L3CacheHitRatio?";
            if (s == c.str()) {
                cout << "Socket" << i << " L3 Cache Hit Ratio\t0\t\t" << endl;
            }
        }

        for (int i = 0; i < counters.getNumSockets(); ++i) {
            stringstream c;
            c << "Socket" << i << "/L2CacheMisses?";
            if (s == c.str()) {
                cout << "Socket" << i << " L2 Cache Misses\t0\t\t" << endl;
            }
        }

        for (int i = 0; i < counters.getNumSockets(); ++i) {
            stringstream c;
            c << "Socket" << i << "/L3CacheMisses?";
            if (s == c.str()) {
                cout << "Socket" << i << " L3 Cache Misses\t0\t\t" << endl;
            }
        }

        {
            stringstream c;
            c << "Frequency?";
            if (s == c.str()) {
                cout << "Frequency system wide\t0\t\tMhz" << endl;
            }
        }

        {
            stringstream c;
            c << "IPC?";
            if (s == c.str()) {
                cout << "IPC system wide\t0\t\t" << endl;
            }
        }

        {
            stringstream c;
            c << "L2CacheHitRatio?";
            if (s == c.str()) {
                cout << "System wide L2 Cache Hit Ratio\t0\t\t" << endl;
            }
        }

        {
            stringstream c;
            c << "L3CacheHitRatio?";
            if (s == c.str()) {
                cout << "System wide L3 Cache Hit Ratio\t0\t\t" << endl;
            }
        }

        {
            stringstream c;
            c << "L2CacheMisses?";
            if (s == c.str()) {
                cout << "System wide L2 Cache Misses\t0\t\t" << endl;
            }
        }

        {
            stringstream c;
            c << "L3CacheMisses?";
            if (s == c.str()) {
                cout << "System wide L3 Cache Misses\t0\t\t" << endl;
            }
        }


        // sensors

        for (int i = 0; i < counters.getNumCores(); ++i) {
            for (int a = 0; a < counters.getNumSockets(); ++a)
                if (a == counters.getSocketId(i)) {
                    stringstream c;
                    c << "Socket" << a << "/CPU" << i << "/Frequency";
                    //c << "CPU" << i << "/frequency";
                    if (s == c.str()) {
                        cout << counters.get<double, ::getAverageFrequency>(i) / 1000000 << endl;
                    }
                }
        }

        for (int i = 0; i < counters.getNumCores(); ++i) {
            for (int a = 0; a < counters.getNumSockets(); ++a)
                if (a == counters.getSocketId(i)) {
                    stringstream c;
                    c << "Socket" << a << "/CPU" << i << "/IPC";
                    if (s == c.str()) {
                        cout << counters.get<double, ::getIPC>(i) << endl;
                    }
                }
        }
        for (int i = 0; i < counters.getNumCores(); ++i) {
            for (int a = 0; a < counters.getNumSockets(); ++a)
                if (a == counters.getSocketId(i)) {
                    stringstream c;
                    c << "Socket" << a << "/CPU" << i << "/L2CacheHitRatio";
                    if (s == c.str()) {
                        cout << counters.get<double, ::getL2CacheHitRatio>(i) << endl;
                    }
                }
        }
        for (int i = 0; i < counters.getNumCores(); ++i) {
            for (int a = 0; a < counters.getNumSockets(); ++a)
                if (a == counters.getSocketId(i)) {
                    stringstream c;
                    c << "Socket" << a << "/CPU" << i << "/L3CacheHitRatio";
                    if (s == c.str()) {
                        cout << counters.get<double, ::getL3CacheHitRatio>(i) << endl;
                    }
                }
        }
        for (int i = 0; i < counters.getNumCores(); ++i) {
            for (int a = 0; a < counters.getNumSockets(); ++a)
                if (a == counters.getSocketId(i)) {
                    stringstream c;
                    c << "Socket" << a << "/CPU" << i << "/L2CacheMisses";
                    if (s == c.str()) {
                        cout << counters.get<uint64, ::getL2CacheMisses>(i) / 1000000 << endl;
                    }
                }
        }
        for (int i = 0; i < counters.getNumCores(); ++i) {
            for (int a = 0; a < counters.getNumSockets(); ++a)
                if (a == counters.getSocketId(i)) {
                    stringstream c;
                    c << "Socket" << a << "/CPU" << i << "/L3CacheMisses";
                    if (s == c.str()) {
                        cout << counters.get<uint64, ::getL3CacheMisses>(i) / 1000000 << endl;
                    }
                }
        }

        for (int i = 0; i < counters.getNumSockets(); ++i) {
            stringstream c;
            c << "Socket" << i << "/BytesReadFromMC";
            if (s == c.str()) {
                cout << double(counters.getSocket<uint64, ::getBytesReadFromMC>(i)) / 1024 / 1024 / 1024 << endl;
            }
        }
        for (int i = 0; i < counters.getNumSockets(); ++i) {
            stringstream c;
            c << "Socket" << i << "/BytesWrittenToMC";
            if (s == c.str()) {
                cout << double(counters.getSocket<uint64, ::getBytesWrittenToMC>(i)) / 1024 / 1024 / 1024 << endl;
            }
        }
        for (int l = 0; l < counters.getQPILinksPerSocket(); ++l) {
            for (int i = 0; i < counters.getNumSockets(); ++i) {
                stringstream c;
                c << "Socket" << i << "/BytesIncomingToQPI" << l;
                if (s == c.str()) {
                    cout << double(counters.getSocket<uint64, ::getIncomingQPILinkBytes>(i, l)) / 1024 / 1024 / 1024 << endl;
                }
            }
        }
        stringstream c;
        c << "QPI_Traffic";
        if (s == c.str()) {
            cout << double(counters.getSystem<uint64, ::getAllIncomingQPILinkBytes>()) / 1024 / 1024 / 1024 << endl;
        }


        for (int i = 0; i < counters.getNumSockets(); ++i) {
            stringstream c;
            c << "Socket" << i << "/Frequency";
            if (s == c.str()) {
                cout << counters.getSocket<double, ::getAverageFrequency>(i) / 1000000 << endl;
            }
        }

        for (int i = 0; i < counters.getNumSockets(); ++i) {
            stringstream c;
            c << "Socket" << i << "/IPC";
            if (s == c.str()) {
                cout << counters.getSocket<double, ::getIPC>(i) << endl;
            }
        }

        for (int i = 0; i < counters.getNumSockets(); ++i) {
            stringstream c;
            c << "Socket" << i << "/L2CacheHitRatio";
            if (s == c.str()) {
                cout << counters.getSocket<double, ::getL2CacheHitRatio>(i) << endl;
            }
        }

        for (int i = 0; i < counters.getNumSockets(); ++i) {
            stringstream c;
            c << "Socket" << i << "/L3CacheHitRatio";
            if (s == c.str()) {
                cout << counters.getSocket<double, ::getL3CacheHitRatio>(i) << endl;
            }
        }


        for (int i = 0; i < counters.getNumSockets(); ++i) {
            stringstream c;
            c << "Socket" << i << "/L2CacheMisses";
            if (s == c.str()) {
                cout << counters.getSocket<uint64, ::getL2CacheMisses>(i) << endl;
            }
        }

        for (int i = 0; i < counters.getNumSockets(); ++i) {
            stringstream c;
            c << "Socket" << i << "/L3CacheMisses";
            if (s == c.str()) {
                cout << counters.getSocket<uint64, ::getL3CacheMisses>(i) << endl;
            }
        }

        {
            stringstream c;
            c << "Frequency";
            if (s == c.str()) {
                cout << double(counters.getSystem<double, ::getAverageFrequency>()) / 1000000 << endl;
            }
        }

        {
            stringstream c;
            c << "IPC";
            if (s == c.str()) {
                cout << double(counters.getSystem<double, ::getIPC>()) << endl;
            }
        }

        {
            stringstream c;
            c << "L2CacheHitRatio";
            if (s == c.str()) {
                cout << double(counters.getSystem<double, ::getL2CacheHitRatio>()) << endl;
            }
        }

        {
            stringstream c;
            c << "L3CacheHitRatio";
            if (s == c.str()) {
                cout << double(counters.getSystem<double, ::getL3CacheHitRatio>()) << endl;
            }
        }

        {
            stringstream c;
            c << "L2CacheMisses";
            if (s == c.str()) {
                cout << double(counters.getSystem<uint64, ::getL2CacheMisses>()) << endl;
            }
        }

        {
            stringstream c;
            c << "L3CacheMisses";
            if (s == c.str()) {
                cout << double(counters.getSystem<uint64, ::getL3CacheMisses>()) << endl;
            }
        }

        // exit
        if (s == "quit" || s == "exit") {
            break;
        }


        cout << "ksysguardd> ";
    }

    return 0;
}
