#include <iostream>

#include "cpucounters.h"

using namespace std;

int fib(int n) {
  if(n == 1 || n == 2)
    return 1;
  else 
    return (fib(n-1) + fib(n-2));
}

int main(void) {
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
    cout << "starting 1..." << endl;
    SystemCounterState before_sstate = getSystemCounterState(); 
    cout << "starting 2..." << endl;
    fib(20);
    cout << "ending 2..." << endl;
    SystemCounterState after_sstate = getSystemCounterState(); 
    cout << "Instructions per clock:" << getIPC(before_sstate,after_sstate) << endl; 
    PCM::getInstance()->cleanup();

    return 0;
}
