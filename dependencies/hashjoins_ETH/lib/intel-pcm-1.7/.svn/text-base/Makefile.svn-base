#
# Copyright (c) 2009-2010 Intel Corporation
# written by Roman Dementiev
#

# Changelog:
#
#  - Added targets for building libraries and modified default target. (cagri)
#

# Now only builds the shared library by default
all: lib/libperf.so
#all: pcm.x cpusensor.x realtime.x

CC=g++
OPT= -g -O3 -pthread -fPIC

msr.o: msr.h msr.cpp 
	$(CC) $(OPT) -c msr.cpp

pci.o: pci.h pci.cpp
	$(CC) $(OPT) -c pci.cpp

cpucounters.o: msr.h msr.cpp cpucounters.h cpucounters.cpp types.h
	$(CC) $(OPT) -c cpucounters.cpp

pcmdaemon.o: msr.h msr.cpp cpucounters.h pcmdaemon.cpp types.h
	$(CC) $(OPT) -c pcmdaemon.cpp

msrtest.x: msrtest.cpp msr.o cpucounters.o cpucounters.h  msr.h  types.h
	$(CC) $(OPT) msrtest.cpp -o msrtest.x msr.o cpucounters.o

cpucounterstest.o: cpucounterstest.cpp cpucounters.h  msr.h  types.h
	$(CC) $(OPT) -c cpucounterstest.cpp

realtime.o: realtime.cpp cpucounters.h  msr.h  types.h
	$(CC) $(OPT) -c realtime.cpp

htscheduling.o: htscheduling.cpp cpucounters.h  msr.h  types.h
	$(CC) $(OPT) -c  htscheduling.cpp

pcm.x: msr.o cpucounters.o cpucounterstest.o pci.o
	$(CC) $(OPT) -lrt msr.o pci.o cpucounters.o cpucounterstest.o -o pcm.x

pcmdaemon.x: msr.o cpucounters.o pcmdaemon.o pci.o
	$(CC) $(OPT) -lrt msr.o cpucounters.o pcmdaemon.o pci.o -o pcmdaemon.x

realtime.x: msr.o cpucounters.o realtime.o
	$(CC) $(OPT) -lpthread -lrt pci.o msr.o cpucounters.o realtime.o -o realtime.x

htscheduling.x: msr.o cpucounters.o htscheduling.o
	$(CC) $(OPT) -lpthread -lrt pci.o msr.o cpucounters.o htscheduling.o -o htscheduling.x


cpusensor.o: cpusensor.cpp cpucounters.h cpuasynchcounter.h msr.h  types.h
	$(CC) $(OPT) -c cpusensor.cpp

cpusensor.x: msr.o cpucounters.o cpusensor.o
	$(CC) $(OPT) -lpthread -lrt msr.o pci.o cpucounters.o cpusensor.o -o cpusensor.x

nice:
	uncrustify --replace -c ~/uncrustify.cfg *.cpp *.h WinMSRDriver/Win7/*.h WinMSRDriver/Win7/*.c WinMSRDriver/WinXP/*.h WinMSRDriver/WinXP/*.c  PCM_Win/*.h PCM_Win/*.cpp  


# Added to create libraries
libperf: cpucounters.h msr.h types.h msr.o cpucounters.o
	ar rcs libperf.a  msr.o cpucounters.o

lib/libperf.so: msr.o cpucounters.o
	mkdir -p lib
	$(CC) $(OPT) -shared  -Wl,-soname,libperf.so -o lib/libperf.so msr.o cpucounters.o

mytest: lib/libperf.so mytest.cpp
	$(CC) $(OPT) -o mytest mytest.cpp -I. -L. -lperf

distdir:

distclean: clean

clean:
	rm -rf *.x *.o *~ lib/ *.a mytest
