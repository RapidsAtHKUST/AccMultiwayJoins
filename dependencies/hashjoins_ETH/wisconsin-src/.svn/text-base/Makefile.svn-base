
.PHONY: all clean doc distclean 

include system.inc

CXX=g++
CPPFLAGS=-Idist/include/ -Ilibconfig-1.2/ -I../src/
#CPPFLAGS+=-DPREFETCH
#CPPFLAGS+=-DOUTPUT_AGGREGATE
#CPPFLAGS+=-DOUTPUT_WRITE_NT
#CPPFLAGS+=-DOUTPUT_WRITE_NORMAL
CPPFLAGS+=-DOUTPUT_ASSEMBLE
#CPPFLAGS+=-DDEBUG #-DDEBUG2
CXXFLAGS=$(SYSFLAGS)
#CXXFLAGS+=-g -O0 #-Wall
CXXFLAGS+=-O3 -DKEY_8B -DCUSTOM_CPU_MAPPING=\"cpu-mapping.txt\" 
LDFLAGS=-Ldist/lib/ -Llibconfig-1.2/.libs/
LDLIBS=-lconfig++ -lpthread -lbz2

USE_PERF_COUNTERS=0

ifeq ($(USE_PERF_COUNTERS),1)
CXXFLAGS += -DPERF_COUNTERS 
LDFLAGS  += -L../lib/intel-pcm-1.7/lib/
LDLIBS   += -lperf
CPPFLAGS += -I../lib/intel-pcm-1.7/
endif


ifeq ($(HOSTTYPE),sparc)
LDLIBS+=-lcpc
CXXFLAGS+=-mcpu=ultrasparc
endif

all: dist multijoin multijoin-serial

FILES = schema.o hash.o parser.o table.o joinerfactory.o page.o \
	algo/nl.o algo/base.cpp ProcessorMap.o Barrier.o loader.o \
	partitionerfactory.o partitioner.o algo/hashbase.o \
	algo/storage.o affinitizer.o algo/hashtable.o \
	comparator.o algo/flatmem.o genzipf.o generator.o cpu_mapping.o perf_counters.o


#################### External source files from ETH implementation ############
cpu_mapping.o: ../src/cpu_mapping.h ../src/cpu_mapping.c
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -c ../src/cpu_mapping.c

genzipf.o:  ../src/genzipf.h ../src/genzipf.c
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -c ../src/genzipf.c

generator.o: ../src/generator.h ../src/generator.c
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -c ../src/generator.c

perf_counters.o: ../src/perf_counters.h ../src/perf_counters.c
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -c ../src/perf_counters.c

# generates assembly for storage.cpp
storage.s: algo/storage.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -fverbose-asm -S -c algo/storage.cpp


#################### end of external source files #############################

clean:
	rm -f *.o
	rm -f algo/*.o
	rm -f multijoin multijoin-serial

distclean: clean
	rm -rf dist

doc: Doxyfile
	doxygen


multijoin-serial: $(FILES) main-serial.o

multijoin: $(FILES) main.o

dist:
	./pre-init.sh
