/*
    Copyright 2011, Spyros Blanas.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <fstream>
#include <exception>
#include <sstream>
#include <vector>
#include <cassert>
#include <pthread.h>

#include <libconfig.h++>

#include "rdtsc.h"
#include "exceptions.h"
#include "schema.h"
#include "table.h"
#include "loader.h"
#include "hash.h"
#include "algo/algo.h"
#include "joinerfactory.h"
#include "partitionerfactory.h"

#include "ProcessorMap.h"
#include "Barrier.h"
#include "lock.h"
#include "affinitizer.h"

#include "../src/perf_counters.h"

using namespace libconfig;
using namespace std;

struct ThreadArg {
	int threadid;
};

/* Those needed by stupid, ugly pthreads. */
unsigned int joinattr1, joinattr2;
BaseAlgo* joiner;
vector<PageCursor*> joinresult;
HashFunction* hashfn;
unsigned long long timer, timer2, timer3;
PThreadLockCVBarrier* barrier;
Lock joinresultlock;
Partitioner* partitioner, * partitioner2;
Affinitizer aff;

#ifdef PERF_COUNTERS
/* Intel PCM Config vars */
char * perfconf;
char * perfout;
#endif

vector<unsigned int> createIntVector(const Setting& line) {
	vector<unsigned int> ret;
	for (int i=0; i<line.getLength(); ++i) {
		unsigned int val = line[i];
		ret.push_back(val-1);
	}
	return ret;
}

inline void initchkpt(void) {
	startTimer(&timer);
	startTimer(&timer2);	
	startTimer(&timer3);	
}

inline void partchkpt(void)
{
	stopTimer(&timer3);
}

inline void buildchkpt(void) {
	stopTimer(&timer2);
}

inline void probechkpt(void) {
	stopTimer(&timer);	// stop clock
}

void* compute(void* value) {
	int threadid = reinterpret_cast<ThreadArg*>(value)->threadid;
	aff.affinitize(threadid);

	barrier->Arrive();	// sync

	if (threadid == 0){
	  initchkpt();
#ifdef PERF_COUNTERS
      //default config and output to stdout
	  PCM_initPerformanceMonitor(perfconf, perfout);
	  PCM_start();
#endif
	}

	SplitResult tin = partitioner->split(threadid);
	SplitResult tout = partitioner2->split(threadid);

	barrier->Arrive();	// sync

	if (threadid == 0){
	  partchkpt();

#ifdef PERF_COUNTERS
        printf("========== Partitioning phase profiling results ==========\n");
        PCM_stop();
        PCM_printResults();
        PCM_start();
#endif


	}
	joiner->build(tin, threadid);					// build

	barrier->Arrive();	// sync

	if (threadid == 0) {
	  buildchkpt();

#ifdef PERF_COUNTERS
	  printf("========== Build phase profiling results ==========\n");
	  PCM_stop();
	  PCM_printResults();
	  PCM_start();
#endif

	}

	PageCursor* t = joiner->probe(tout, threadid);	// probe

	barrier->Arrive();	// sync

	if (threadid == 0) {
        probechkpt();

#ifdef PERF_COUNTERS
        printf("========== Probe phase profiling results ==========\n");
        PCM_stop();
        PCM_printResults();
        PCM_cleanup();
        printf("===================================================\n");
#endif

    }

	joinresultlock.lock();
	joinresult.push_back(t);	// remember result
	joinresultlock.unlock();

	return 0;
}

int main(int argc, char** argv) {
	ThreadArg* ta;
	int nothreads;
	
	// ensuring that one file has been specified
	if (argc <= 1) {
		cout << "error: configuration file is not specified" << endl;
		return 2;
	}

#ifdef PERF_COUNTERS
    // if Intel PCM enabled then there can be 2 more parameters on cmdline
    if (argc == 3) 
        perfconf = argv[2];
    else
        perfconf = NULL;

    if (argc == 4)
        perfout = argv[3];
    else
        perfout = NULL;
#endif

	// system sanity checks
	assert(sizeof(char)==1);
	assert(sizeof(void*)==sizeof(long));

	try {
		Table* tin, * tout;
		Schema sin, sout;
		string datapath, infilename, outfilename, outputfile;
		unsigned int bucksize;
		vector<unsigned int> select1, select2;

		// parsing cfg file
		Config cfg;
		cout << "Loading configuration file... " << flush;
		cfg.readFile(argv[1]);
		datapath = (const char*) cfg.lookup("path");
		bucksize = cfg.lookup("bucksize");

		sin = Schema::create(cfg.lookup("build.schema"));
		WriteTable wr1;
		wr1.init(&sin, bucksize);
		tin = &wr1;
		infilename = (const char*) cfg.lookup("build.file");
		joinattr1 = cfg.lookup("build.jattr");
		joinattr1--;
		select1 = createIntVector(cfg.lookup("build.select"));

		sout = Schema::create(cfg.lookup("probe.schema"));
		WriteTable wr2;
		wr2.init(&sout, bucksize);
		tout = &wr2;
		outfilename = (const char*) cfg.lookup("probe.file");
		joinattr2 = cfg.lookup("probe.jattr");
		joinattr2--;
		select2 = createIntVector(cfg.lookup("probe.select"));

		outputfile = (const char*) cfg.lookup("output");

        	/* commenting out, we allow int keys as well */
		// assert(tin->schema()->getColumnType(joinattr1)==CT_LONG);
		// assert(tout->schema()->getColumnType(joinattr2)==CT_LONG);

		ProcessorMap pc;
		if (!cfg.lookupValue("threads", nothreads)) {
			nothreads = pc.NumberOfProcessors();
			cfg.getRoot().add("threads", Setting::TypeInt) = nothreads;
		}
		barrier = new PThreadLockCVBarrier(nothreads);
		aff.init(nothreads, pc.NumberOfProcessors());

		string flatmemstr = "no";
		cfg.getRoot()["algorithm"].lookupValue("flatmem", flatmemstr);
		if (flatmemstr == "yes") {
			cfg.getRoot()["hash"].add("pagesize", Setting::TypeInt) = 0;
			int jattr1 = cfg.getRoot()["build"]["jattr"];
			cfg.getRoot()["hash"].add("attribute", Setting::TypeInt) = jattr1;
		}

		joiner = JoinerFactory::createJoiner(cfg);

		partitioner = PartitionerFactory::createPartitioner(cfg, 
				cfg.lookup("partitioner.build"));
		partitioner2 = PartitionerFactory::createPartitioner(cfg, 
				cfg.lookup("partitioner.probe"));

		cout << "ok" << endl;

        // the old just parsing version
		// wr1.load(datapath+infilename, "|");

		// wr2.load(datapath+outfilename, "|");
        /* Now it's possible to generate relations on the fly */
        bool generate = false;
        if(cfg.lookupValue("build.generate", generate) && generate){
            cout << "Generating data in memory for relation R ... " << flush;
            wr1.generate(cfg.lookup("build.relation-size"),
                         cfg.lookup("build.alphabet-size"),
                         cfg.lookup("build.zipf-param"),
                         cfg.lookup("build.seed"));
        }
        else {
            //the old method: loads from file.
            // load files in memory
            cout << "Loading data in memory... " << flush;
            wr1.load(datapath+infilename, "|");
        }
		cout << "ok" << endl;
        
        generate = false;
        if(cfg.lookupValue("probe.generate", generate) && generate){
            cout << "Generating data in memory for relation S ... " << flush;
            wr2.generate(cfg.lookup("probe.relation-size"),
                         cfg.lookup("probe.alphabet-size"),
                         cfg.lookup("probe.zipf-param"),
                         cfg.lookup("probe.seed"));
        }
        else {
            //the old method: loads from file.
            // load files in memory
            cout << "Loading data in memory... " << flush;
            wr2.load(datapath+outfilename, "|");
        }
        cout << "ok" << endl;


		/* tin, tout are loaded with data now */
		cout << "Running join algorithm... " << flush;
		partitioner->init(tin);
		partitioner2->init(tout);
		joiner->init(tin->schema(), select1, joinattr1,
				tout->schema(), select2, joinattr2);
		if (flatmemstr=="yes") {
			dynamic_cast<FlatMemoryJoiner*>(joiner)->custominit(tin, tout);
		}
		
		joinresult.reserve(nothreads+1);
		pthread_t* threadpool = new pthread_t[nothreads];
		ta = new ThreadArg[nothreads];
		pthread_attr_t attr;

		pthread_attr_init (&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
		pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);

		for (int i=0; i<nothreads; ++i) {
			ta[i].threadid = i;
			assert(!pthread_create(&threadpool[i], &attr, compute, &ta[i]));
		}

		pthread_attr_destroy(&attr);
		for (int i=0; i<nothreads; ++i)
			assert(!pthread_join(threadpool[i], NULL));

		delete barrier;
		delete[] threadpool;

		joiner->destroy();
		delete joiner;

		cout << "ok" << endl;

		/* joinresult of type vector<PageCursor*> has the generated output */
		// output for validation
#ifdef DEBUG
		cout << "Outputting code for validation... " << flush;
		ofstream foutput((datapath+outputfile).c_str());
		for (vector<PageCursor*>::iterator i1=joinresult.begin(); 
				i1!=joinresult.end(); ++i1) {
			Page* b;
			if (!*i1)
				continue;
			void* tuple;
			while (b = (*i1)->readNext()) {
				int i = 0;
				while(tuple = b->getTupleOffset(i++)) {
					foutput << (*i1)->schema()->prettyprint(tuple, '|') << '\n';
				}
			}
		}
		foutput << flush;
		foutput.close();
		cout << "ok" << endl;
		
#endif

		// be nice, free some mem
		wr1.close();
		wr2.close();
		partitioner->destroy();
		partitioner2->destroy();
		for (vector<PageCursor*>::iterator i1=joinresult.begin(); 
				i1!=joinresult.end(); ++i1) {
			(*i1)->close();
			delete *i1;
		}
	} catch (IllegalSchemaDeclarationException& e) {
		cout << "failed" << endl;
		cout << "EXCEPTION: ";
		cout << "Schema cannot be created from configuration file." << endl;
		throw;
		return 2;
	} catch (IllegalConversionException& e) {
		cout << "failed" << endl;
		cout << "EXCEPTION: ";
		cout << "An illegal conversion was attempted." << endl;
		throw;
		return 2;
	} catch (UnknownAlgorithmException& e) {
		cout << "failed" << endl;
		cout << "EXCEPTION: ";
		cout << "Unknown join algorithm." << endl;
		throw;
		return 2;
	} catch (ParseException& e) {
		cout << "failed" << endl;
		cout << "EXCEPTION: Parse exception, at line " 
			<< e.getLine() << ": " << e.getError() << endl;
		throw;
		return 2;
	} catch (ConfigException& e) {
		cout << "failed" << endl;
		cout << "EXCEPTION: ";
		cout << "Configuration file is in wrong format." << endl;
		throw;
		return 2;
	} catch (PageFullException& e) {
		cout << "failed" << endl;
		cout << "EXCEPTION: ";
		cout << "Page full when inserting key " << e.value << endl;
		throw;
		return 2;
	} catch (exception& e) {
		cout << "failed" << endl;
		cout << "EXCEPTION: Standard exception: " << e.what() << endl;
		throw;
		return 2;
	}

	// bye
	cout << endl << "RUNTIME TOTAL, BUILD+PART, PART (cycles): " << endl;
	cerr << timer << "\t" << timer2 << "\t" << timer3;
    cout << endl;

	delete[] ta;
	return 0;
}
