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

#include "partitioner.h"
#include "table.h"

#ifdef DEBUG
#include <cassert>
#include <iostream>
using std::cout;
using std::endl;
#endif

// #define VERBOSE
// #define VERBOSE2

//#define DEBUGALLOC

#ifdef DEBUGALLOC
#include <cstdio>
#include <cstdlib>
void* operator new (size_t size)
{
	void* ret = malloc(size);
	assert(ret != NULL);
	fprintf(stderr, "+ %016p\n", ret);
	return ret;
}

void operator delete (void* p)
{
	free(p);
	fprintf(stderr, "- %016p\n", p);
}
#endif

Partitioner::Partitioner(const libconfig::Config& cfg, 
		const libconfig::Setting& node, 
		const libconfig::Setting& hashnode)
	: barrier(0)
{ 
	nthreads = cfg.lookup("threads");
	pagesize = node["pagesize"];
	attribute = node["attribute"];
	attribute--;
	hashfn = HashFactory::createHashFunction(hashnode);
}

Partitioner::~Partitioner()
{
	delete hashfn;
}

/**
 * Returns the passed parameter as a single partition.
 * That is, no partitioning is done.
 *
 * Contract: 
 * 1. All threads calling split() will return the same value. 
 * 2. No thread will return before the last thread has called split().
 *
 */
SplitResult Partitioner::split(int threadid) 
{
	if (threadid == 0) {
		result.push_back(input);
	}

	barrier->Arrive();

	return &result;
}

/**
 * Initialization method for partitioner. This is called by a single thread.
 */
void Partitioner::init(PageCursor* t) 
{
	input = t;
	barrier = new PThreadLockCVBarrier(nthreads);
}

/**
 * Uninit method for partitioner. This is called by a single thread.
 */ 
void Partitioner::destroy()
{
	delete barrier;
}



ParallelPartitioner::ParallelPartitioner(const libconfig::Config& cfg, 
		const libconfig::Setting& node,
		const libconfig::Setting& hashnode)
	: Partitioner(cfg, node, hashnode)
{
}

/**
 * Creates one output buffer and lock per partition.
 */
void ParallelPartitioner::init(PageCursor* t) 
{
	Partitioner::init(t);

	for (int i=0; i<hashfn->buckets(); ++i) {
		AtomicWriteTable* tmp = new AtomicWriteTable();
		tmp->init(t->schema(), pagesize);
		result.push_back(tmp);
	}
}

/**
 * Parallel buffers split: all threads share the buffers and write to them
 * atomically.
 */
SplitResult ParallelPartitioner::split(int threadid) 
{
	Page* page;
	void* tup;
	int i;
	unsigned int hashval;

	Schema* schema = input->schema();

	while (page = input->atomicReadNext()) {
		i = 0;
		while (tup = page->getTupleOffset(i++)) {
			hashval = hashfn->hash(schema->asLong(tup, attribute));
#ifdef DEBUG
			assert(hashval < result.size() && hashval >= 0);
			assert(schema == result[hashval]->schema());
#endif
			((AtomicWriteTable*)result[hashval])->append(tup);
		}
	}
	
	barrier->Arrive();

	return &result;
}

/**
 * Reverses \a init().
 * Frees each output buffer.
 * Frees the lock array.
 */
void ParallelPartitioner::destroy()
{
	for (int i=0; i<hashfn->buckets(); ++i) {
		result[i]->close();
		delete result[i];
	}
	result.clear();

	Partitioner::destroy();
}



IndependentPartitioner::IndependentPartitioner(const libconfig::Config& cfg, 
		const libconfig::Setting& node,
		const libconfig::Setting& hashnode)
	: Partitioner(cfg, node, hashnode)
{
}

void IndependentPartitioner::init(PageCursor* t) 
{
	Partitioner::init(t);

	tempstore = new vector<WriteTable*>[nthreads];

	for (int i=0; i<hashfn->buckets(); ++i) {
		WriteTable* tmp = new WriteTable();
		tmp->init(t->schema(), pagesize);
		result.push_back(tmp);
		for (int j=0; j<nthreads; ++j) {
			WriteTable* tmp2 = new WriteTable();
			tmp2->init(t->schema(), pagesize/nthreads);
			tempstore[j].push_back(tmp2);
		}
	}
}

SplitResult IndependentPartitioner::split(int threadid) 
{
	Page* page;
	void* tup;
	int i;
	unsigned int hashval;

	Schema* schema = input->schema();

	while (page = input->atomicReadNext()) {
		i = 0;
		while (tup = page->getTupleOffset(i++)) {
			hashval = hashfn->hash(schema->asLong(tup, attribute));
#ifdef DEBUG
			assert(hashval < tempstore[threadid].size() && hashval >= 0);
			assert(schema == tempstore[threadid][hashval]->schema());
#endif
			tempstore[threadid][hashval]->append(tup);
		}
	}
	
	barrier->Arrive();

	for (int i=threadid; i<hashfn->buckets(); i+=nthreads) {
		for (int j=0; j<nthreads; ++j) {
			((WriteTable*)result[i])->concatenate(*tempstore[j][i]);
		}
	}

	barrier->Arrive();

	return &result;
}
/**
 * Reverses \a init().
 */
void IndependentPartitioner::destroy()
{
	for (int i=0; i<hashfn->buckets(); ++i) {
		for (int j=0; j<nthreads; ++j) {
			// We don't close tempstore[j][i] because this will result to a
			// double free; concatenate() causes the pages to be pointed to by
			// both tables.
			//
			delete tempstore[j][i];
		}
	}

	for (vector<PageCursor*>::iterator i1 = result.begin(); 
			i1 != result.end(); ++i1) 
	{
		(*i1)->close();
		delete *i1;
	}

	Partitioner::destroy();
}

DerekPartitioner::DerekPartitioner(const libconfig::Config& cfg, 
		const libconfig::Setting& node,
		const libconfig::Setting& hashnode)
	: Partitioner(cfg, node, hashnode)
{
}

SplitResult DerekPartitioner::split(int threadid) 
{
	return Partitioner::split(threadid);
}

RadixPartitioner::RadixPartitioner(const libconfig::Config& cfg, 
		const libconfig::Setting& node, 
		const libconfig::Setting& hashnode)
	: Partitioner(cfg, node, hashnode), maxkey(0)
{
	string hashfnname = hashnode["fn"];
	if (hashfnname != "modulo")
		throw NotYetImplemented();
	totalpasses = static_cast<short>(static_cast<int>(node["passes"]));
}

/**
 * Do radix partitioning.
 * Prerequisites: 
 * 	- Schema must have been initialized.
 * 	- Class variable attribute must point to the partition attribute.
 * 	- \a dest can hold as many tuples as \a source can.
 *
 * @param source Page to read from.
 * @param offset First tuple to partition. Different per thread.
 * @param items Number of tuples to partition.
 * @param dest Page to write to. (All tuples have been pre-allocated.)
 * @param hashfunc Pointer to hash function.
 * @param globalhist Global histogram.
 * @param iteroffset Offset if hashing to bucket 0, which can't be deduced from
 * the histogram.
 * @param localhist Local, thread-specific histogram.
 */
void radixpartition(Page* source, const int offset, const int items, 
		HashFunction* hashfunc, Schema& schema, const int attribute,
		Page* dest,
		const unsigned int* globalhist, const unsigned int iteroffset,
		vector<unsigned int>& localhist)
{
	void* srctup;
	unsigned int h;
	unsigned int globaloff;
	unsigned int localoff;
	void* dsttup;

	for (int i=0; i<items; ++i) {
		srctup = source->getTupleOffset(offset + i);
		h = hashfunc->hash(schema.asLong(srctup, attribute));
		globaloff = (h != 0 ? globalhist[h-1] : iteroffset);
		localoff = localhist[h];
		++localhist[h];

		dsttup = dest->getTupleOffset(globaloff + localoff);
#if VERBOSE2
		printf("Copy from %p h[0x%08x]=%d to %p\n", 
				srctup, 
				schema.asLong(srctup, attribute), 
				h, 
				dsttup);
#endif
#ifdef DEBUG
		assert(dsttup);
#endif
		schema.copyTuple(dsttup, srctup);
	}
}

/** 
 * Fills in \a histogram. 
 */
void createhistogram(Page* source, const int offset, const int items, 
		HashFunction* hashfunc, Schema& schema, const int attribute,
		unsigned int* histogram)
{
	void* srctup;

	for (int i=0; i<items; ++i) {
		srctup = source->getTupleOffset(offset + i);
		unsigned int h = hashfunc->hash(schema.asLong(srctup, attribute));
		++histogram[h];
	}
}

/** 
 * Place \a output cursor i at location[i].
 * @param threadid Unique threadid in 0, 1, ..., \a nthreads - 1.
 * @param nthreads Number of threads.
 */
void placecursors(const int threadid, const int nthreads, 
		const unsigned int tuplesize,
		const vector<unsigned int>& location, Page* page, SplitResult results)
{
	unsigned int pagesize;
	void* data;

	int items = location.size() / nthreads;
	int start = items * threadid;
	if (threadid == nthreads-1) {
		// Compensate for last few tuples, 
		// if \a items has been rounded down.
		items = location.size() - start;
#ifdef DEBUG
		assert(results->size() == start + items);
#endif
	} 
	
	FakeTable* p;
	if (threadid == 0) {
		// Treat first bucket of first thread differently, instead of polluting
		// the for loop below.
		pagesize = location[0] * tuplesize;
		data = page->getTupleOffset(0);
#ifdef DEBUG
		assert(start == 0);
		assert(data!=NULL);
		p = dynamic_cast<FakeTable*>((*results)[0]);
		assert(p != NULL);
#else
		p = (FakeTable*)((*results)[0]);
#endif
		p->place(pagesize, tuplesize, data);
		++start;
		--items;
	}

	for (int i=start; i<start+items; ++i) {
		pagesize = (location[i] - location[i-1]) * tuplesize;
		data = page->getTupleOffset(location[i-1]);
#ifdef DEBUG
		assert(data!=NULL);
		p = dynamic_cast<FakeTable*>((*results)[i]);
		assert(p != NULL);
#else
		p = (FakeTable*)((*results)[i]);
#endif
		p->place(pagesize, tuplesize, data);
	}
}

/** 
 * Make histograms[tid][hash] = sum(i=0...tid){histograms[i][hash]},
 * in parallel. Each thread works on a partition of tid.
 * @param threadid Unique threadid in 0, 1, ..., \a nthreads - 1.
 * @param nthreads Number of threads.
 * @param histsize Size of histogram. May be less than histogram[i].size(), as
 * we are overprovisioning and make histogram[i] big enough to hold any
 * iteration.
 * @param histograms Per-thread histograms.
 * @param globalhist Global histogram.
 */
void combinehistogram(const int threadid, const int nthreads, 
		const unsigned int histsize,
		vector<vector<unsigned int> >& histograms, unsigned int* globalhist)
{
#ifdef DEBUG
	// all histograms are of equal size
	assert(histograms[threadid].size() == histograms[0].size());
#endif

	int items = histsize / nthreads;
	int start = items * threadid;
	if (threadid == nthreads-1) {
		// Compensate for last few tuples, 
		// if \a items has been rounded down.
		items = histsize - start;
	}
	for (int h=start; h<start+items; ++h) {
		for (int tid=2; tid<nthreads; ++tid) {
			// Leave histogram[0] alone, it's always zero.
			// Leave histogram[1] alone, histogram[1] + histogram[0] =
			// = histogram[1] + 0 = histogram[1].
			histograms[tid][h]+=histograms[tid-1][h];
		}
		globalhist[h] += histograms[nthreads-1][h];
	}
}

vector<unsigned int>& RadixPartitioner::realsplit(int threadid) 
{
#ifdef DEBUG
	// all histograms are of equal size
	assert(histograms[threadid].size() == histograms[0].size());
#endif
	
	unsigned int histsize; 
	unsigned int accum = 1;

	// Loop as many times as specified.
	for (int pass=0; pass<totalpasses; ++pass) {

		histsize = vec_hf[pass]->buckets();

		for (int iter=0; iter < accum; ++iter) {
			unsigned int tuplesinblock;
			unsigned int iteroffset;

			if (pass == 0) {
				tuplesinblock = totaltuples;
				iteroffset = 0;
			} else if (iter == 0) {
				tuplesinblock = offsets[pass-1][iter];
				iteroffset = 0;
			} else {
				tuplesinblock = offsets[pass-1][iter] - offsets[pass-1][iter-1];
				iteroffset = offsets[pass-1][iter-1];
			}

			unsigned int items = tuplesinblock / nthreads;
			unsigned int start = items * threadid;
			if (threadid == nthreads-1) {
				// Compensate for last few tuples, 
				// if \a items has been rounded down.
				items = tuplesinblock - start;
			}
			start += iteroffset;

#ifdef DEBUG
			assert(oldbuf->getTupleOffset(start+items-1) != NULL);
			if ( (threadid == nthreads-1) && (iter == accum-1) )
				// If last thread and last iteration, make sure we read 
				// the very last tuple in the page.
				assert(oldbuf->getTupleOffset(start+items) == NULL);
#ifdef VERBOSE
			cout << "Pass " << pass << " Iter " << iter << " Thread " 
				<< threadid << " will partition "
				<< items << " tuples, starting from " << start << endl;
#endif
#endif

			unsigned int* globalhist = &offsets[pass][iter * histsize];

			// 1. Scan once; create per-thread histograms.
			//
			if (threadid == nthreads-1) {
				createhistogram(oldbuf, start, items, 
						vec_hf[pass], schema, attribute, 
						globalhist);
			} else {
				createhistogram(oldbuf, start, items, 
						vec_hf[pass], schema, attribute, 
						&histograms[threadid+1][0]);
			}
			barrier->Arrive();

			// 2. Combine histograms, ideally in parallel.
			//
			// Make histogram[tid][hash] = sum(i=0...tid){histogram[i][hash]},
			// in parallel. Each thread works on a partition of tid.
			combinehistogram(threadid, nthreads, histsize, 
					histograms,	globalhist);
			barrier->Arrive();

			// Then, a single thread computes 
			// globalhist[hash] = sum(i=0...hash){histogram[maxtid][i]}. 
			if (threadid == 0) {
				globalhist[0] += iteroffset;
				for (int i=1; i<histsize; ++i) {
					globalhist[i] += globalhist[i-1];
				}
			}
			barrier->Arrive();

			// 3. Call radixpartition (scans twice) to partition.
			// If last pass, place cursors on output.
			//
			radixpartition(oldbuf, start, items, 
					vec_hf[pass], schema, attribute, 
					newbuf,
					globalhist, iteroffset, histograms[threadid]);

			// 4. Reset histograms.
			//
			for (int i=0; i<histsize; ++i) {
				histograms[threadid][i] = 0;
			}

			// Barrier below could be eliminated by double-buffering.
			// Without double buffering, thread i may overwrite thread's i+1
			// local histogram, while i+1 has not yet exiting radixpartition.
			//
			// Barrier would still be necessary for last iteration, as thread 0
			// might run ahead and change value of newbuf before thread i calls
			// radixpartition with old value of newbuf.
			//
			barrier->Arrive();

		}

		accum *= histsize;

		// 5. Swap in and out buffers.
		//
		if (threadid == 0) {
			Page* tmp = oldbuf;
			oldbuf = newbuf;
			newbuf = tmp;
#ifdef DEBUG
			unsigned long long ulltotaltuples = totaltuples;
			memset(newbuf->getTupleOffset(0), 0xDEADDEAD, 
					ulltotaltuples * schema.getTupleSize());
#endif
		}
		barrier->Arrive();

	}

	return offsets[totalpasses-1];
}

unsigned int pageCopy(void* dest, const Buffer* page);

/** 
 * Copies tuples from \a t to \a p in parallel. Does not reset \a t.
 */
void parallelCopyTuples(
		const unsigned int threadid, const unsigned int nthreads,
		vector<PageCursor*>& inputs, vector<unsigned int>& destoffset,
		Page* p, Schema& s) 
{
#ifdef DEBUG
	assert(inputs.size() == destoffset.size());
#endif

	int tupnum;
	Page* page;
	void* tup;
	unsigned int base = destoffset[threadid];

	while (page = inputs[threadid]->readNext()) {
		void* dest = p->getTupleOffset(base);
		base += (pageCopy(dest, page) / s.getTupleSize());
	}
}

SplitResult RadixPartitioner::split(int threadid) 
{
	parallelCopyTuples(threadid, nthreads, 
			partitionedinput, destoffsets, oldbuf, schema);
	barrier->Arrive();

	vector<unsigned int>& splitresult = realsplit(threadid);

	// construct return objects from oldbuf
	placecursors(threadid, nthreads, schema.getTupleSize(), 
			splitresult, getOutputBuffer(), &result);
	barrier->Arrive();

	return &result;
}

/** 
 * Counts tuples in \a t, populates histogram and resets the PageCursor.
 */
unsigned long RadixPartitioner::countTuples(PageCursor* t) {
	unsigned long ret = 0;

	int tupnum;
	Page* page;
	void* tup;

	t->reset();

	while (page = t->readNext()) {
		tupnum = 0;
		while (tup = page->getTupleOffset(tupnum++)) {
			++ret;
			unsigned long long curkey = schema.asLong(tup, attribute);
			maxkey = max(maxkey, curkey);
		}
	}

	t->reset();

	return ret;
}

/** 
 * Copies tuples from \a t to \a p and resets \a t.
 */
void copyTuples(PageCursor* t, Page* p, Schema& s) 
{
	int tupnum;
	Page* page;
	void* tup;

	t->reset();

	while (page = t->readNext()) {
		tupnum = 0;
		while (tup = page->getTupleOffset(tupnum++)) {
			assert(p->canStoreTuple());
			void* dest = p->allocateTuple();
			s.copyTuple(dest, tup);
		}
	}

	t->reset();
}

/**
 * Read all input in memory, store consequtively.
 * (This time is not counted in perf experiments.)
 */
void RadixPartitioner::init(PageCursor* t) 
{
	unsigned int accum = 0;

	Partitioner::init(t);
	schema = *(t->schema());

	// Scan once, get tuple count and byte count.
	totaltuples = countTuples(t);

#ifdef DEBUG
	assert(totaltuples!=0);
#endif

	unsigned int tupsz = schema.getTupleSize();
	unsigned long long ulltotaltuples = totaltuples; 
	oldbuf = new Page(ulltotaltuples*tupsz, tupsz);
	newbuf = new Page(ulltotaltuples*tupsz, tupsz);

	// Scan twice, copy data. 
	copyTuples(t, newbuf, schema);
	copyTuples(t, oldbuf, schema);

	memset(oldbuf->getTupleOffset(0), 0xDEADDEAD, 
			ulltotaltuples * schema.getTupleSize());
	memset(newbuf->getTupleOffset(0), 0xDEADDEAD, 
			ulltotaltuples * schema.getTupleSize());
	
	// Prepare temporary objects for split().
	unsigned int maxhashkey = 0; //<<< max hash key in all iterations
	vec_hf = dynamic_cast<ModuloHashFunction*>(hashfn)->generate(totalpasses);
	for (int pass=0; pass<totalpasses; ++pass) {
		maxhashkey = max(maxhashkey, vec_hf[pass]->buckets());
	}

	// Split t among multiple threads.
	//
	partitionedinput = t->split(nthreads);
#ifdef DEBUG
	assert(partitionedinput.size() == nthreads);
#endif
	accum = 0;
	destoffsets.push_back(accum);
	for (int i=0; i<nthreads-1; ++i) {
		accum += countTuples(partitionedinput[i]);
		destoffsets.push_back(accum);
	}

	// If threadid==0, local histogram is always 0. However, we have to
	// remember how many tuples we have seen for each hash bucket. 
	//
	for (int trd=0; trd<nthreads; ++trd) {
		histograms.push_back(vector<unsigned int>(maxhashkey, 0));
	}

	// Global histograms. The thread with threadid==0 is responsible for
	// computing and resetting them.
	//
	accum = 1;
	for (int pass=0; pass<totalpasses; ++pass) {
		accum *= vec_hf[pass]->buckets();
		offsets.push_back(vector<unsigned int>(accum, 0));
	}	

	for (int i=0; i<hashfn->buckets(); ++i) {
		FakeTable* tmp = new FakeTable(&schema);
		result.push_back(tmp);
	}
}

void RadixPartitioner::destroy()
{
	input->close();	// this should be a no-op
	for (int i=0; i<partitionedinput.size(); ++i)
		partitionedinput[i]->close();
	partitionedinput.clear();
	destoffsets.clear();

	for (int i=0; i<vec_hf.size(); ++i)
		delete vec_hf[i];
	vec_hf.clear();
	histograms.clear();
	offsets.clear();

	delete oldbuf;
	delete newbuf;

	Partitioner::destroy();
}
