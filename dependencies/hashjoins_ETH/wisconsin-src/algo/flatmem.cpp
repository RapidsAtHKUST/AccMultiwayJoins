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

#include "algo.h"

FlatMemoryJoiner::FlatMemoryJoiner (const libconfig::Config& cfg) 
	: HashBase(cfg.getRoot()), 
		partitioner(cfg, cfg.getRoot()["hash"], cfg.getRoot()["hash"]), 
		probetable(NULL), totaltuples(0), phistogram(NULL)
{
}

FlatMemoryJoiner::~FlatMemoryJoiner()
{
}

void FlatMemoryJoiner::destroy()
{
	delete probetable;
	partitioner.destroy();
	for (int i=0; i<nthreads; ++i) {
		//	result[i]->close();
		//	(main.cpp will free output buffers)
	}
	result.clear();
	HashBase::destroy();
}

/** 
 * Counts tuples in \a t.
 */
unsigned long countTuples(PageCursor* t) {
	unsigned long ret = 0;

	int tupnum;
	Page* page;
	void* tup;

	t->reset();

	while (page = t->readNext()) {
		tupnum = 0;
		while (tup = page->getTupleOffset(tupnum++)) {
			++ret;
		}
	}

	t->reset();

	return ret;
}

void copyTuples(PageCursor* t, Page* p, Schema& s);


void FlatMemoryJoiner::init(
	Schema* schema1, vector<unsigned int> select1, unsigned int jattr1,
	Schema* schema2, vector<unsigned int> select2, unsigned int jattr2)
{
	// This class only works under a special build format.
	// Do simple sanity checks to enforce that.
	assert(jattr1 == 0);
	assert(schema1->columns() == select1.size() + 1);
	for (int i=0; i<select1.size(); ++i) {
		assert(select1[i] == i+1);
	}

	HashBase::init(schema1, select1, jattr1, schema2, select2, jattr2);

	for (int i=0; i<nthreads; ++i) {
		WriteTable* wt = new WriteTable();
		wt->init(sout, outputsize);
		result.push_back(wt);
	}
}

void FlatMemoryJoiner::custominit(Table* tbuild, Table* tprobe)
{
	partitioner.init(tbuild);

	unsigned int tupsz = tprobe->schema()->getTupleSize();

	totaltuples = countTuples(tprobe);

	unsigned long long ulltotaltuples = totaltuples; 
	probetable = new Page(ulltotaltuples*tupsz, tupsz);
	copyTuples(tprobe, probetable, *(tprobe->schema()));
}

void FlatMemoryJoiner::build(SplitResult ignored, int threadid) 
{
	vector<unsigned int>& ret = partitioner.realsplit(threadid);
	if (threadid == 0) {
		phistogram = &ret;
	}
}

PageCursor* FlatMemoryJoiner::probe(SplitResult ignored, int threadid) 
{
	WriteTable& ret = *(result[threadid]);
	const vector<unsigned int>& histogram = *phistogram;
	Page& buildpage = *(partitioner.getOutputBuffer());
	Page& probepage = *probetable;

	unsigned int items = totaltuples / nthreads;
	unsigned int start = items * threadid;
	if (threadid == nthreads-1) {
		// Compensate for last few tuples, 
		// if \a items has been rounded down.
		items = totaltuples - start;
	}

	void* tup1;	///< points to the tuple from build
	void* tup2; ///< points to the tuple from probe
	unsigned int bstart;	///< start of hash bucket (an offset) in build page
	unsigned int bitems;	///< number of tuples in hash bucket
	char tmp[sout->getTupleSize()];	///< output tuple construction space
	unsigned int curbuc;	///< current hash bucket being examined
	
	for (unsigned int probetid = start; 
			probetid < start + items; 
			++probetid) {

		tup2 = probepage.getTupleOffset(probetid);
		curbuc = _hashfn->hash(s2->asLong(tup2, ja2));

		bstart = (curbuc != 0 ? histogram[curbuc-1] : 0);
		bitems = histogram[curbuc] - bstart;
		for (unsigned int buildtid = bstart; 
				buildtid < bstart + bitems; 
				++buildtid) {
			tup1 = buildpage.getTupleOffset(buildtid);

			if ( sbuild->asLong(tup1,ja1) != s2->asLong(tup2,ja2) )
				continue;

			// Now, tup1 and tup2 match. Bring the bytes together.
#if defined(OUTPUT_ASSEMBLE)
			// copy payload of first tuple to destination
			if (s1->getTupleSize()) 
				s1->copyTuple(tmp, sbuild->calcOffset(tup1,1));

			// copy each column to destination
			for (unsigned int j=0; j<sel2.size(); ++j)
				sout->writeData(tmp,		// dest
						s1->columns()+j,	// col in output
						s2->calcOffset(tup2, sel2[j]));	// src for this col
#if defined(OUTPUT_WRITE_NORMAL)
			ret.append(tmp);
#elif defined(OUTPUT_WRITE_NT)
			ret.nontemporalappend16(tmp);
#endif
#endif

#if !defined(OUTPUT_AGGREGATE) && !defined(OUTPUT_ASSEMBLE)
			__asm__ __volatile__ ("nop");
#endif
		}

	}

	return result[threadid];
}
