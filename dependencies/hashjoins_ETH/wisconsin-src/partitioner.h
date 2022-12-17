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

#ifndef __MYPARTITIONER__
#define __MYPARTITIONER__

#include <vector>
#include <libconfig.h++>
using std::vector;

#include "table.h"
#include "Barrier.h"
#include "hash.h"

typedef vector<PageCursor*>* SplitResult;

class Partitioner {
	public:
		Partitioner(const libconfig::Config& cfg, 
				const libconfig::Setting& node,
				const libconfig::Setting& hashnode);
		~Partitioner();
		virtual void init(PageCursor* t);
		virtual SplitResult split(int threadid);
		virtual void destroy();

	protected:
		int nthreads;
		vector<PageCursor*> result;
		PageCursor* input;
		PThreadLockCVBarrier* barrier;
		int pagesize;
		HashFunction* hashfn;
		int attribute;
};

class ParallelPartitioner : public Partitioner {
	public:
		ParallelPartitioner(const libconfig::Config& cfg, 
				const libconfig::Setting& node,
				const libconfig::Setting& hashnode);
		virtual void init(PageCursor* t);
		virtual SplitResult split(int threadid);
		virtual void destroy();
};

class IndependentPartitioner : public Partitioner {
	public:
		IndependentPartitioner(const libconfig::Config& cfg, 
				const libconfig::Setting& node,
				const libconfig::Setting& hashnode);
		virtual void init(PageCursor* t);
		virtual SplitResult split(int threadid);
		virtual void destroy();
	private:
		vector<WriteTable*>* tempstore;
};

class DerekPartitioner : public Partitioner {
	public:
		DerekPartitioner(const libconfig::Config& cfg, 
				const libconfig::Setting& node,
				const libconfig::Setting& hashnode);
		virtual SplitResult split(int threadid);
};

class RadixPartitioner : public Partitioner {
	public:
		RadixPartitioner(const libconfig::Config& cfg, 
				const libconfig::Setting& node,
				const libconfig::Setting& hashnode);
		virtual void init(PageCursor* t);
		virtual SplitResult split(int threadid);
		virtual void destroy();

		vector<unsigned int>& realsplit(int threadid);
		inline Page* getOutputBuffer() { return oldbuf; }

	private:
		unsigned long countTuples(PageCursor* t);

		// temporary objects here
		vector<HashFunction*> vec_hf;
		unsigned long long maxkey;
		unsigned short totalpasses;
		unsigned long totaltuples;
		Page* oldbuf;
		Page* newbuf;
		Schema schema;
		vector< vector<unsigned int> > histograms;
		vector< vector<unsigned int> > offsets;

		vector<PageCursor*> partitionedinput;
		vector<unsigned int> destoffsets;
};

#endif
