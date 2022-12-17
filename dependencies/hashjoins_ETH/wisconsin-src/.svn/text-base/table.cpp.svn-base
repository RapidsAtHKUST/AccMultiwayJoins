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

#include "custom_asserts.h"
#include "table.h"
#include "loader.h"
#include <glob.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

//extern "C" {
#include "generator.h"  /* create_relation_xx */
//}

void Table::init(Schema *s, unsigned int size) 
{
	_schema = s;
	data = 0;
	cur = 0;
}

void Table::close() {
	LinkedTupleBuffer* t = data;
	while (t) {
		t=data->getNext();
		delete data;
		data = t;
	}

	reset();
}

void WriteTable::append(const char** data, unsigned int count) {
	unsigned int s = _schema->getTupleSize();
	if (!last->canStore(s)) {
		// create a new bucket
		LinkedTupleBuffer* tmp = new LinkedTupleBuffer(size, s);
		// link it as the next bucket of last
		last->setNext(tmp);
		// make last point to the new bucket
		last = tmp;
	}

	void* target = last->allocateTuple();
	dbg2assert(target!=NULL);
	dbg2assert(count==_schema->columns());
	_schema->parseTuple(target, data);
}

void WriteTable::append(const vector<string>& input) {
	unsigned int s = _schema->getTupleSize();
	if (!last->canStore(s)) {
		// create a new bucket
		LinkedTupleBuffer* tmp = new LinkedTupleBuffer(size, s);
		// link it as the next bucket of last
		last->setNext(tmp);
		// make last point to the new bucket
		last = tmp;
	}

	void* target = last->allocateTuple();
	dbg2assert(target!=NULL);
	_schema->parseTuple(target, input);
}

void WriteTable::append(const void* const src) {
	unsigned int s = _schema->getTupleSize();

	if (!last->canStore(s)) {
		// create a new bucket
		LinkedTupleBuffer* tmp = new LinkedTupleBuffer(size, s);
		// link it as the next bucket of last
		last->setNext(tmp);
		// make last point to the new bucket
		last = tmp;
	}

	void* target = last->allocateTuple();
	dbg2assert(target!=NULL);
	_schema->copyTuple(target, src);
}

void WriteTable::nontemporalappend16(const void* const src) {
	unsigned int s = _schema->getTupleSize();
	dbgassert(s==16);

	if (!last->canStore(s)) {
		// create a new bucket
		LinkedTupleBuffer* tmp = new LinkedTupleBuffer(size, s);
		// link it as the next bucket of last
		last->setNext(tmp);
		// make last point to the new bucket
		last = tmp;
	}

	void* target = last->allocateTuple();
	dbg2assert(target!=NULL);
#ifdef __x86_64__
	__asm__ __volatile__ (
			"	movntiq %q2, 0(%4)	\n"
			"	movntiq %q3, 8(%4)	\n"
			: "=m" (*(unsigned long long*)target), "=m" (*(unsigned long long*)((char*)target+8))
			: "r" (*(unsigned long long*)((char*)src)), "r" (*((unsigned long long*)(((char*)src)+8))), "r" (target)
			: "memory");
#else
#warning MOVNTI not known for this architecture
	_schema->copyTuple(target, src);
#endif
}

void WriteTable::concatenate(const WriteTable& table)
{
#ifdef DEBUG
	assert(_schema->getTupleSize() == table._schema->getTupleSize());
#endif
	if (data == 0) {
		data = table.data;
	}
	last->setNext(table.data);
	last = table.last;
}

void WriteTable::init(Schema* s, unsigned int size)
{
	Table::init(s, size);

	this->size=size;
	data = new LinkedTupleBuffer(size, s->getTupleSize());
	last = data;
	cur = data;
}

void AtomicWriteTable::append(const void* const src) {
	unsigned int s = _schema->getTupleSize();
	lock.lock();
	if (!last->canStore(s)) {
		// create a new bucket
		LinkedTupleBuffer* tmp = new LinkedTupleBuffer(size, s);
		// link it as the next bucket of last
		last->setNext(tmp);
		// make last point to the new bucket
		last = tmp;
	}

	void* target = last->allocateTuple();
	lock.unlock();
	dbg2assert(target!=NULL);
	_schema->copyTuple(target, src);
}

void AtomicWriteTable::append(const vector<string>& input) {
	unsigned int s = _schema->getTupleSize();
	lock.lock();
	if (!last->canStore(s)) {
		// create a new bucket
		LinkedTupleBuffer* tmp = new LinkedTupleBuffer(size, s);
		// link it as the next bucket of last
		last->setNext(tmp);
		// make last point to the new bucket
		last = tmp;
	}

	void* target = last->allocateTuple();
	lock.unlock();
	dbg2assert(target!=NULL);
	_schema->parseTuple(target, input);
}

/**
 * Returns end of chain starting at \a head, NULL if \a head is NULL.
 */
LinkedTupleBuffer* findChainEnd(LinkedTupleBuffer* head)
{
	LinkedTupleBuffer* prev = NULL;
	while (head) 
	{
		prev = head;
		head = head->getNext();
	}
	return prev;
}

Table::LoadErrorT WriteTable::load(const string& filepattern, 
		const string& separators)
{
	Loader loader(separators[0]);
	loader.load(filepattern, *this);
	return LOAD_OK;
}

Table::LoadErrorT WriteTable::generate(int relationSize, int alphSize, 
                                       double zipfParam, int seed)
{
    relation_t rel;

    seed_generator(seed);

    if(zipfParam > 0.0) {
        // create rel with zipf
        create_relation_zipf(&rel, relationSize, alphSize, zipfParam);
    }
    else if(relationSize == alphSize){
        // create pk relation
        create_relation_pk(&rel, relationSize);
    }
    else {
        // create fk relation
        create_relation_fk(&rel, relationSize, alphSize);
    }

    for(int i = 0; i < rel.num_tuples; i++){
        append((const void* const)&rel.tuples[i]);
    }

    delete_relation(&rel);

    return LOAD_OK;
}

/**
 * Splits pages from this table, round-robin, to \a nthreads new PageCursors.
 * Call invalidates this table. 
 */
vector<PageCursor*> Table::split(int nthreads)
{
	vector<PageCursor*> ret = vector<PageCursor*>();

	for (int i=0; i<nthreads; ++i) {
		Table* pt = new WriteTable();
		pt->init(_schema, 0);
		ret.push_back(pt);
	}

	int curthr = 0;
	Bucket* pb = getRoot();
	Bucket* pbnew = NULL;
	Table* curtbl = NULL;
	while (pb) {
		pbnew = pb->getNext();

		curtbl = (Table*) ret[curthr];
		pb->setNext(curtbl->data);
		curtbl->data = pb;

		curthr = (curthr + 1) % nthreads;
		pb = pbnew;
	}

	for (int i=0; i<nthreads; ++i) {
		ret[i]->reset();
	}

	data = NULL;
	cur = NULL;

	return ret;
}

