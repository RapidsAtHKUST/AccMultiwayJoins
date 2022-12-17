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

#include "hashtable.h"
#include "../exceptions.h"
#include <stdlib.h>

void * myalloc(int size){
  void * mem = 0;
  posix_memalign(&mem, 64, size);
  return mem;
}

void HashTable::init(unsigned int nbuckets, unsigned int bucksize, unsigned int tuplesize)
{
  lock = new Lock[nbuckets];
  bucket = (void**)new char[sizeof(void*) * nbuckets];
	for (int i=0; i<nbuckets; ++i) {
		// allocating space for data + free pointer + next pointer
	  bucket[i] = new char[bucksize + 2*sizeof(void*)];//myalloc(bucksize + 2*sizeof(void*));//
		void** free = (void**)(((char*)bucket[i]) + bucksize);
		void** next = (void**)(((char*)bucket[i]) + bucksize + sizeof(void*));
		*next = 0;
		*free = bucket[i];
	}

	this->bucksize = bucksize;
	this->tuplesize = tuplesize;
	this->nbuckets = nbuckets;
}

void HashTable::destroy()
{
	for (int i=0; i<nbuckets; ++i) {
		void* tmp = bucket[i];
		void* cur = *(void**)(((char*)bucket[i]) + bucksize + sizeof(void*));
		while (cur) {
			delete[] (char*)tmp;
			tmp = cur;
			cur = *(void**)(((char*)cur) + bucksize + sizeof(void*));
		} 
		delete[] (char*) tmp;
	}

	delete[] lock;
	delete[] (char*) bucket;
}

void* HashTable::allocate(unsigned int offset)
{
#ifdef DEBUG
	assert(0 <= offset && offset<nbuckets);
#endif
	void* data = bucket[offset];
	void** freeloc = (void**)((char*)data + bucksize);
	void* ret;
	if ((*freeloc) <= ((char*)data + bucksize - tuplesize)) {
		// Fast path: it fits!
		//
		ret = *freeloc;
		*freeloc = ((char*)(*freeloc)) + tuplesize;
		return ret;
	}

	// Allocate new page and make bucket[offset] point to it.
	//
	//throw PageFullException(offset);
	
	ret = new char[bucksize + 2*sizeof(void*)];
	bucket[offset] = ret;

	void** nextloc = (void**)(((char*)ret) + bucksize + sizeof(void*));
	*nextloc = data;

	freeloc = (void**)(((char*)ret) + bucksize);
	*freeloc = ((char*)ret) + tuplesize;

	return ret;
}

HashTable::Iterator HashTable::createIterator()
{
	return Iterator(bucksize, tuplesize);
}

HashTable::Iterator::Iterator(unsigned int bucksize, unsigned int tuplesize)
	: bucksize(bucksize), tuplesize(tuplesize), cur(0), free(0), next(0)
{
}
