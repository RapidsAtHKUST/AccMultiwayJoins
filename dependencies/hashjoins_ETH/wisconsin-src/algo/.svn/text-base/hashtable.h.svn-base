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

#include "../lock.h"

#ifdef DEBUG
#include <cassert>
#endif

class HashTable {
	public:
		void init(unsigned int nbuckets, unsigned int bucksize, unsigned int tuplesize);
		void destroy();

		/**
		 * Allocates a tuple at bucket at \a offset. Call is not atomic and
		 * might result in a new memory allocation if page is full.
		 * @return Location that has \a tuplesize bytes for writing.
		 */
		void* allocate(unsigned int offset);

		/**
		 * Allocates a tuple at bucket at \a offset atomically.
		 * @return Location that has \a tuplesize bytes for writing.
		 */
		inline void* atomicAllocate(unsigned int offset)
		{
#ifdef DEBUG
			assert(0 <= offset && offset<nbuckets);
#endif
			void* ret;
			lock[offset].lock();
			ret = allocate(offset);
			lock[offset].unlock();
			return ret;
		}

		class Iterator {
			friend class HashTable;
			public:
				Iterator() : bucksize(0), tuplesize(0), cur(0), next(0), free(0) { }
				Iterator(unsigned int bucksize, unsigned int tuplesize);
				
				inline void* readnext() 
				{
					void* ret;

					if (cur < free) {
						ret = cur;
						cur = ((char*)cur) + tuplesize;
						return ret;
					} else if (next != 0) {
						ret = next;
						cur = ((char*)next) + tuplesize;
						free = *(void**)((char*)next + bucksize);
						next = *(void**)((char*)next + bucksize + sizeof(void*));
						// Return null if last chunk exists but is empty.
						// Caveat: will not work correctly if there are empty
						// chunks in the chain.
						return ret < free ? ret : 0;
					} else {
						return 0;
					}
				}

			private:
				void* cur;
				void* free;
				void* next;
				const unsigned int bucksize;
				const unsigned int tuplesize;
		};

		Iterator createIterator();

		inline void placeIterator(Iterator& it, unsigned int offset)
		{
			void* start = bucket[offset];
			it.cur = start;
			it.free = *(void**)((char*)start + bucksize);
			it.next = *(void**)((char*)start + bucksize + sizeof(void*));
		}

		inline void prefetch(unsigned int offset)
		{
#ifdef __x86_64__
			__asm__ __volatile__ ("prefetcht0 %0" :: "m" (*(unsigned long long*) bucket[offset & (nbuckets-1)]));
#endif
		}

	private:
		Lock* lock;
		void** bucket;
		
		unsigned int tuplesize;
		unsigned int bucksize; 
		unsigned int nbuckets; 
};
