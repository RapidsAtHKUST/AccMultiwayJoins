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

#include "atomics.h"

inline bool Buffer::isValidAddress(void* loc, unsigned int len) 
{
	return (data <= loc) && (loc <= ((char*)data + maxsize - len));
}

inline bool Buffer::canStore(unsigned int len) 
{
	return isValidAddress(free, len);
}

inline void* Buffer::allocate(unsigned int len) 
{
	if (!canStore(len))
		return 0; //nullptr;

	void* ret = free;
	free = reinterpret_cast<char*>(free) + len;
	return ret;
}

inline void* Buffer::atomicAllocate(unsigned int len) 
{
	void* oldval;
	void* newval;
	void** val = &free;

	newval = *val;
	do {
		if (!canStore(len))
			return 0; //nullptr;

		oldval = newval;
		newval = (char*)oldval + len;
		newval = atomic_compare_and_swap(val, oldval, newval);
	} while (newval != oldval);
	return newval;
}

inline const unsigned int Buffer::getUsedSpace()
{
	return reinterpret_cast<char*>(free) - reinterpret_cast<char*>(data);
}

inline void* TupleBuffer::getTupleOffset(unsigned int pos) 
{
	char* f = reinterpret_cast<char*>(free);
	char* d = reinterpret_cast<char*>(data);
	char* ret = d + static_cast<unsigned long long>(pos)*tuplesize;
	return ret < f ? ret : 0;
}

inline void* TupleBuffer::allocateTuple()
{
	return Buffer::allocate(tuplesize);
}

inline void* TupleBuffer::atomicAllocateTuple()
{
	return Buffer::atomicAllocate(tuplesize);
}

inline bool TupleBuffer::isValidTupleAddress(void* loc) 
{
	return Buffer::isValidAddress(loc, tuplesize);
}
