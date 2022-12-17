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

#include "page.h"
#include "custom_asserts.h"

#include <cstring>

#ifndef NULL
#define NULL 0
#endif

Buffer::Buffer(void* data, unsigned int size, void* free)
	: data(data), maxsize(size), owner(false), free(free)
{ 
	if (free == NULL)
		this->free = reinterpret_cast<char*>(data)+size;
}

Buffer::Buffer(unsigned long size)
	: maxsize(size), owner(true)
{
	data = new char[size];
	dbgassert(data != NULL);
#ifdef DEBUG
	memset(data, 0xBF, size);
#endif
	free = data;
}

Buffer::~Buffer() 
{
	if (owner)
		delete[] reinterpret_cast<char*>(data);

	data = 0;
	free = 0;
	maxsize = 0;
	owner = false;
}

TupleBuffer::TupleBuffer(unsigned long size, unsigned int tuplesize)
	: Buffer(size), tuplesize(tuplesize)
{ 
	// Sanity check: Fail if page doesn't fit even a single tuple.
	//
	dbgassert(size >= tuplesize);
}

TupleBuffer::TupleBuffer(void* data, unsigned int size, void* free, unsigned int tuplesize)
	: Buffer(data, size, free), tuplesize(tuplesize)
{ 
	// Sanity check: Fail if page doesn't fit even a single tuple.
	//
	dbgassert(size >= tuplesize);
}

unsigned int pageCopy(void* dest, const Buffer* page)
{
	unsigned int ret = ((char*)(page->free)) - ((char*)(page->data));
	memcpy(dest, page->data, ret);
	return ret;
}
