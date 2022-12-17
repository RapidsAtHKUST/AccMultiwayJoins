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

#ifndef __MYBUCKET__
#define __MYBUCKET__

#include <cassert>

#include "page.h"

/**
 * @deprecated
 */
class Bucket : public Page {
	public:
		/**
		 * Creates a bucket of size \a size, holding tuples which are
		 * \a tuplesize bytes each.
		 * \param size Bucket size in bytes.
		 * \param tuplesize Size of tuples in bytes.
		 */
		Bucket(unsigned int size, unsigned int tuplesize) 
			: Page(size, tuplesize), next(0) { }

		/** 
		 * Returns a pointer to next bucket.
		 * @return Next bucket, or NULL if it doesn't exist.
		 */
		Bucket* getNext();

		/**
		 * Sets the pointer to the next bucket.
		 * @param bucket Pointer to next bucket.
		 */
		void setNext(Bucket* const bucket);

	private:
		Bucket* next;
};

inline void Bucket::setNext(Bucket* const bucket) {
	next = bucket;
}

inline Bucket* Bucket::getNext() {
	return next;
}

#endif
