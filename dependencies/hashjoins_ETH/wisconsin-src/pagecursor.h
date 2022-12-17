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

#ifndef __MYPAGECURSOR__
#define __MYPAGECURSOR__

#include "page.h"
#include "schema.h"

class PageCursor {
	public:
		/**
		 * Returns the next non-requested page, or NULL if no next page exists.
		 * Not necessarily thread-safe!!!
		 */
		virtual Page* readNext() { return atomicReadNext(); }

		/**
		 * Returns the next non-requested page, or NULL if no next page exists.
		 * A synchronized version of readNext().
		 */
		virtual Page* atomicReadNext() = 0;

		/** Return \ref Schema object for all pages. */
		virtual Schema* schema() = 0;

		/**
		 * Resets the reading point to the start of the table.
		 */
		virtual void reset() { }

		/**
		 * Closes the cursor.
		 */
		virtual void close() { }

		virtual vector<PageCursor*> split(int nthreads) { }
};

#endif // __MYPAGECURSOR__
