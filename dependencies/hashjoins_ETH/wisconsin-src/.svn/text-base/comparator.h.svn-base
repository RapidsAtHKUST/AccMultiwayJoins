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

#ifndef __MY_COMPARATOR__
#define __MY_COMPARATOR__

#include "schema.h"

class Comparator {
	public:
		/**
		 * Initializes comparator object.
		 * @param lct ColumnSpec in left tuple.
		 * @param loff Offset in left tuple.
		 * @param rct ColumnSpec in right tuple.
		 * @param roff Offset in right tuple.
		 */
		Comparator() : loffset(0), roffset(0), size(0), eq(NULL), less(NULL) { }
		
		void init(ColumnSpec lct, unsigned int loff, 
				ColumnSpec rct, unsigned int roff);

		bool equal(void* ltup, void* rtup) {
			char* lreal = (char*)ltup + loffset;
			char* rreal = (char*)rtup + roffset;
			return eq(lreal, rreal, size);
		}

		bool lessthan(void* ltup, void* rtup) {
			char* lreal = (char*)ltup + loffset;
			char* rreal = (char*)rtup + roffset;
			return less(lreal, rreal, size);
		}

		bool equallessthan(void* ltup, void* rtup) {
			return (lessthan(ltup, rtup)) || (equal(ltup, rtup));
		}

		/* Assuming Less, Eq, Greater are collectivelly exhaustive.
		 * Therefore, if Less and Eq are both false, then Greater is true.
		 */

		bool greaterthan(void* ltup, void* rtup) {
			return (!lessthan(ltup, rtup)) && (!equal(ltup, rtup));
		}

		bool equalgreaterthan(void* ltup, void* rtup) {
			return (!lessthan(ltup, rtup));
		}

	private:
		int loffset, roffset;
		int size;	// passed to strncmp for CT_CHAR comparisons
		bool (*eq)(void*, void*, int);
		bool (*less)(void*, void*, int);
};

#endif
