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

NestedLoops::NestedLoops(const libconfig::Setting& cfg) 
	: BaseAlgo(cfg), t1(0) {
	size = cfg["bucksize"];
}

/** 
 * Build phase reads tuples from \a t and stores them, join attribute first, in
 * a new table pointed to by private variable \a t1.
 */
void NestedLoops::build(SplitResult tin, int threadid) {
	WriteTable* wrtmp = new WriteTable();
	wrtmp->init(sbuild, size);
	PageCursor* t = (*tin)[0];
	prepareBuild(wrtmp, t);
	t->reset();
	t1 = wrtmp;
}

void NestedLoops::prepareBuild(WriteTable* wrtmp, PageCursor* t) {
	char tmp[sout->getTupleSize()];
	int i = 0;
	void* tup;
	Page* b;
	while(b = t->readNext()) {
		i = 0;
		while(tup = b->getTupleOffset(i++)) {
			sbuild->writeData(tmp, 0, t->schema()->calcOffset(tup, ja1));
			for (unsigned int j=0; j<sel1.size(); ++j)
				sbuild->writeData(tmp,		// dest
						j+1,	// col in output
						t->schema()->calcOffset(tup, sel1[j]));	// src for this col
			wrtmp->append(tmp);
		}
	}
}

/** Probe phase does a nested loops join between \a t and local \a t1. */
PageCursor* NestedLoops::probe(SplitResult tin, int threadid) {
	WriteTable* ret = new WriteTable();
	ret->init(sout, size);
	Page* p1, * p2;
	PageCursor* t = (*tin)[0];
	while(p1 = t1->readNext()) {
		while(p2 = t->readNext()) {
			joinPagePage1(ret, p1, p2);
		}
		t->reset();
	}
	return ret;
}

void NestedLoops::destroy() {
	BaseAlgo::destroy();
	delete t1;
}

void NestedLoops::joinPagePage1(WriteTable* output, Page* p1, Page* p2) {
	int i = 0;
	void* tup;
	while (tup = p2->getTupleOffset(i++)) {
		joinPageTup(output, p1, tup);
	}
}

void NestedLoops::joinPageTup(WriteTable* output, Page* page, void* tuple) {
	int i = 0;
	char tmp[sout->getTupleSize()];
	void* tup;
	while (tup = page->getTupleOffset(i++)) {
		if (sbuild->asLong(tup,0) == s2->asLong(tuple,ja2) ) {
			// copy payload of first tuple to destination
			if (s1->getTupleSize()) 
				s1->copyTuple(tmp, sbuild->calcOffset(tup,1));

			// copy each column to destination
			for (unsigned int j=0; j<sel2.size(); ++j)
				sout->writeData(tmp,		// dest
						s1->columns()+j,	// col in output
						s2->calcOffset(tuple, sel2[j]));	// src for this col
#ifndef SUSPEND_OUTPUT
			output->append(tmp);
#endif
		}
	}
}

