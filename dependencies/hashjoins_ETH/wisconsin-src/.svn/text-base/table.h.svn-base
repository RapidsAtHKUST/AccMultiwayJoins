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

#ifndef __MYTABLE__
#define __MYTABLE__

#include <string>
#include <vector>
#include "schema.h"
#include "page.h"
#include "lock.h"

class TupleBufferCursor;
typedef TupleBufferCursor PageCursor;

class TupleBufferCursor 
{
	public:
		/**
		 * Returns the next non-requested page, or NULL if no next page exists.
		 * Not necessarily thread-safe!!!
		 */
		virtual TupleBuffer* readNext() { return atomicReadNext(); }

		/**
		 * Returns the next non-requested page, or NULL if no next page exists.
		 * A synchronized version of readNext().
		 */
		virtual TupleBuffer* atomicReadNext() = 0;

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

		virtual vector<PageCursor*> split(int nthreads) = 0;

		virtual ~TupleBufferCursor() { }
};

/**
 * Container for a linked list of LinkedTupleBuffer.
 * Table is not useful, it is mainly used to hide append() method from 
 * inappropriate places in the code.
 */
class Table : public TupleBufferCursor {
	public:
		/**
		 * Constructs a new table.
		 * @param s Reference to the schema of the new table.
		 */
		Table() : _schema(NULL), data(NULL), cur(NULL) { }
		virtual ~Table() { }

		enum LoadErrorT
		{
			LOAD_OK = 0
		};

		virtual LoadErrorT load(const string& filepattern, 
				const string& separators) = 0;

		/**
		 * Initializes a new table.
		 * @param s The schema of the new table.
		 */
		virtual void init(Schema* s, unsigned int size);

		/** 
		 * Returns the first bucket of the list.
		 * @return Firt bucket in the list.
		 */
		inline LinkedTupleBuffer* getRoot()
		{
			return data;
		}

		/**
		 * Returns the next non-requested bucket, or NULL if no next bucket exists.
		 * Not thread-safe!!!
		 */
		inline LinkedTupleBuffer* readNext()
		{
			LinkedTupleBuffer* ret = cur;
			if (cur)
				cur = cur->getNext();
			return ret;
		}


		/**
		 * Returns the next non-requested bucket, or NULL if no next bucket exists.
		 * A synchronized version of readNext().
		 */
		inline LinkedTupleBuffer* atomicReadNext()
		{
			LinkedTupleBuffer* oldval;
			LinkedTupleBuffer* newval;

			newval = cur;

			do
			{
				if (newval == NULL)
					return NULL;

				oldval = newval;
				newval = oldval->getNext();
				newval = (LinkedTupleBuffer*)atomic_compare_and_swap((void**)&cur, oldval, newval);

			} while (newval != oldval);

			return newval;
		}

		/**
		 * Resets the reading point to the start of the table.
		 */
		inline void reset()
		{
			cur = data;
		}

		/**
		 * Close the table, ie. destroy all data associated with it.
		 * Not closing the table will result in a memory leak.
		 */
		virtual void close();

		/** Return associated \ref Schema object. */
		Schema* schema() 
		{ 
			return _schema;
		}

		vector<PageCursor*> split(int nthreads);

	protected:
		Schema* _schema;
		LinkedTupleBuffer* data;
		/* volatile */ LinkedTupleBuffer* cur;
};

class WriteTable : public Table {
	public:
		WriteTable() : last(NULL), size(0) { }
		virtual ~WriteTable() { }

		virtual void whatever()  { }
		void init(Schema* s, unsigned int size);

		/**
		 * Loads a single text file, where each line is a tuple and each field
		 * is separated by any character in the \a separators string.
		 */
		LoadErrorT load(const string& filepattern, const string& separators);

    /**
     * Generates tuples on-the-fly in memory.
     */
    LoadErrorT generate(int relationSize, int alphSize, 
                        double zipfParam, int seed);
		/** 
		 * Appends the \a input at the end of this table, creating new
		 * buckets as necessary.
		 */
		virtual void append(const vector<string>& input);
		virtual void append(const char** data, unsigned int count);
		virtual void append(const void* const src);

		void nontemporalappend16(const void* const src);

		/**
		 * Appends the table to this table.
		 * PRECONDITION: Caller must check that schemas are same.
		 */
		void concatenate(const WriteTable& table);

	protected:
		LinkedTupleBuffer* last;
		unsigned int size;
};

class AtomicWriteTable : public WriteTable {
	public:
		virtual ~AtomicWriteTable() { }

		virtual void append(const vector<string>& input);
		virtual void append(const void* const src);
	private:
		Lock lock;
};

class FakeTable : public PageCursor {
	public:
		FakeTable(Schema* s)
			: sch(s), firsttime(false) 
		{ }

		void place(unsigned int pagesz, unsigned int tuplesz, void* data) 
		{
			fakepage.place(pagesz, tuplesz, data);
			firsttime = true;
		}

		virtual Page* atomicReadNext()
		{
			if (firsttime) {
				firsttime = false;
				return &fakepage;
			} else {
				return NULL;
			}
		}

		virtual Schema* schema() { return sch; }

		virtual void reset() { firsttime = true; }

		virtual vector<PageCursor*> split(int nthreads)
		{
			throw NotYetImplemented();
		}
	
	private:
		FakePage fakepage;
		Schema* sch;
		bool firsttime;

};
#endif
