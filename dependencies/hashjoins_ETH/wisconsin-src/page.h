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

#ifndef __MYBUFFER__
#define __MYBUFFER__

class Buffer 
{
	public:
		friend unsigned int pageCopy(void*, const Buffer*);

		/**
		 * Creates a Buffer to operate on the block starting at \a *data.
		 * Class does not own the block.
		 * @param data Start of block.
		 * @param size Size of block.
		 * @param free Free section of the block. If NULL, it is assumed that
		 * the entire block is full with useful data and no further writes are
		 * allowed.
		 */
		Buffer(void* data, unsigned int size, void* free);

		/**
		 * Allocates an empty buffer of specified size that is owned by this
		 * instance.
		 * \param size Buffer size in bytes.
		 */
		Buffer(unsigned long size);
		~Buffer();

		/**
		 * Returns true if \a len bytes can be stored in this page.
		 * Does not guarantee that a subsequent allocation will succeed, as
		 * some other thread might beat us to allocate.
		 * @param len Bytes to store.
		 * @return True if \a len bytes can be stored in this page.
		 */
		inline bool canStore(unsigned int len);

		/**
		 * Returns a memory location and moves the \a free pointer
		 * forward. NOT THREAD-SAFE!
		 * @return Memory location good for writing up to \a len bytes, else
		 * NULL.
		 */
		inline void* allocate(unsigned int len);

		/**
		 * Returns a memory location and atomically moves the \a free pointer
		 * forward. The memory location is good for writing up to \a len
		 * bytes. 
		 * @return Memory location good for writing up to \a len bytes, else
		 * NULL.
		 */
		inline void* atomicAllocate(unsigned int len);

		/**
		 * Returns whether this address is valid for reading \a len bytes.
		 * That is, checks if \a loc points anywhere between \a data and \a
		 * data + \a maxsize - \a len.
		 * @return True if address points into this page and \a len bytes can
		 * be read off from it.
		 */
		inline bool isValidAddress(void* loc, unsigned int len);

		/**
		 * Clears contents of page.
		 */
		inline void clear() { free = data; }

		// TODO: isOwner(): returns true if owner
		// TODO: copy method: returns non-owner class
		// TODO: transferOwnership(from, to): if two data fields equal, transfers ownership
		
		/**
		 * Returns the capacity of this buffer.
		 */
		inline const unsigned int capacity() { return maxsize; }

		/**
		 * Returns the used space of this buffer.
		 */
		inline const unsigned int getUsedSpace();

	protected:
		/** Data segment of page. */
		void* data;

		unsigned long maxsize;
		
		/**
		 * True if class owns the memory at \a *data and is responsible for its
		 * deallocation, false if not.
		 */
		bool owner;

		/** 
		 * Marks the free section of data segment, therefore data <= free <
		 * data+maxsize.
		 */
		/* volatile */ void* free;
};


class TupleBuffer : public Buffer
{
	public:
		TupleBuffer(void* data, unsigned int size, void* free, unsigned int tuplesize);

		/**
		 * Creates a buffer of size \a size, holding tuples which are
		 * \a tuplesize bytes each.
		 * \param size Buffer size in bytes.
		 * \param tuplesize Size of tuples in bytes.
		 */
		TupleBuffer(unsigned long size, unsigned int tuplesize);
		~TupleBuffer() { }

		/**
		 * Returns true if a tuple can be stored in this page.
		 * @return True if a tuple can be stored in this page.
		 */
		inline bool canStoreTuple() 
		{ 
			return canStore(tuplesize); 
		}

		/**
		 * Returns the pointer to the start of the \a pos -th tuple,
		 * or NULL if this tuple doesn't exist in this page.
		 *
		 * Note: 
		 * MemMappedTable::close() assumes that data==getTupleOffset(0).
		 */
		inline void* getTupleOffset(unsigned int pos);

		/**
		 * Returns whether this address is valid, ie. points anywhere between 
		 * \a data and \a data + \a maxsize - \a tuplesize.
		 * @return True if address points into this page.
		 */
		inline bool isValidTupleAddress(void* loc);

		/**
		 * Returns a memory location and moves the \a free pointer
		 * forward. NOT THREAD-SAFE!
		 * @return Memory location good for writing a tuple, else NULL.
		 */
		inline void* allocateTuple();

		/**
		 * Returns a memory location and atomically moves the \a free pointer
		 * forward. The memory location is good for writing up to \a len
		 * bytes. 
		 * @return Memory location good for writing a tuple, else NULL.
		 */
		inline void* atomicAllocateTuple();

		class Iterator {
			friend class TupleBuffer;

			protected:
				Iterator(TupleBuffer* p) : tupleid(0), page(p) { }

			public:
				Iterator& operator= (Iterator& rhs) {
					page = rhs.page;
					tupleid = rhs.tupleid;
					return *this;
				}

				inline
				void place(TupleBuffer* p)
				{
					page = p;
					reset();
				}

				inline
				void* next()
				{
					return page->getTupleOffset(tupleid++);
				}

				inline
				void reset()
				{
					tupleid = 0;
				}

			private:
				int tupleid;
				TupleBuffer* page;
		};

		Iterator createIterator()
		{
			return Iterator(this);
		}

	protected:
		unsigned int tuplesize;

};


/**
 * A \a TupleBuffer with a "next" pointer.
 * @deprecated Only used by Table classes.
 */
class LinkedTupleBuffer : public TupleBuffer {
	public:

		LinkedTupleBuffer(void* data, unsigned int size, void* free, 
				unsigned int tuplesize)
			: TupleBuffer(data, size, free, tuplesize), next(0)
		{ }

		/**
		 * Creates a bucket of size \a size, holding tuples which are
		 * \a tuplesize bytes each.
		 * \param size Bucket size in bytes.
		 * \param tuplesize Size of tuples in bytes.
		 */
		LinkedTupleBuffer(unsigned int size, unsigned int tuplesize) 
			: TupleBuffer(size, tuplesize), next(0) { }

		/** 
		 * Returns a pointer to next bucket.
		 * @return Next bucket, or NULL if it doesn't exist.
		 */
		LinkedTupleBuffer* getNext()
		{
			return next;
		}

		/**
		 * Sets the pointer to the next bucket.
		 * @param bucket Pointer to next bucket.
		 */
		void setNext(LinkedTupleBuffer* const bucket)
		{
			next = bucket;
		}	

	private:
		LinkedTupleBuffer* next;
};

typedef TupleBuffer Page;
typedef LinkedTupleBuffer Bucket;

class FakePage : public Page {
	public:
		FakePage(unsigned int size, unsigned int tuplesz, void* existingdata) 
			: Page(size, tuplesz)
		{
			delete[] reinterpret_cast<char*>(data);
			data = existingdata;
			// invalidate free pointer, making page "full".
			free = reinterpret_cast<char*>(data) + maxsize;
		}

		FakePage()
			: Page(0, 0)
		{ 
			delete[] reinterpret_cast<char*>(data);
			data = 0;	// NULL
			free = 0;	// NULL
		}

		void place(unsigned int size, unsigned int tuplesz, void* existingdata) 
		{
			maxsize = size;
			tuplesize = tuplesz;
			data = existingdata;
			// invalidate free pointer, making page "full".
			free = reinterpret_cast<char*>(data) + maxsize;
		}

		virtual ~FakePage() { }
};

#include "buffer.inl"

#endif
