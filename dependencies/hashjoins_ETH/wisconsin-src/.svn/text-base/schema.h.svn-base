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

#ifndef __MYSCHEMA__
#define __MYSCHEMA__

#include <vector>
#include <utility>
#include <string>
#include <cstring>
#include <cassert>
#include "libconfig.h++"

#include "exceptions.h"

using namespace std;

enum ColumnType {
	CT_INTEGER,	/**< Integer is sizeof(int), either 32 or 64 bits. */
	CT_LONG,	/**< Long is sizeof(long long), exactly 64 bits. */
	CT_DECIMAL,	/**< Decimal is sizeof(double). */
	CT_CHAR,	/**< Char is variable length, no zero padding is done. */
	CT_POINTER	/**< Pointer is sizeof(void*). */
};

typedef std::pair<ColumnType, unsigned int> ColumnSpec;

#include "comparator.h"

class Schema {
	public:
		Schema() : totalsize(0) { }
		/**
		 * Add new column at the end of this schema.
		 * @param ct Column type.
		 * @param size Max characters (valid for CHAR(x) type only).
		 */
		void add(ColumnType ct, unsigned int size = 0);

		void add(ColumnSpec desc);

		/** 
		 * Returns columnt type at position \a pos.
		 * @param pos Position in schema.
		 * @return Column type.
		 */
		ColumnType getColumnType(unsigned int pos);

		/** 
		 * Returns a pair of (type, length).
		 * @param pos Position in schema.
		 * @return pair of (type, length).
		 */
		ColumnSpec get(unsigned int pos);

		/**
		 * Get total number of columns.
		 * @return Total number of columns.
		 */
		unsigned int columns();

		/**
		 * Get a tuple size, in bytes, for this schema.
		 * @return Tuple size in bytes.
		 */
		unsigned int getTupleSize();

		/**
		 * Return a string representation of the data in column \a pos.
		 * @param data Tuple to work on.
		 * @param pos Position of column to parse.
		 * @throw IllegalConversionException.
		 */
		const char* asString(void* data, unsigned int pos);

		/**
		 * Calculate the position of data item \a pos inside tuple \a data.
		 * @param data Tuple to work on.
		 * @param pos Position in tuple.
		 * @return Pointer to data of column \a pos.
		 */
		void* calcOffset(void* data, unsigned int pos);

		/**
		 * Return an integer representation of the data in column \a pos.
		 * @param data Tuple to work on.
		 * @param pos Position of column to parse.
		 * @throw IllegalConversionException.
		 */
		int asInt(void* data, unsigned int pos);

		/**
		 * Return an integer representation of the data in column \a pos.
		 * @param data Tuple to work on.
		 * @param pos Position of column to parse.
		 * @throw IllegalConversionException.
		 */
		long long asLong(void* data, unsigned int pos);

		/**
		 * Return a double representation of the data in column \a pos.
		 * @param data Tuple to work on.
		 * @param pos Position of column to parse.
		 * @throw IllegalConversionException.
		 */
		double asDouble(void* data, unsigned int pos);

		/**
		 * Return the data in column \a pos as a pointer.
		 * @param data Tuple to work on.
		 * @param pos Position of column to parse.
		 * @throw IllegalConversionException.
		 */
		void* asPointer(void* data, unsigned int pos);

		/**
		 * Write the input \a data to the tuple pointed by \a dest.
		 * @pre Caller must have preallocated enough memory at \a dest.
		 * @param dest Destination tuple to write.
		 * @param pos Position in the tuple.
		 * @param data Data to write.
		 */
		void writeData(void* dest, unsigned int pos, const void* const data);

		/**
		 * Parse the input vector \a input, convert each element to
		 * the appropriate type (according to the schema) and 
		 * call @ref writeData on that.
		 * @pre Caller must have preallocated enough memory at \a dest.
		 * @param dest Destination tuple to write.
		 * @param input Vector of string inputs.
		 */
		void parseTuple(void* dest, const std::vector<std::string>& input);
		void parseTuple(void* dest, const char** input);

    /** 
     * Parse the given relation allocated sequentially in the 
     * memory pointed by input
     */
    void parseTuple(void* dest, const char* input);

		/**
		 * Returns a string representation of each column in the tuple.
		 * @param data Tuple to parse.
		 * @return Vector of strings.
		 */
		std::vector<std::string> outputTuple(void* data);

		/**
		 * Copy a tuple from \a src to \a dest. The number of bytes
		 * copied is equal to \ref getTupleSize.
		 * @pre Caller must have preallocated enough memory at \a dest.
		 * @param dest Destination address.
		 * @param src Source address.
		 */
		void copyTuple(void* dest, const void* const src);

		/** 
		 * Pretty-print the tuple, using character \a sep as separator.
		 */
		string prettyprint(void* tuple, char sep);

		/**
		 * Create Schema object from configuration node.
		 */
		static Schema create(const libconfig::Setting& line);

		/**
		 * Create Comparator object.
		 */
		static Comparator createComparator(Schema& lhs, unsigned int lpos, 
										Schema& rhs, unsigned int rpos);

		static Comparator createComparator(Schema& lhs, unsigned int lpos, 
										ColumnSpec& rhs);

		static Comparator createComparator(ColumnSpec& lhs, 
										Schema& rhs, unsigned int rpos);

	private:
		vector<ColumnType> vct;
		vector<int> voffset;
		int totalsize;
};

inline unsigned int Schema::getTupleSize() {
	return totalsize;
}


inline void Schema::writeData(void* dest, unsigned int pos, const void* const data) {
#ifdef DEBUG2
	assert(pos<columns());
	assert(sizeof(void*)==sizeof(long));
#endif
	void* d = reinterpret_cast<char*>(dest)+voffset[pos];

	// copy
	switch (vct[pos]) {
		case CT_POINTER: {
			const long* val = reinterpret_cast<const long*>(data);
			*reinterpret_cast<long*>(d) = *val;
			break;
		}
		case CT_INTEGER: {
			const int* val = reinterpret_cast<const int*>(data);
			*reinterpret_cast<int*>(d) = *val;
			break;
		}
		case CT_LONG: {
			const long long* val = reinterpret_cast<const long long*>(data);
			*reinterpret_cast<long long*>(d) = *val;
			break;
		}
		case CT_DECIMAL: {
			const double* val2 = reinterpret_cast<const double*>(data);
			*reinterpret_cast<double*>(d) = *val2;
			break;
		}
		case CT_CHAR: {
			const char* p = reinterpret_cast<const char*>(data);
			char* t = reinterpret_cast<char*>(d);
			
			//write p in t
			while (*(t++) = *(p++))
				;
			break;
		}
	}

}

inline void Schema::copyTuple(void* dest, const void* const src) {
	memcpy(dest, src, totalsize);
}

#endif
