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

#include "schema.h"	// for ColumnType

#include <cstring>

bool MakesNoSense(void* lhs, void* rhs, int n)
{
	throw ComparisonException();
}

// Compare Int with {Int, Long, Double}
//

bool IntIntEqual(void* lhs, void* rhs, int n)
{
	return (*(int*)lhs) == (*(int*)rhs);
}

bool IntIntLess(void* lhs, void* rhs, int n)
{
	return (*(int*)lhs) < (*(int*)rhs);
}

bool IntLongEqual(void* lhs, void* rhs, int n)
{
	return ((long long)(*(int*)lhs)) == (*(long long*)rhs);
}

bool IntLongLess(void* lhs, void* rhs, int n)
{
	return (long long)(*(int*)lhs) < (*(long long*)rhs);
}

bool IntDoubleEqual(void* lhs, void* rhs, int n)
{
	return (double)(*(int*)lhs) == (*(double*)rhs);
}

bool IntDoubleLess(void* lhs, void* rhs, int n)
{
	return (double)(*(int*)lhs) < (*(double*)rhs);
}

// Compare Long with {Int, Long, Double}
//

bool LongIntEqual(void* lhs, void* rhs, int n)
{
	return (*(long long*)lhs) == (long long)(*(int*)rhs);
}

bool LongIntLess(void* lhs, void* rhs, int n)
{
	return (*(long long*)lhs) < (long long)(*(int*)rhs);
}

bool LongLongEqual(void* lhs, void* rhs, int n)
{
	return (*(long long*)lhs) == (*(long long*)rhs);
}

bool LongLongLess(void* lhs, void* rhs, int n)
{
	return (*(long long*)lhs) < (*(long long*)rhs);
}

bool LongDoubleEqual(void* lhs, void* rhs, int n)
{
	return (double)(*(long long*)lhs) == (*(double*)rhs);
}

bool LongDoubleLess(void* lhs, void* rhs, int n)
{
	return (double)(*(long long*)lhs) < (*(double*)rhs);
}

// Compare Double with {Int, Long, Double}
//

bool DoubleIntEqual(void* lhs, void* rhs, int n)
{
	return (*(double*)lhs) == (double)(*(int*)rhs);
}

bool DoubleIntLess(void* lhs, void* rhs, int n)
{
	return (*(double*)lhs) < (double)(*(int*)rhs);
}

bool DoubleLongEqual(void* lhs, void* rhs, int n)
{
	return (*(double*)lhs) == (double)(*(long long*)rhs);
}

bool DoubleLongLess(void* lhs, void* rhs, int n)
{
	return (*(double*)lhs) < (double)(*(long long*)rhs);
}

bool DoubleDoubleEqual(void* lhs, void* rhs, int n)
{
	return (*(double*)lhs) == (*(double*)rhs);
}

bool DoubleDoubleLess(void* lhs, void* rhs, int n)
{
	return (*(double*)lhs) < (*(double*)rhs);
}

// Char to char comparison
// FIXME Buggy behavior if comapring "AB" with "ABCD" and n==2. 
// Strings will be reported as equal, while they are not.
//

bool CharCharEqual(void* lhs, void* rhs, int n)
{
	return strncmp((char*)lhs, (char*)rhs, n) == 0;
}

bool CharCharLess(void* lhs, void* rhs, int n)
{
	return strncmp((char*)lhs, (char*)rhs, n) < 0;
}

// Pointer to pointer comparison
//

bool VoidVoidEqual(void* lhs, void* rhs, int n)
{
	return (*(void**)lhs) == (*(void**)rhs);
}

// Comparator class code.
//

void Comparator::init(ColumnSpec lct, unsigned int loff, 
					   ColumnSpec rct, unsigned int roff) 
{
	loffset = loff;
	roffset = roff;
	size = 0;

	switch(lct.first) {

		// First argument is int...
		//
		case CT_INTEGER:
			switch(rct.first) {
				case CT_INTEGER:
					eq = IntIntEqual;
					less = IntIntLess;
					break;
				case CT_LONG:
					eq = IntLongEqual;
					less = IntLongLess;
					break;
				case CT_DECIMAL:
					eq = IntDoubleEqual;
					less = IntDoubleLess;
					break;
				case CT_CHAR:
				case CT_POINTER:
				default:
					throw ComparisonException();
					break;
			}
			break;

		// First argument is long...
		//
		case CT_LONG:
			switch(rct.first) {
				case CT_INTEGER:
					eq = LongIntEqual;
					less = LongIntLess;
					break;
				case CT_LONG:
					eq = LongLongEqual;
					less = LongLongLess;
					break;
				case CT_DECIMAL:
					eq = LongDoubleEqual;
					less = LongDoubleLess;
					break;
				case CT_CHAR:
				case CT_POINTER:
				default:
					throw ComparisonException();
					break;
			}
			break;

		// First argument is double...
		//
		case CT_DECIMAL:
			switch(rct.first) {
				case CT_INTEGER:
					eq = DoubleIntEqual;
					less = DoubleIntLess;
					break;
				case CT_LONG:
					eq = DoubleLongEqual;
					less = DoubleLongLess;
					break;
				case CT_DECIMAL:
					eq = DoubleDoubleEqual;
					less = DoubleDoubleLess;
					break;
				case CT_CHAR:
				case CT_POINTER:
				default:
					throw ComparisonException();
					break;
			}
			break;

		// First argument is pointer...
		// Only equals comparisons with other pointers make sense.
		//
		case CT_POINTER:
			if (rct.first != CT_POINTER)
				throw ComparisonException();

			eq = VoidVoidEqual;
			less = MakesNoSense;
			break;

		// First argument is string...
		//
		case CT_CHAR:
			if (rct.first != CT_CHAR)
				throw ComparisonException();

			eq = CharCharEqual;
			less = CharCharLess;
			size = min(lct.second, rct.second);
			break;

		// Huh?
		//
		default:
			throw ComparisonException();
			break;
	}
}
