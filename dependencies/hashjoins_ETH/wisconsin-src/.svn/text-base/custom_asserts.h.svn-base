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

#ifndef __MYCUSTOMASSERTS__
#define __MYCUSTOMASSERTS__

#include <cassert>

inline void dbg2assert(int expression) 
{
#ifdef DEBUG2
	assert(expression);
#endif
}

inline void dbgassert(int expression) 
{
#ifdef DEBUG
	assert(expression);
#endif
}

inline void dbgassertimplies(int condition, int expression) 
{
#ifdef DEBUG
	if (condition)
		assert(expression);
#endif
}

#endif
