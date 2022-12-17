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

#ifndef __MYPROTOTYPE__
#define __MYPROTOTYPE__

#include <vector>
#include <string>

/**
 * Parses text input into binary data.
 */
class Parser {
	public:
		Parser (char separator) : _sep(separator) { }

		/** 
		 * Returns the offsets of all data contained in this \a line in 
		 * variable \a result. Will read until it encounters a \0 in \a line,
		 * leaving code vulnerable to overflow attacks.
		 * @return The number of valid entries in \a result.
		 */
		int parseLine(char* line, const char** result);
	private:
		char _sep;
};

#endif
