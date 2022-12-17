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

#ifndef __LOADER__
#define __LOADER__

#include <string>
using std::string;

#include "table.h"

class Loader {
	public:
		Loader(const char separator);

		void load(const string& filename, WriteTable& output);

	private:
		bool isBz2(const string& filename);

		char* readFullLine(char* cur, const char* bufstart, const int buflen);

		/** Column separating character. */
		const char sep;

		/** Maximum characters in line. */
		static const unsigned int MAX_LINE = 1024;

		/** Maximum columns in line. */
		static const unsigned int MAX_COL = 64;

};

#endif
