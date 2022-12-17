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

#include "parser.h"
using namespace std;

int Parser::parseLine(char* line, const char** result) {
	char* s=line; /**< Points to beginning of token. */
	char* p=line; /**< Points to end of token. */
	int ret = 0;

	while (*s) {
		while (*p!=_sep && *p)	// while not separator or end-of-line
			p++;
		if (s!=p)	// eats null fields
			result[ret++]=s;
		if (*p) {	// stopped because of separator
			(*p)=0;
			s=++p;
		} else {	// stopped because of end of string
			break;	// normal loop exit point
		}
	}
	return ret;
}
