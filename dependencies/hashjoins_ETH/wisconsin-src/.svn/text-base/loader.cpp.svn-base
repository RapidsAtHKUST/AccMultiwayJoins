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

#include <fstream>
using std::ifstream;

#include "bzlib.h"

#include "loader.h"
#include "parser.h"

Loader::Loader(const char separator)
	: sep(separator) 
{
}

/**
 * Loads \a filename, parses it and outputs it at the WriteTable \a output.
 * Automatically checks if input is a BZ2 file and uncompresses on the fly,
 * using libbz2.
 */
void Loader::load(const string& filename, WriteTable& output)
{
	const char* parseresult[MAX_COL];
	int parseresultcount;

	Parser parser(sep);

	if (!isBz2(filename))
	{
		char linebuf[MAX_LINE];
		ifstream f(filename.c_str(), ifstream::in);

		if (!f)
			throw FileNotFoundException();

		f.getline(linebuf, MAX_LINE);
		while(f) {
			parseresultcount = parser.parseLine(linebuf, parseresult);
			output.append(parseresult, parseresultcount);
			f.getline(linebuf, MAX_LINE);
		}
		f.close();
	} 
	else 
	{
		FILE*   f;
		BZFILE* b;
		int     bzerror;
		int     nBuf;
		int     nWritten;

		int 	unused = 0;

		char* 	decbuf;
		const int decbufsize = 1024*1024;
		decbuf = new char[decbufsize];

		f = fopen (filename.c_str(), "rb");
		if (!f) {
			throw LoadBZ2Exception();
		}
		b = BZ2_bzReadOpen (&bzerror, f, 0, 0, NULL, 0);
		if (bzerror != BZ_OK) {
			BZ2_bzReadClose (&bzerror, b);
			throw LoadBZ2Exception();
		}

		bzerror = BZ_OK;

		while (bzerror == BZ_OK) {
			nBuf = BZ2_bzRead ( &bzerror, b, decbuf+unused, decbufsize-unused);

			// Check for error during decompression.
			//
			if (bzerror != BZ_OK && bzerror != BZ_STREAM_END) {
				BZ2_bzReadClose ( &bzerror, b );
				throw LoadBZ2Exception();
			}

			// No error; parse decompressed buffer.
			//
			char* p = decbuf;
			char* usablep = decbuf;
			assert(nBuf + unused <= decbufsize);
			while (p < decbuf + nBuf + unused) {
				p = readFullLine(usablep, decbuf, nBuf + unused);

				// Check if buffer didn't have a full line. If yes, break.
				//
				if (usablep == p) {
					break;
				}

				// We have a full line at usablep. Parse it.
				//
				parseresultcount = parser.parseLine(usablep, parseresult);
				output.append(parseresult, parseresultcount);

				usablep = p;
			}

			// Copy leftovers to start of buffer and call bzRead so
			// that it writes output immediately after existing data. 
			unused = decbuf + nBuf + unused - p;
			assert(unused < p - decbuf);  // no overlapped memcpy
			memcpy(decbuf, p, unused);
		}

		assert (bzerror == BZ_STREAM_END);
		BZ2_bzReadClose ( &bzerror, b );

		delete[] decbuf;
	}
}

/** 
 * Reads one full line from buffer. 
 *
 * If buffer has at least one line, returns the start of the next line and 
 * \a cur points to a null-terminated string.
 *
 * Otherwise, returns \a cur.
 */
char* Loader::readFullLine(char* cur, const char* bufstart, const int buflen)
{
	char* oldcur = cur;

	while(cur >= bufstart && cur < (bufstart + buflen))
	{
		if ((*cur) == '\n')
		{
			*cur = 0;
			return ++cur;
		}

		cur++;
	}

	return oldcur;
}

/**
 * Returns true if input file is BZ2-compressed. 
 * Check is done by looking at the three first bytes of the file: if "BZh",
 * then it is assumed to be a valid file to be parsed by libbz2.
 */
bool Loader::isBz2(const string& filename)
{
	char header[4];

	ifstream f(filename.c_str(), ifstream::in);
	f.get(header, 4);
	f.close();

	return string(header) == "BZh";
}
