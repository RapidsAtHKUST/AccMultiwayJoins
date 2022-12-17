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

#include <sstream>
#include <cassert>
#include <cstdlib>
#include "schema.h"
#include <iomanip>
#include <algorithm>

using namespace std;

void Schema::add(ColumnSpec desc) {
	add(desc.first, desc.second);
}

void Schema::add(ColumnType ct, unsigned int size) {
	vct.push_back(ct);
	voffset.push_back(totalsize);

	int s=-1;
	switch (ct) {
		case CT_INTEGER:
			s = sizeof(int);
			break;
		case CT_LONG:
			s = sizeof(long long);
			break;
		case CT_DECIMAL:
			s = sizeof(double);
			break;
		case CT_CHAR:
			s = size;
			break;
		case CT_POINTER:
			s = sizeof(void*);
			break;
	}
	assert(s!=-1);
	totalsize+=s;
}

unsigned int Schema::columns() {
#ifdef DEBUG2
	assert(vct.size()==voffset.size());
#endif
	return vct.size();
}

ColumnType Schema::getColumnType(unsigned int pos) {
#ifdef DEBUG2
	assert(pos<columns());
#endif
	return vct[pos];
}

ColumnSpec Schema::get(unsigned int pos) {
#ifdef DEBUG2
	assert(pos<columns());
#endif
	unsigned int val1 = voffset[pos];
	unsigned int val2 = (pos != columns()-1) ? voffset[pos+1] : totalsize;
	return make_pair(vct[pos], val2-val1);
}

const char* Schema::asString(void* data, unsigned int pos) {
#ifdef DEBUG2
	assert(pos<columns());
	if (vct[pos]!=CT_CHAR)
		throw IllegalConversionException();
#endif
	char* d = reinterpret_cast<char*> (data);
	return reinterpret_cast<const char*> (d+voffset[pos]);
}

long long Schema::asLong(void* data, unsigned int pos) {
#ifdef DEBUG2
	assert(pos<columns());
	if (vct[pos]!=CT_LONG)
		throw IllegalConversionException();
#endif
	char* d = reinterpret_cast<char*> (data);
	return *reinterpret_cast<long long*> (d+voffset[pos]);
}

int Schema::asInt(void* data, unsigned int pos) {
#ifdef DEBUG2
	assert(pos<columns());
	if (vct[pos]!=CT_INTEGER)
		throw IllegalConversionException();
#endif
	char* d = reinterpret_cast<char*> (data);
	return *reinterpret_cast<int*> (d+voffset[pos]);
}

double Schema::asDouble(void* data, unsigned int pos) {
#ifdef DEBUG2
	assert(pos<columns());
	if (vct[pos]!=CT_DECIMAL)
		throw IllegalConversionException();
#endif
	char* d = reinterpret_cast<char*> (data);
	return *reinterpret_cast<double*> (d+voffset[pos]);
}

void* Schema::asPointer(void* data, unsigned int pos) {
#ifdef DEBUG2
	assert(pos<columns());
	if (vct[pos]!=CT_POINTER)
		throw IllegalConversionException();
	assert(sizeof(void*)==sizeof(long));
#endif
	char* d = reinterpret_cast<char*> (data);
	return reinterpret_cast<void*>(*reinterpret_cast<long*> (d+voffset[pos]));
}

void* Schema::calcOffset(void* data, unsigned int pos) {
#ifdef DEBUG2
	assert(pos<columns());
#endif
	return reinterpret_cast<char*>(data)+voffset[pos];
}

void Schema::parseTuple(void* dest, const vector<string>& input) {
#ifdef DEBUG2
	assert(input.size()>=columns());
#endif
	const char** data = new const char*[columns()];
	for (unsigned int i=0; i<columns(); ++i) {
		data[i] = input[i].c_str();
	}
	parseTuple(dest, data);
	delete[] data;
}

void Schema::parseTuple(void* dest, const char** input) {
	for (unsigned int i=0; i<columns(); ++i) {
		switch (vct[i]) {
			case CT_INTEGER: 
				int val;
				val = atoi(input[i]);
				writeData(dest, i, &val);
				break;
			case CT_LONG: 
				long long val3;
				val3 = atoll(input[i]);
				writeData(dest, i, &val3);
				break;
			case CT_DECIMAL:
				double val2;
				val2 = atof(input[i]);
				writeData(dest, i, &val2);
				break;
			case CT_CHAR:
				writeData(dest, i, input[i]);
				break;
			case CT_POINTER:
				throw IllegalConversionException();
				break;
		}
	}
}

void Schema::parseTuple(void* dest, const char* input) {
    char * tmp = (char*) input;
    unsigned int offset = 0;
	for (unsigned int i=0; i<columns(); ++i) {
		switch (vct[i]) {
			case CT_INTEGER: 
				int val;
				val = *((int*)tmp);
				writeData(dest, i, &val);
                tmp += sizeof(int);
				break;
			case CT_LONG: 
				long long val3;
				val3 = *((long long*)tmp);
				writeData(dest, i, &val3);
                tmp += sizeof(long long);
				break;
			case CT_DECIMAL:
				double val2;
				val2 = *((double*)tmp);
				writeData(dest, i, &val2);
                tmp += sizeof(double);
				break;
			case CT_CHAR:
				writeData(dest, i, tmp);
                tmp += sizeof(char);
				break;
			case CT_POINTER:
				throw IllegalConversionException();
				break;
		}
	}
}

vector<string> Schema::outputTuple(void* data) {
	vector<string> ret;
	for (unsigned int i=0; i<columns(); ++i) {
		ostringstream oss;
		switch (vct[i]) {
			case CT_INTEGER: 
				oss << asInt(data, i);
				break;
			case CT_LONG: 
				oss << asLong(data, i);
				break;
			case CT_DECIMAL:
				// every decimal in TPC-H has 2 digits of precision
				oss << setiosflags(ios::fixed) << setprecision(2) << asDouble(data, i);
				break;
			case CT_CHAR:
				oss << asString(data, i);
				break;
			case CT_POINTER:
				throw IllegalConversionException();
				break;
		}
		ret.push_back(oss.str());
	}
	return ret;
}

string Schema::prettyprint(void* tuple, char sep)
{
	string ret;
	const vector<string>& tokens = outputTuple(tuple);
	for (int i=0; i<tokens.size()-1; ++i)
		ret += tokens[i] + sep;
	if (tokens.size() > 1)
		ret += tokens[tokens.size()-1];
	return ret;
}

Schema Schema::create(const libconfig::Setting& line) {
	Schema ret;
	for (int i=0; i<line.getLength(); ++i) {
		string val = line[i];
		transform(val.begin(), val.end(), val.begin(), ::tolower);
		if (val.find("int")==0) {
			ret.add(CT_INTEGER);
		} else if (val.find("long")==0) {
			ret.add(CT_LONG);
		} else if (val.find("char")==0) {
			// char, check for integer
			string::size_type c = val.find("(");
			if (c==string::npos)
				throw IllegalSchemaDeclarationException();
			istringstream iss(val.substr(++c));
			int len;
			iss >> len;
			ret.add(CT_CHAR, len+1);	// compensating for \0
		} else if (val.find("dec")==0) {
			ret.add(CT_DECIMAL);
		} else {
			throw IllegalSchemaDeclarationException();
		}
	}
	return ret;
}

Comparator Schema::createComparator(Schema& lhs, unsigned int lpos, Schema& rhs, unsigned int rpos)
{
	Comparator c;
	c.init(lhs.get(lpos), lhs.voffset[lpos], 
			rhs.get(rpos), rhs.voffset[rpos]);
	return c;
}

Comparator Schema::createComparator(Schema& lhs, unsigned int lpos, ColumnSpec& rhs)
{
	Comparator c;
	c.init(lhs.get(lpos), lhs.voffset[lpos], 
			rhs, 0);
	return c;
}

Comparator Schema::createComparator(ColumnSpec& lhs, Schema& rhs, unsigned int rpos)
{
	Comparator c;
	c.init(lhs, 0, 
			rhs.get(rpos), rhs.voffset[rpos]);
	return c;
}
