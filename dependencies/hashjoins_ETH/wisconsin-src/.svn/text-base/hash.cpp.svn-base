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

#include <string>
using std::string;

#include "hash.h"
#include "exceptions.h"

#ifdef DEBUG
#include <cassert>
#endif 

/**
 * Returns the base 2 logarithm of the next higher power of two.
 */
unsigned int getlogarithm(unsigned int k) {
	--k;
	int m = 1 << (sizeof(int)*8-1);
	for (unsigned int i=0; i<sizeof(int)*8; ++i) {
		if ((m >> i) & k)
			return sizeof(int)*8-i;
	}
	return 0;
}

HashFunction::HashFunction(int min, int max, unsigned int k) : _min(min), _max(max) {
	if (k==0 || k==1)
		_k=1;
	else
		_k = getlogarithm(k);
}

unsigned int HashFunction::buckets() {
	return 1 << _k;
}

HashFunction* HashFactory::createHashFunction(const libconfig::Setting& node)
{
	HashFunction* hashfn;

	int k = node["buckets"];
	int min = node["range"][0];
	int max = node["range"][1];
	string hashfnname = node["fn"];

	if (hashfnname == "range") {
		hashfn = new RangePartitionHashFunction(min, max, k);
	} else if (hashfnname == "modulo") {
		unsigned int skipbits = 0;
		node.lookupValue("skipbits", skipbits);
		hashfn = new ModuloHashFunction(min, max, k, skipbits);
	} else if (hashfnname == "magic") {
		hashfn = new MagicHashFunction(min, max, k);
	} else {
		throw UnknownHashException();
	}

	return hashfn;
}

vector<HashFunction*> ModuloHashFunction::generate(unsigned int passes)
{
	vector<HashFunction*> ret;

	unsigned int totalbitsset = getlogarithm(buckets()-1);
	unsigned int bitsperpass = totalbitsset / passes;

	for (int i=0; i<passes-1; ++i) {
		ret.push_back(new ModuloHashFunction(_min, _max, 
					(1 << bitsperpass), 
					_skipbits + totalbitsset - ((i+1)*bitsperpass)));
	}

	bitsperpass = totalbitsset - ((passes-1) * bitsperpass);
	ret.push_back(new ModuloHashFunction(_min, _max, 
				(1 << bitsperpass), 
				_skipbits));

#ifdef DEBUG
	unsigned int kfinal = dynamic_cast<ModuloHashFunction*>(ret[0])->_k;
	for (int i=1; i<ret.size(); ++i) {
		kfinal |= dynamic_cast<ModuloHashFunction*>(ret[i])->_k;
		assert( (dynamic_cast<ModuloHashFunction*>(ret[i])->_k 
					& dynamic_cast<ModuloHashFunction*>(ret[i-1])->_k) 
				== 0 );
	}
	assert(kfinal==this->_k);
#endif
	
	return ret;
}
