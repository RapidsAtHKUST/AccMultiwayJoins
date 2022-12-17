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

#ifndef __MYHASHFUNCTION__
#define __MYHASHFUNCTION__

#include <vector>
#include <libconfig.h++>

using std::vector;

class HashFunction {
	public:
		/**
		 * Creates a new hashing object, rounding the number of buckets
		 * \a k to the next power of two. Formally \a k will become
		 * \f$ k = 2^{\lceil{\log_2{k}}\rceil} \f$
		 * @param min Minimum hash value
		 * @param max Maximum hash value
		 * @param k Number of buckets. Value will be rounded to the next
		 * higher power of two.
		 */
		HashFunction(int min, int max, unsigned int k);
		
		/**
		 * Returns the bucket number \f$n\in[0,k)\f$ for this \a value.
		 * @param value Value to hash. Must be within bounds.
		 * @return Bucket number.
		 */
		virtual unsigned int hash(long long value) = 0;

		virtual unsigned int buckets();

	protected:
		int _min, _max;
		unsigned int _k;	/**< \f$ \_k=log_2(k) \f$, where \f$k\f$ is number of buckets */

};

class RangePartitionHashFunction : public HashFunction {
	public:
		RangePartitionHashFunction(int min, int max, unsigned int k)
			: HashFunction(min, max, k) { }

		inline unsigned int hash(long long value) {
			unsigned long long val = (value-_min);
			val <<= _k;
			return val / (_max-_min+1);
		}
};

class ModuloHashFunction : public HashFunction {
	public:
		/**
		 * Skipbits parameter defines number of least-significant bits which
		 * will be discarded before computing the hash.
		 */
		ModuloHashFunction(int min, int max, unsigned int k, unsigned int skipbits)
			: HashFunction(min, max, k) { 
			_skipbits = skipbits;
			_k = ((1 << _k) - 1) << _skipbits;	// _k is used as the modulo mask
		}

		/** Return h(x) = x mod k. */
		inline unsigned int hash(long long value) {
			return ((value-_min) & _k) >> _skipbits;
		}

		inline unsigned int buckets() {
			return (_k >> _skipbits) + 1;
		}

		/** Generate set of hash functions to be used for multiple passes. */
		vector<HashFunction*> generate(unsigned int passes);

	private:
		unsigned int _skipbits;
};

class MagicHashFunction : public ModuloHashFunction {
	public:
		MagicHashFunction(int min, int max, unsigned int k)
			: ModuloHashFunction(min, max, k, 0) { }

		/** 
		 * The domain of o_orderkey in TPC-H has the weird property that it 
		 * sets least signifiant bits 3 and 4 to zero. This hash function 
		 * has been designed to work around this peculiarity. */
		inline unsigned int hash(long long value) {
			return ( ( (value >> 2) & ~7L ) | (value & 7) ) & _k;
		}
};

class HashFactory {
	public:
		static HashFunction* createHashFunction(const libconfig::Setting& node);
};

#endif
