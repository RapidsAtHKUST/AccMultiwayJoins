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

#ifndef __MYPREFETCHING__
#define __MYPREFETCHING__

#if defined(__sparc__)

/* Only tested on UltraSPARC T1 */
#define __prefetch(addr) {                      \
        __asm__ __volatile__ (                  \
                        "prefetch %0, 20"       \
                        : : "m"(*(addr))        \
        );                                      \
}

#elif defined(__i386__) || defined(__x86_64__)

#warning Prefetching support for intel machines not yet written.
#define __prefetch(addr) ;

#else

#warning No supported architecture.

#endif

#endif
