#!/bin/bash

. system.inc

cd libconfig-1.2/
./configure --prefix $PWD/../dist CFLAGS="$SYSFLAGS -O3" CXXFLAGS="$SYSFLAGS -O3"
make -j install
make distclean
cd ..

cd bzip2-1.0.5/
make -j install PREFIX=$PWD/../dist CFLAGS="$SYSFLAGS -O3"
make clean
cd ..
