#!/bin/bash

THREADS="08" # 12 16 24 64"
EXPO="4 6 8 9 10 11 12 13 15 17"
PASSES="1 2"
ALGORTHMS="parallel independent"

for t in $THREADS; do
	mkdir -p $t
	for a in $ALGORTHMS; do
		for e in $EXPO; do
			m4  -DALGORITHM=$a	\
				-DBUCKETS=$((1<<$e))	\
				-DSKIPBITS=$((24-$e-1))	\
				-DPAGESIZE=$((1<<$((24-$e+4))))	\
				-DTHREADS=$t	\
				template.m4		> $t/`printf %06d $((1<<$e))`_${a}.conf
			for passes in $PASSES; do
				m4  -DALGORITHM=radix	\
					-DBUCKETS=$((1<<$e))	\
					-DSKIPBITS=$((24-$e-1))	\
					-DPAGESIZE=$((1<<$((24-$e+4))))	\
					-DTHREADS=$t	\
					-DNUMPASSES=$passes	\
					template.radix.m4	> $t/`printf %06d $((1<<$e))`_radix${passes}.conf
				m4  -DALGORITHM=radix	\
					-DBUCKETS=$((1<<$e))	\
					-DSKIPBITS=$((24-$e-1))	\
					-DPAGESIZE=$((1<<$((24-$e+4))))	\
					-DTHREADS=$t	\
					-DNUMPASSES=$passes	\
					template.radixsteal.m4	> $t/`printf %06d $((1<<$e))`_radix${passes}steal.conf
			done
		done
	done
done
