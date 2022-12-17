#!/usr/bin/python

import sys

fp_in = open(sys.argv[1])
fp_out = open(sys.argv[2], 'w')
line = fp_in.readline()

while line:
    l = line.split()
    fp_out.write("{} {}\n".format(int(l[1]), int(l[0])))
    line = fp_in.readline()


