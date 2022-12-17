import random

# Generate 16x bigger probe side.
#
maxkeyval = 16*1024*1024;

for iteration in xrange(16):
	values = range(1, maxkeyval+1)
	random.shuffle(values)

	for i in xrange(len(values)):
		print str((iteration*maxkeyval) + (i+1)) + '|' + str(values[i])
