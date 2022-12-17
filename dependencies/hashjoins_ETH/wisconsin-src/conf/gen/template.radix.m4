# vi:ts=2

path:	"../../datagen/";
bucksize:	1048576 ;

partitioner:
{
	build:
	{
		algorithm:	"ALGORITHM";
		pagesize:		PAGESIZE;
		attribute:	1;
		passes:			NUMPASSES;
	};

	probe:
	{
		algorithm:	"ALGORITHM";
		pagesize:		PAGESIZE;
		attribute:	1;
		passes:			NUMPASSES;
	};

	hash:
	{
		fn:				"modulo";
		range:		[1,16777216];
		buckets:	BUCKETS;
		skipbits:	SKIPBITS;
	};
};

build:
{
	file: 	"016M_build.tbl";
	schema: ("long", "long");
	jattr:	1;
	select:	(2);
	#it's possible to generate instead of loading
	generate: true;
	relation-size: 16777216; #128000000;
	alphabet-size: 16777216; #128000000;
	zipf-param: 0.00;
	seed: 12345;
};

probe:
{
	file:		"256M_probe.tbl";
	schema:	("long", "long");
	jattr:	1;
	select:	(2);
	#it's possible to generate instead of loading
	generate: true;
	relation-size: 268435456; #128000000;
	alphabet-size: 16777216; #128000000;
	zipf-param: 0.00;
	seed: 54321;
};

output:	"test.tbl";

hash:
{
	fn:				"modulo";
	range:		[1,16777216];
	buckets:	8388608;
};

algorithm:
{
	copydata:				"yes";
	partitionbuild:	"yes";
	buildpagesize:  32;
	partitionprobe:	"yes";
};

threads:		THREADS;
