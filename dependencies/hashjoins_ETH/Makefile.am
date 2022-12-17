AUTOMAKE_OPTIONS = foreign

LIBDIR = 

if PERF_COUNTERS
LIBDIR += lib/intel-pcm-1.7
endif

SUBDIRS = $(LIBDIR) src

EXTRA_DIST = lib			\
	     wisconsin-src		\
	     doc/Doxyfile		\
	     pcm.cfg			\
	     README			\
	     src/cpu-mapping.txt.README \
	     wisconsin-src/cpu-mapping.txt.README
