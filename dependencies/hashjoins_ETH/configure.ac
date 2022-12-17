#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

#AC_PREREQ([2.68])
AC_INIT([multicore-hashjoins], [0.1], [cagri.balkesen@inf.ethz.ch])
AM_INIT_AUTOMAKE(multicore-hashjoins, 0.1)

AC_CONFIG_SRCDIR([src/npj_types.h])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CC
AC_PROG_CXX
AC_PROG_MAKE_SET

# Checks for header files.
AC_CHECK_HEADERS([stdint.h stdlib.h string.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_C_RESTRICT
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T

# Checks for libraries.
AC_CHECK_LIB([m], [pow], [],[
         echo "math library is required for this program"
         exit -1])

AC_CHECK_LIB([pthread], [pthread_mutex_init], ,
             AC_MSG_ERROR(pthread library not found))

# Checks for library functions.
AC_FUNC_ERROR_AT_LINE
AC_FUNC_MALLOC
AC_CHECK_FUNCS([gettimeofday memset pow])
AC_CHECK_FUNCS([pthread_attr_setaffinity_np pthread_barrier_wait])

# Check for conditional compilation options
# Use 8B keys?
AC_ARG_ENABLE(key8B,
   [  --enable-key8B  use 8B keys and values making tuples 16B  [default=no]],
   [enable_key8B="$enableval"], 
   [enable_key8B="no"])

AM_CONDITIONAL([KEY_8B], [test "$enable_key8B" = "yes"])

# Do performance counter monitoring?
AC_ARG_ENABLE(perfcounters,
   [  --enable-perfcounters  enable performance counter monitoring with Intel PCM  [default=no]],
   [enable_perfcounters="$enableval" CC="g++"], 
   [enable_perfcounters="no"])

AM_CONDITIONAL([PERF_COUNTERS], [test "$enable_perfcounters" = "yes"])

#if test "$enable_perfcounters" = "yes"; then
#     AC_CHECK_LIB([perf], [printf], [], [
#                     echo "Intel PCM library is not found! Build lib/ and add to LD_LIBRARY_PATH."
#		     exit -1])
#fi

# Output debug messages?
AC_ARG_ENABLE(debug,
   [  --enable-debug  enable debug messages on commandline  [default=no]],
   [enable_debug="$enableval"], 
   [enable_debug="no"])

AM_CONDITIONAL([DEBUG], [test "$enable_debug" = "yes"])

# Padding of NPO buckets to cache line size?
AC_ARG_ENABLE(paddedbucket,
   [  --enable-paddedbucket  enable padding of buckets to cache line size in NPO  [default=no]],
   [enable_paddedbucket="$enableval"], 
   [enable_paddedbucket="no"])

AM_CONDITIONAL([PADDEDBUCKET], [test "$enable_paddedbucket" = "yes"])

# Disable timing?
AC_ARG_ENABLE(timing,
   [  --enable-timing  enable execution timing  [default=yes]],
   [enable_timing="$enableval"], 
   [enable_timing="yes"])

AM_CONDITIONAL([NOTIMING], [test "$enable_timing" = "no"])

# Disable timing?
AC_ARG_ENABLE(syncstats,
   [  --enable-syncstats  enable synchronization timing stats  [default=no]],
   [enable_syncstats="$enableval"], 
   [enable_syncstats="no"])

AM_CONDITIONAL([SYNCSTATS], [test "$enable_syncstats" = "yes"])

# Enable fine-granular task decomposition based skew handling in Radix joins
AC_ARG_ENABLE(skewhandling,
   [  --enable-skewhandling  enable fine-granular task decomposition based skew handling in radix?  [default=no]],
   [enable_skewhandling="$enableval"], 
   [enable_skewhandling="no"])

AM_CONDITIONAL([SKEW_HANDLING], [test "$enable_skewhandling" = "yes"])

# Enable prefetching in No Partitioning Join?
AC_ARG_ENABLE(prefetch-npj,
   [  --enable-prefetch-npj  enable prefetching in No Partitioning Join?  [default=no]],
   [enable_prefetchnpj="$enableval"], 
   [enable_prefetchnpj="no"])

AM_CONDITIONAL([PREFETCH_NPJ], [test "$enable_prefetchnpj" = "yes"])

# Experimental, works but not tested extensively.
# Enable software write-combining optimization in partitioning
AC_ARG_ENABLE(swwc-part,
   [  --enable-swwc-part  enable software write-combining optimization in partitioning (Experimental, not tested extensively)?  [default=no]],
   [enable_swwcpart="$enableval"], 
   [enable_swwcpart="no"])

AM_CONDITIONAL([USE_SWWC_OPTIMIZED_PART], [test "$enable_swwcpart" = "yes"])

AC_CONFIG_FILES([Makefile src/Makefile])
AC_OUTPUT
