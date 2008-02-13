INCLUDES = -I$(top_srcdir)/libcelt
METASOURCES = AUTO

TESTS = type-test ectest cwrs32-test cwrs64-test real-fft-test dft-test

noinst_PROGRAMS = type-test ectest cwrs32-test cwrs64-test real-fft-test dft-test

type_test_SOURCES = type-test.c
ectest_SOURCES = ectest.c
cwrs32_test_SOURCES = cwrs32-test.c
cwrs64_test_SOURCES = cwrs64-test.c
real_fft_test_SOURCES = real-fft-test.c
dft_test_SOURCES = dft-test.c

LDADD = $(top_builddir)/libcelt/libcelt.la
