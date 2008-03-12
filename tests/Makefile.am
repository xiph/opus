INCLUDES = -I$(top_srcdir)/libcelt
METASOURCES = AUTO

TESTS = type-test ectest cwrs32-test cwrs64-test real-fft-test dft-test laplace-test mdct-test rotation-test

noinst_PROGRAMS = type-test ectest cwrs32-test cwrs64-test real-fft-test dft-test laplace-test mdct-test rotation-test

type_test_SOURCES = type-test.c
ectest_SOURCES = ectest.c
cwrs32_test_SOURCES = cwrs32-test.c
cwrs64_test_SOURCES = cwrs64-test.c
real_fft_test_SOURCES = real-fft-test.c
dft_test_SOURCES = dft-test.c
laplace_test_SOURCES = laplace-test.c
mdct_test_SOURCES = mdct-test.c
rotation_test_SOURCES = rotation-test.c

LDFLAGS = -static
LDADD = $(top_builddir)/libcelt/libcelt.la
