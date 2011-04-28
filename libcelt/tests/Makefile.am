INCLUDES = -I$(top_srcdir)/libcelt
METASOURCES = AUTO

TESTS = type-test ectest cwrs32-test dft-test laplace-test mdct-test mathops-test tandem-test

noinst_PROGRAMS = type-test ectest cwrs32-test dft-test laplace-test mdct-test mathops-test tandem-test

type_test_SOURCES = type-test.c
ectest_SOURCES = ectest.c
cwrs32_test_SOURCES = cwrs32-test.c
dft_test_SOURCES = dft-test.c
laplace_test_SOURCES = laplace-test.c
mdct_test_SOURCES = mdct-test.c
#rotation_test_SOURCES = rotation-test.c
mathops_test_SOURCES = mathops-test.c
tandem_test_SOURCES = tandem-test.c
tandem_test_LDADD = $(top_builddir)/libcelt/libcelt@LIBCELT_SUFFIX@.la
