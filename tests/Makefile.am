INCLUDES = -I$(top_srcdir)/libcelt
METASOURCES = AUTO

TESTS = ectest cwrs32-test cwrs64-test

bin_PROGRAMS = ectest cwrs32-test cwrs64-test

ectest_SOURCES = ectest.c
ectest_LDADD = $(top_builddir)/libcelt/libcelt.la

cwrs32_test_SOURCES = cwrs32-test.c
cwrs32_test_LDADD = $(top_builddir)/libcelt/libcelt.la

cwrs64_test_SOURCES = cwrs64-test.c
cwrs64_test_LDADD = $(top_builddir)/libcelt/libcelt.la
