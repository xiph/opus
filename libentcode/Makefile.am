INCLUDES = 
METASOURCES = AUTO
lib_LTLIBRARIES = libentcode.la
libentcode_la_SOURCES = bitrdec.c bitree.c bitrenc.c ecintrin.h entcode.c \
	entdec.c entenc.c laplace.c mfrngdec.c mfrngenc.c probdec.c probenc.c probmod.c
bin_PROGRAMS = ectest
ectest_SOURCES = ectest.c
ectest_LDADD = $(top_builddir)/libentcode/libentcode.la
noinst_HEADERS = bitrdec.h bitree.h bitrenc.h ecintrin.h entcode.h entdec.h \
	entenc.h mfrngcod.h probdec.h probenc.h probmod.h
