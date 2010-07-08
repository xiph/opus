
INCLUDES = -I$(top_srcdir)/celt/libcelt/ -I$(top_srcdir)/silk/interface 

lib_LTLIBRARIES = libietfcodec.la
libietfcodec_la_SOURCES = hybrid_decoder.c hybrid_encoder.c

noinst_HEADERS = hybrid_decoder.h hybrid_encoder.h

pkginclude_HEADERS = hybrid.h

noinst_PROGRAMS = test_hybrid
test_hybrid_SOURCES = test_hybrid.c $(top_srcdir)/silk/test/SKP_debug.c
test_hybrid_LDADD = libietfcodec.la $(top_builddir)/celt/libcelt/libcelt0.la $(top_builddir)/silk/libSKP_SILK_SDK.la
