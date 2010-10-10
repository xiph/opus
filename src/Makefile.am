
INCLUDES = -I$(top_srcdir)/celt/libcelt/ -I$(top_srcdir)/silk/interface 

lib_LTLIBRARIES = libietfcodec.la
libietfcodec_la_SOURCES = harmony_decoder.c harmony_encoder.c

noinst_HEADERS = harmony_decoder.h harmony_encoder.h

pkginclude_HEADERS = harmony.h

noinst_PROGRAMS = test_harmony
test_harmony_SOURCES = test_harmony.c $(top_srcdir)/silk/test/SKP_debug.c
test_harmony_LDADD = libietfcodec.la $(top_builddir)/celt/libcelt/libcelt0.la $(top_builddir)/silk/libSKP_SILK_SDK.la
