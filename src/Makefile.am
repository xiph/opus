
INCLUDES = -I$(top_srcdir)/celt/libcelt/ -I$(top_srcdir)/silk/interface 

lib_LTLIBRARIES = libietfcodec.la
libietfcodec_la_SOURCES = opus_decoder.c opus_encoder.c

noinst_HEADERS = opus_decoder.h opus_encoder.h

pkginclude_HEADERS = opus.h

noinst_PROGRAMS = test_opus opus_enc opus_dec

test_opus_SOURCES = test_opus.c $(top_srcdir)/silk/test/SKP_debug.c
test_opus_LDADD = libietfcodec.la $(top_builddir)/celt/libcelt/libcelt0.la $(top_builddir)/silk/libSKP_SILK_SDK.la

opus_enc_SOURCES = opus_enc.c $(top_srcdir)/silk/test/SKP_debug.c
opus_enc_LDADD = libietfcodec.la $(top_builddir)/celt/libcelt/libcelt0.la $(top_builddir)/silk/libSKP_SILK_SDK.la

opus_dec_SOURCES = opus_dec.c $(top_srcdir)/silk/test/SKP_debug.c
opus_dec_LDADD = libietfcodec.la $(top_builddir)/celt/libcelt/libcelt0.la $(top_builddir)/silk/libSKP_SILK_SDK.la
