lib_LTLIBRARIES = libopus.la

INCLUDES = -I$(top_srcdir)/libcelt -I$(top_srcdir)/silk -I$(top_srcdir)/silk/float -I$(top_srcdir)/silk/fixed

include celt_sources.mk
include silk_sources.mk
include opus_sources.mk

libopus_la_SOURCES = $(CELT_SOURCES) $(SILK_SOURCES) $(OPUS_SOURCES)

pkginclude_HEADERS = src/opus.h

noinst_PROGRAMS = test_opus

test_opus_SOURCES = src/test_opus.c

test_opus_LDADD = libopus.la -lm
