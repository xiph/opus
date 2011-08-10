AUTOMAKE_OPTIONS = subdir-objects

lib_LTLIBRARIES = libopus.la

SUBDIRS = . libcelt

INCLUDES = -I$(top_srcdir)/libcelt -I$(top_srcdir)/silk -I$(top_srcdir)/silk/float -I$(top_srcdir)/silk/fixed

include celt_sources.mk
include silk_sources.mk
include opus_sources.mk

if FIXED_POINT
SILK_SOURCES += $(SILK_SOURCES_FIXED)
else
SILK_SOURCES += $(SILK_SOURCES_FLOAT)
endif

include celt_headers.txt
include silk_headers.txt
include opus_headers.txt

libopus_la_SOURCES = $(CELT_SOURCES) $(SILK_SOURCES) $(OPUS_SOURCES)

pkginclude_HEADERS = src/opus.h

noinst_HEADERS = $(OPUS_HEAD) $(SILK_HEAD) $(CELT_HEAD)

noinst_PROGRAMS = test_opus test_repacketizer

test_opus_SOURCES = src/test_opus.c

test_opus_LDADD = libopus.la -lm

test_repacketizer_SOURCES = src/test_repacketizer.c

test_repacketizer_LDADD = libopus.la -lm
