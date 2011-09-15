AUTOMAKE_OPTIONS = subdir-objects

lib_LTLIBRARIES = libopus.la

SUBDIRS = . celt/tests doc

INCLUDES = -I$(top_srcdir)/celt -I$(top_srcdir)/silk -I$(top_srcdir)/silk/float -I$(top_srcdir)/silk/fixed

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

pkginclude_HEADERS = src/opus.h src/opus_multistream.h celt/opus_types.h celt/opus_defines.h

noinst_HEADERS = $(OPUS_HEAD) $(SILK_HEAD) $(CELT_HEAD)

noinst_PROGRAMS = test_opus test_repacketizer opus_compare

test_opus_SOURCES = src/test_opus.c

test_opus_LDADD = libopus.la -lm

test_repacketizer_SOURCES = src/test_repacketizer.c

test_repacketizer_LDADD = libopus.la -lm

opus_compare_SOURCES = src/opus_compare.c
opus_compare_LDADD = -lm

if CUSTOM_MODES
pkginclude_HEADERS += celt/opus_custom.h
noinst_PROGRAMS += test_opus_custom
test_opus_custom_SOURCES = celt/test_opus_custom.c
test_opus_custom_LDADD = libopus.la -lm
endif

EXTRA_DIST = opus.pc.in opus-uninstalled.pc.in

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = opus.pc
