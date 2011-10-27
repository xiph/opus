AUTOMAKE_OPTIONS = subdir-objects

lib_LTLIBRARIES = libopus.la

SUBDIRS = . celt/tests doc

INCLUDES = -I$(top_srcdir)/include -I$(top_srcdir)/celt -I$(top_srcdir)/silk -I$(top_srcdir)/silk/float -I$(top_srcdir)/silk/fixed

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

pkginclude_HEADERS = include/opus.h include/opus_multistream.h include/opus_types.h include/opus_defines.h

noinst_HEADERS = $(OPUS_HEAD) $(SILK_HEAD) $(CELT_HEAD)

noinst_PROGRAMS = opus_demo repacketizer_demo opus_compare test_opus_api test_opus_encode test_opus_decode

TESTS = test_opus_api test_opus_decode test_opus_encode

opus_demo_SOURCES = src/opus_demo.c

opus_demo_LDADD = libopus.la -lm

repacketizer_demo_SOURCES = src/repacketizer_demo.c

repacketizer_demo_LDADD = libopus.la -lm

opus_compare_SOURCES = src/opus_compare.c
opus_compare_LDADD = -lm

test_opus_api_SOURCES = tests/test_opus_api.c tests/test_opus_common.h
test_opus_api_LDADD = libopus.la -lm

test_opus_encode_SOURCES = tests/test_opus_encode.c tests/test_opus_common.h
test_opus_encode_LDADD = libopus.la -lm

test_opus_decode_SOURCES = tests/test_opus_decode.c tests/test_opus_common.h
test_opus_decode_LDADD = libopus.la -lm

if CUSTOM_MODES
pkginclude_HEADERS += include/opus_custom.h
noinst_PROGRAMS += opus_custom_demo
opus_custom_demo_SOURCES = celt/opus_custom_demo.c
opus_custom_demo_LDADD = libopus.la -lm
endif

EXTRA_DIST = opus.pc.in opus-uninstalled.pc.in

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = opus.pc
