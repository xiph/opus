AUTOMAKE_OPTIONS = subdir-objects

lib_LTLIBRARIES = libopus.la

DIST_SUBDIRS = doc

INCLUDES = -I$(top_srcdir)/include -I$(top_srcdir)/celt -I$(top_srcdir)/silk -I$(top_srcdir)/silk/float -I$(top_srcdir)/silk/fixed

include celt_sources.mk
include silk_sources.mk
include opus_sources.mk

if FIXED_POINT
SILK_SOURCES += $(SILK_SOURCES_FIXED)
else
SILK_SOURCES += $(SILK_SOURCES_FLOAT)
endif

include celt_headers.mk
include silk_headers.mk
include opus_headers.mk

libopus_la_SOURCES = $(CELT_SOURCES) $(SILK_SOURCES) $(OPUS_SOURCES)
libopus_la_LDFLAGS = -no-undefined -version-info @OPUS_LT_CURRENT@:@OPUS_LT_REVISION@:@OPUS_LT_AGE@

pkginclude_HEADERS = include/opus.h include/opus_multistream.h include/opus_types.h include/opus_defines.h

noinst_HEADERS = $(OPUS_HEAD) $(SILK_HEAD) $(CELT_HEAD)

noinst_PROGRAMS = opus_demo repacketizer_demo opus_compare tests/test_opus_api tests/test_opus_encode tests/test_opus_decode celt/tests/test_unit_cwrs32 celt/tests/test_unit_dft celt/tests/test_unit_entropy celt/tests/test_unit_laplace celt/tests/test_unit_mathops celt/tests/test_unit_mdct celt/tests/test_unit_rotation celt/tests/test_unit_types

TESTS = celt/tests/test_unit_types celt/tests/test_unit_mathops celt/tests/test_unit_entropy celt/tests/test_unit_laplace celt/tests/test_unit_dft celt/tests/test_unit_mdct celt/tests/test_unit_rotation celt/tests/test_unit_cwrs32 tests/test_opus_api tests/test_opus_decode tests/test_opus_encode

opus_demo_SOURCES = src/opus_demo.c

opus_demo_LDADD = libopus.la -lm

repacketizer_demo_SOURCES = src/repacketizer_demo.c

repacketizer_demo_LDADD = libopus.la -lm

opus_compare_SOURCES = src/opus_compare.c
opus_compare_LDADD = -lm

tests_test_opus_api_SOURCES = tests/test_opus_api.c tests/test_opus_common.h
tests_test_opus_api_LDADD = libopus.la -lm

tests_test_opus_encode_SOURCES = tests/test_opus_encode.c tests/test_opus_common.h
tests_test_opus_encode_LDADD = libopus.la -lm

tests_test_opus_decode_SOURCES = tests/test_opus_decode.c tests/test_opus_common.h
tests_test_opus_decode_LDADD = libopus.la -lm

celt_tests_test_unit_cwrs32_SOURCES = celt/tests/test_unit_cwrs32.c
celt_tests_test_unit_cwrs32_LDADD = -lm

celt_tests_test_unit_dft_SOURCES = celt/tests/test_unit_dft.c
celt_tests_test_unit_dft_LDADD = -lm

celt_tests_test_unit_entropy_SOURCES = celt/tests/test_unit_entropy.c
celt_tests_test_unit_entropy_LDADD = -lm

celt_tests_test_unit_laplace_SOURCES = celt/tests/test_unit_laplace.c
celt_tests_test_unit_laplace_LDADD = -lm

celt_tests_test_unit_mathops_SOURCES = celt/tests/test_unit_mathops.c
celt_tests_test_unit_mathops_LDADD = -lm

celt_tests_test_unit_mdct_SOURCES = celt/tests/test_unit_mdct.c
celt_tests_test_unit_mdct_LDADD = -lm

celt_tests_test_unit_rotation_SOURCES = celt/tests/test_unit_rotation.c
celt_tests_test_unit_rotation_LDADD = -lm

celt_tests_test_unit_types_SOURCES = celt/tests/test_unit_types.c
celt_tests_test_unit_types_LDADD = -lm

if CUSTOM_MODES
pkginclude_HEADERS += include/opus_custom.h
noinst_PROGRAMS += opus_custom_demo
opus_custom_demo_SOURCES = celt/opus_custom_demo.c
opus_custom_demo_LDADD = libopus.la -lm
endif

EXTRA_DIST = version.mk \
	     opus.pc.in \
             opus-uninstalled.pc.in \
             opus.m4 \
             tests/run_vectors.sh \
             opus.sln \
             celt/celt.vcxproj \
             celt/celt.vcxproj.filters \
             src/opus_demo.vcxproj \
             src/opus.vcxproj \
             src/opus.vcxproj.filters \
             src/opus_demo.vcxproj.filters \
             tests/test_opus_decode.vcxproj.filters \
             tests/test_opus_decode.vcxproj \
             tests/test_opus_encode.vcxproj.filters \
             tests/test_opus_encode.vcxproj \
             tests/test_opus_api.vcxproj.filters \
             tests/test_opus_api.vcxproj \
             silk/float/silk_float.vcxproj.filters \
             silk/float/silk_float.vcxproj \
             silk/fixed/silk_fixed.vcxproj.filters \
             silk/fixed/silk_fixed.vcxproj \
             silk/silk_common.vcxproj \
             silk/silk_common.vcxproj.filters \
             win32/genversion.bat \
             win32/config.h

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = opus.pc

m4datadir = $(datadir)/aclocal
m4data_DATA = opus.m4

# Targets to build and install just the library without the docs
opus check-opus install-opus: export NO_DOXYGEN = 1

opus: all
check-opus: check
install-opus: install


# Or just the docs
docs:
	cd doc && $(MAKE) $(AM_MAKEFLAGS)

install-docs:
	cd doc && $(MAKE) $(AM_MAKEFLAGS) install


# Or everything (by default)
all-local:
	@[ -n "$(NO_DOXYGEN)" ] || cd doc && $(MAKE) $(AM_MAKEFLAGS)

install-data-local:
	@[ -n "$(NO_DOXYGEN)" ] || cd doc && $(MAKE) $(AM_MAKEFLAGS) install

clean-local:
	-cd doc && $(MAKE) $(AM_MAKEFLAGS) clean

uninstall-local:
	cd doc && $(MAKE) $(AM_MAKEFLAGS) uninstall


.PHONY: opus check-opus install-opus docs install-docs
