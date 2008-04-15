# Disable automatic dependency tracking if using other tools than gcc and gmake
#AUTOMAKE_OPTIONS = no-dependencies


pkginclude_HEADERS = celt.h celt_types.h celt_header.h
EXTRA_DIST= match-test.sh

#INCLUDES = -I$(top_srcdir)/include -I$(top_builddir)/include -I$(top_builddir) @OGG_CFLAGS@

TESTS = match-test.sh

noinst_SCRIPTS = match-test.sh

lib_LTLIBRARIES = libcelt.la

# Sources for compilation in the library
libcelt_la_SOURCES = bands.c celt.c cwrs.c ecintrin.h entcode.c \
	entdec.c entenc.c header.c kfft_single.c kiss_fft.c kiss_fftr.c laplace.c mdct.c \
	modes.c pitch.c psy.c quant_bands.c quant_pitch.c rangedec.c rangeenc.c rate.c \
	vq.c

#noinst_HEADERS =

libcelt_la_LDFLAGS = -version-info @CELT_LT_CURRENT@:@CELT_LT_REVISION@:@CELT_LT_AGE@

noinst_HEADERS = _kiss_fft_guts.h arch.h bands.h fixed_c5x.h fixed_c6x.h \
	cwrs.h ecintrin.h entcode.h entdec.h entenc.h fixed_generic.h \
	kfft_double.h kfft_single.h kiss_fft.h kiss_fftr.h laplace.h mdct.h mfrngcod.h \
	mathops.h modes.h os_support.h pgain_table.h pitch.h psy.h \
	quant_bands.h quant_pitch.h rate.h stack_alloc.h vq.h

noinst_PROGRAMS = testcelt dump_modes
testcelt_SOURCES = testcelt.c
testcelt_LDADD = libcelt.la
INCLUDES = 
#libcelt_la_LIBADD = 

dump_modes_SOURCES = dump_modes.c
dump_modes_LDADD = libcelt.la
