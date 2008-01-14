# Disable automatic dependency tracking if using other tools than gcc and gmake
#AUTOMAKE_OPTIONS = no-dependencies


#EXTRA_DIST=

#INCLUDES = -I$(top_srcdir)/include -I$(top_builddir)/include -I$(top_builddir) @OGG_CFLAGS@

lib_LTLIBRARIES = libcelt.la

# Sources for compilation in the library
libcelt_la_SOURCES = bands.c celt.c cwrs.c fftwrap.c mdct.c modes.c pitch.c \
	psy.c quant_bands.c quant_pitch.c rate.c smallft.c vq.c

#noinst_HEADERS =

libcelt_la_LDFLAGS = -version-info @CELT_LT_CURRENT@:@CELT_LT_REVISION@:@CELT_LT_AGE@

noinst_HEADERS = arch.h bands.h celt.h cwrs.h fftwrap.h mdct.h modes.h \
	os_support.h pgain_table.h pitch.h psy.h quant_bands.h quant_pitch.h rate.h \
	smallft.h vq.h

noinst_PROGRAMS = testcelt
testcelt_SOURCES = testcelt.c
testcelt_LDADD = $(top_builddir)/libentcode/libentcode.la \
	libcelt.la
INCLUDES = -I$(top_srcdir)/libentcode
libcelt_la_LIBADD = $(top_builddir)/libentcode/libentcode.la
